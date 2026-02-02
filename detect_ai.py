"""Detect AI-generated code in files using LLM perplexity."""

import argparse
import dataclasses as dcls
import os
import statistics
import sys
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODELS = ["gpt2"]
DEFAULT_THRESHOLD = 12.0
DEFAULT_STRATEGY = "min"
DEFAULT_MAX_LENGTH = 1024
DEFAULT_MIN_TOKENS = 5


@dcls.dataclass(frozen=True)
class DetectionConfig:
    """Configuration for detection thresholds and limits."""

    max_length: int
    min_tokens: int
    strategy: str
    threshold: float


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


def parse_model_list(values: Iterable[str]) -> List[str]:
    """Normalize comma-separated model IDs into a flat list."""
    models: List[str] = []
    for value in values:
        if not value:
            continue
        for item in value.split(","):
            item = item.strip()
            if item:
                models.append(item)
    return models or DEFAULT_MODELS


def resolve_device(device: Optional[str]) -> torch.device:
    """Resolve the requested torch device or pick a sensible default."""
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def effective_max_length(tokenizer, requested: int) -> int:
    """Clamp requested max length to the tokenizer's model limit when set."""
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len and max_len < 1_000_000:
        return min(requested, max_len)
    return requested


def load_model(
    model_id: str, device: torch.device
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load a causal LM and tokenizer onto the requested device."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(device)
    model.eval()
    return model, tokenizer


def get_perplexity(
    text: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    *,
    max_length: int,
    min_tokens: int,
) -> float:
    """Compute perplexity for text using a preloaded model/tokenizer."""
    max_length = effective_max_length(tokenizer, max_length)
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    if inputs["input_ids"].size(1) < min_tokens:
        return float("inf")

    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    return torch.exp(outputs.loss).item()


def aggregate(scores: List[float], strategy: str) -> float:
    """Combine per-model perplexities using the requested strategy."""
    valid = [score for score in scores if score != float("inf")]
    if not valid:
        return float("inf")
    if strategy == "min":
        return min(valid)
    if strategy == "mean":
        return sum(valid) / len(valid)
    if strategy == "median":
        return statistics.median(valid)
    raise ValueError(f"Unknown strategy: {strategy}")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments and environment defaults."""
    parser = argparse.ArgumentParser(
        description="Detect AI-generated code via LLM perplexity."
    )
    parser.add_argument("files", nargs="*", help="Files to analyze.")
    parser.add_argument(
        "--models",
        default=_env_str("AI_DETECT_MODELS", ""),
        help="Comma-separated list of model IDs (or set AI_DETECT_MODELS).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=_env_float("AI_DETECT_THRESHOLD", DEFAULT_THRESHOLD),
        help="Perplexity threshold under which code is flagged.",
    )
    parser.add_argument(
        "--strategy",
        choices=("min", "mean", "median"),
        default=_env_str("AI_DETECT_STRATEGY", DEFAULT_STRATEGY),
        help="How to combine multi-model perplexities.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=_env_int("AI_DETECT_MAX_LENGTH", DEFAULT_MAX_LENGTH),
        help="Max tokens to evaluate per file.",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=_env_int("AI_DETECT_MIN_TOKENS", DEFAULT_MIN_TOKENS),
        help="Skip files shorter than this token length.",
    )
    parser.add_argument(
        "--device",
        default=_env_str("AI_DETECT_DEVICE", ""),
        help="Torch device override (e.g. cpu, cuda).",
    )
    return parser.parse_args()


def compute_scores(
    file_path: str,
    model_cache: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]],
    *,
    max_length: int,
    min_tokens: int,
) -> List[float]:
    """Compute and print per-model perplexity scores for a file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    scores: List[float] = []
    for model_id, (model, tokenizer) in model_cache.items():
        score = get_perplexity(
            content,
            model,
            tokenizer,
            max_length=max_length,
            min_tokens=min_tokens,
        )
        scores.append(score)
        if score == float("inf"):
            score_display = "skipped"
        else:
            score_display = f"{score:.2f}"
        print(f"File: {file_path} | Model: {model_id} | Perplexity: {score_display}")

    return scores


def analyze_file(
    file_path: str,
    model_cache: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]],
    *,
    config: DetectionConfig,
) -> bool:
    """Analyze a file and return True if it should be flagged."""
    scores = compute_scores(
        file_path,
        model_cache,
        max_length=config.max_length,
        min_tokens=config.min_tokens,
    )

    combined = aggregate(scores, config.strategy)
    if combined != float("inf"):
        print(
            f"File: {file_path} | Combined ({config.strategy}) perplexity: {combined:.2f}"
        )

    if combined < config.threshold:
        print(f"⚠️ WARNING: {file_path} looks highly predictable (Potential AI).")
        return True

    return False


def main() -> int:
    """Run the CLI entrypoint."""
    args = parse_args()
    files = args.files
    if not files:
        print("No files provided.")
        return 0

    models = parse_model_list([args.models])
    device = resolve_device(args.device or None)
    model_cache: Dict[str, Tuple[AutoModelForCausalLM, AutoTokenizer]] = {}
    config = DetectionConfig(
        max_length=args.max_length,
        min_tokens=args.min_tokens,
        strategy=args.strategy,
        threshold=args.threshold,
    )
    flagged = False

    for model_id in models:
        model_cache[model_id] = load_model(model_id, device)

    for file_path in files:
        if analyze_file(
            file_path,
            model_cache,
            config=config,
        ):
            flagged = True

    if flagged:
        return 1  # Exit with 1 to fail the build when AI code is detected
    return 0


if __name__ == "__main__":
    sys.exit(main())
