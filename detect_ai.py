"""Detect AI-generated code in files using LLM perplexity."""

import dataclasses as dcls
import os
import statistics
import sys
import typing
from argparse import ArgumentParser, Namespace
from collections.abc import Iterable

import torch
from torch import cuda
from torch import device as Device
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


def parse_model_list(values: Iterable[str]) -> list[str]:
    """Normalize comma-separated model IDs into a flat list."""
    models: list[str] = []
    for value in values:
        if not value:
            continue

        for item in value.split(","):
            if item := item.strip():
                models.append(item)
    return models or DEFAULT_MODELS


def resolve_device(device: str | Device | None) -> Device:
    """Resolve the requested torch device or pick a sensible default."""

    match device:
        case str():
            return Device(device)
        case Device():
            return device
        case None:
            device = "cuda" if cuda.is_available() else "cpu"
            return Device(device)
        case _:
            raise RuntimeError("Unreachable.")


def effective_max_length(tokenizer: AutoTokenizer, requested: int) -> int:
    """Clamp requested max length to the tokenizer's model limit when set."""
    max_len = getattr(tokenizer, "model_max_length", None)
    if max_len and max_len < 1_000_000:
        return min(requested, max_len)
    else:
        return requested


@dcls.dataclass
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


@typing.no_type_check
def load_model(model_id: str, device: Device | None = None) -> LoadedModel:
    """Load a causal LM and tokenizer onto the requested device."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.to(resolve_device(device))
    model.eval()
    return LoadedModel(model, tokenizer)


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


def aggregate_scores(scores: list[float], strategy: str) -> float:
    """Combine per-model perplexities using the requested strategy."""

    if not (valid := [score for score in scores if score != float("inf")]):
        return float("inf")

    elif strategy == "min":
        return min(valid)

    elif strategy == "mean":
        return sum(valid) / len(valid)

    elif strategy == "median":
        return statistics.median(valid)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def parse_args() -> Namespace:
    """Parse CLI arguments and environment defaults."""
    parser = ArgumentParser(description="Detect AI-generated code via LLM perplexity.")
    _ = parser.add_argument("files", nargs="*", help="Files to analyze.")
    _ = parser.add_argument(
        "--models",
        default=str(os.getenv("AI_DETECT_MODELS") or ""),
        help="Comma-separated list of model IDs (or set AI_DETECT_MODELS).",
    )
    _ = parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.getenv("AI_DETECT_THRESHOLD") or DEFAULT_THRESHOLD),
        help="Perplexity threshold under which code is flagged.",
    )
    _ = parser.add_argument(
        "--strategy",
        choices=("min", "mean", "median"),
        default=str(os.getenv("AI_DETECT_STRATEGY") or DEFAULT_STRATEGY),
        help="How to combine multi-model perplexities.",
    )
    _ = parser.add_argument(
        "--max-length",
        type=int,
        default=int(os.getenv("AI_DETECT_MAX_LENGTH") or DEFAULT_MAX_LENGTH),
        help="Max tokens to evaluate per file.",
    )
    _ = parser.add_argument(
        "--min-tokens",
        type=int,
        default=int(os.getenv("AI_DETECT_MIN_TOKENS") or DEFAULT_MIN_TOKENS),
        help="Skip files shorter than this token length.",
    )
    _ = parser.add_argument(
        "--device",
        default=str(os.getenv("AI_DETECT_DEVICE") or ""),
        help="Torch device override (e.g. cpu, cuda).",
    )
    return parser.parse_args()


def compute_scores(
    file_path: str,
    model_cache: dict[str, LoadedModel],
    *,
    max_length: int,
    min_tokens: int,
) -> dict[str, float]:
    """Compute and print per-model perplexity scores for a file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    scores: dict[str, float] = {}
    for model_id, loaded in model_cache.items():
        score = get_perplexity(
            content,
            loaded.model,
            loaded.tokenizer,
            max_length=max_length,
            min_tokens=min_tokens,
        )
        scores[model_id] = score
        score_display = "skipped" if score == float("inf") else f"{score:.2f}"
        print(f"File: {file_path} | Model: {model_id} | Perplexity: {score_display}")

    return scores


def analyze_file(
    file_path: str,
    model_cache: dict[str, LoadedModel],
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

    combined = aggregate_scores(list(scores.values()), config.strategy)

    if combined != float("inf"):
        print(
            f"File: {file_path} | Combined ({config.strategy}) perplexity: {combined:.2f}"
        )

    if combined < config.threshold:
        print(f"⚠️ WARNING: {file_path} looks highly predictable (Potential AI).")
        return True
    else:
        return False


def load_requested_models(models: list[str], device: str | Device | None = None):
    model_cache: dict[str, LoadedModel] = {}
    for model_id in models:
        model_cache[model_id] = load_model(model_id, device)
    return model_cache


def main() -> int:
    """Run the CLI entrypoint."""
    args = parse_args()

    if not (files := args.files):
        print("No files provided.")
        return 0

    models = parse_model_list([args.models])
    resolve_device(args.device or None)
    config = DetectionConfig(
        max_length=args.max_length,
        min_tokens=args.min_tokens,
        strategy=args.strategy,
        threshold=args.threshold,
    )

    model_cache = load_requested_models(models)

    analysis_results = [
        analyze_file(path, model_cache, config=config) for path in files
    ]

    # Exit with 1 to fail the build when AI code is detected.
    return int(any(analysis_results))


if __name__ == "__main__":
    sys.exit(main())
