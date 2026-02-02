# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def calculate_perplexity(code_snippet):
    model_name = "gpt2"  # Or a code-specific model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    inputs = tokenizer(code_snippet, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])

    return torch.exp(outputs.loss).item()


# Lower scores often suggest AI generation
print(calculate_perplexity("def add(a, b): return a + b"))

# %%

if __name__ == "__main__":
    with open("llm_code.txt") as f:
        llm_code = f.read()
    print(f"Llm code score: {calculate_perplexity(llm_code)}")

# %%
