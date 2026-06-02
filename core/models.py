"""
core/models.py
--------------
Model catalogue for each supported LLM provider.
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelInfo:
    id: str
    display_name: str
    context_window: int   # tokens
    notes: str = ""


PROVIDER_MODELS: Dict[str, List[ModelInfo]] = {
    "openai": [
        ModelInfo("gpt-4o",          "GPT-4o",           128_000, "Best reasoning"),
        ModelInfo("gpt-4o-mini",     "GPT-4o Mini",       128_000, "Fast & cheap"),
        ModelInfo("gpt-4-turbo",     "GPT-4 Turbo",       128_000, "High quality"),
        ModelInfo("gpt-3.5-turbo",   "GPT-3.5 Turbo",     16_385,  "Budget option"),
    ],
    "cohere": [
        ModelInfo(
            "command-a-03-2025",
            "Command A",
            256_000,
            "Latest Cohere model"
        ),
    ],
    "huggingface": [
        ModelInfo("meta-llama/Llama-3.3-70B-Instruct",    "Llama 3.3 70B",   128_000, "Strong open model"),
        ModelInfo("mistralai/Mistral-7B-Instruct-v0.3",   "Mistral 7B v0.3",  32_768, "Lightweight"),
        ModelInfo("Qwen/Qwen2.5-72B-Instruct",            "Qwen 2.5 72B",    131_072, "Excellent reasoning"),
        ModelInfo("microsoft/Phi-3.5-mini-instruct",      "Phi-3.5 Mini",    128_000, "Very fast"),
    ],
}

PROVIDER_DISPLAY_NAMES = {
    "openai": "OpenAI",
    "cohere": "Cohere",
    "huggingface": "HuggingFace (free)",
}

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "cohere": "command-a-03-2025",
    "huggingface": "meta-llama/Llama-3.3-70B-Instruct",
}


def model_options(provider: str) -> List[str]:
    return [m.id for m in PROVIDER_MODELS.get(provider, [])]


def model_display_names(provider: str) -> Dict[str, str]:
    return {m.id: m.display_name for m in PROVIDER_MODELS.get(provider, [])}
