try:
    import flash_attn  # noqa: F401
    hf_attn_implementation = "flash_attention_2"
except Exception:
    hf_attn_implementation = "sdpa"

import importlib.resources as res

from pydantic import BaseModel
from typing import Any
from transformers import (AutoProcessor, AutoTokenizer, AutoModelForCausalLM,
                          Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration,
                          Qwen3VLForConditionalGeneration,
                          Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration)

CONFIG_BASE = res.files('config')


class ModelConfig(BaseModel):
    llm_class: Any
    tokenizer_class: Any
    attn_implementation: str | Any


MODEL_CONFIGS: dict[tuple[str], ModelConfig] = {
    ("agentcpm-gui-8b", ): ModelConfig(
        llm_class=AutoModelForCausalLM,
        tokenizer_class=AutoTokenizer,
        attn_implementation="sdpa"
    ),
    ("qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct",
     "ui-tars-1.5-7b",
     "mimo-vl-7b-sft", "mimo-vl-7b-rl", "mimo-vl-7b-sft-2508", "mimo-vl-7b-rl-2508",
     "gui-owl-7b", "gui-owl-32b",
     "ui-venus-navi-7b", "ui-venus-navi-72b"): ModelConfig(
        llm_class=Qwen2_5_VLForConditionalGeneration,
        tokenizer_class=AutoProcessor,
        attn_implementation=hf_attn_implementation
    ),
    ("qwen3-vl-4b-instruct", "qwen3-vl-4b-thinking", "qwen3-vl-8b-instruct", "qwen3-vl-8b-thinking"): ModelConfig(
        llm_class=Qwen3VLForConditionalGeneration,
        tokenizer_class=AutoProcessor,
        attn_implementation=hf_attn_implementation
    ),
    ("ui-tars-2b-sft", "ui-tars-7b-sft", "ui-tars-7b-dpo", "ui-tars-72b-sft", "ui-tars-72b-dpo"): ModelConfig(
        llm_class=Qwen2VLForConditionalGeneration,
        tokenizer_class=AutoProcessor,
        attn_implementation=hf_attn_implementation
    ),
    ("glm-4.1v-9b-thinking", ): ModelConfig(
        llm_class=Glm4vForConditionalGeneration,
        tokenizer_class=AutoProcessor,
        attn_implementation=hf_attn_implementation
    ),
    ("glm-4.5v", ): ModelConfig(
        llm_class=Glm4vMoeForConditionalGeneration,
        tokenizer_class=AutoProcessor,
        attn_implementation=hf_attn_implementation
    ),
    ("magicgui-cpt", "magicgui-rft"): ModelConfig(
        llm_class=Qwen2VLForConditionalGeneration,
        tokenizer_class=Qwen2VLProcessor,
        attn_implementation=hf_attn_implementation
    ),
}


def model_config_handler(model_name: str) -> ModelConfig:
    '''
    Get the model config for a given model name.

    If the model name is not found, a ValueError will be raised.
    '''
    for _names, _config in MODEL_CONFIGS.items():
        if model_name in _names:
            return _config
    else:
        raise ValueError(f"Model {model_name} not found in {MODEL_CONFIGS}")
