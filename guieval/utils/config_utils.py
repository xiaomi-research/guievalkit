try:
    import flash_attn  # noqa: F401

    hf_attn_implementation = "flash_attention_2"
    print('flash_attn not installed.')
except Exception:
    hf_attn_implementation = "sdpa"

from transformers import (AutoProcessor, AutoTokenizer, AutoModelForCausalLM,
                          Qwen2VLForConditionalGeneration, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration,
                          Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration)


def model_config_handler(model_name):
    if model_name == "agentcpm-gui-8b":
        LLM_CLASS = AutoModelForCausalLM
        attn_implementation = "sdpa"
        TOKENIZER_CLASS = AutoTokenizer
    elif model_name in [
        "qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct", "ui-tars-1.5-7b", "mimo-vl-7b-sft", "mimo-vl-7b-rl",
        "mimo-vl-7b-sft-2508", "mimo-vl-7b-rl-2508", "gui-owl-7b", "gui-owl-32b",
        "ui-venus-navi-7b", "ui-venus-navi-72b"]:
        LLM_CLASS = Qwen2_5_VLForConditionalGeneration
        attn_implementation = hf_attn_implementation
        TOKENIZER_CLASS = AutoProcessor
    elif model_name in [
        "ui-tars-2b-sft", "ui-tars-7b-sft", "ui-tars-7b-dpo", "ui-tars-72b-sft", "ui-tars-72b-dpo"]:
        LLM_CLASS = Qwen2VLForConditionalGeneration
        attn_implementation = hf_attn_implementation
        TOKENIZER_CLASS = AutoProcessor
    elif model_name == "glm-4.1v-9b-thinking":
        LLM_CLASS = Glm4vForConditionalGeneration
        attn_implementation = hf_attn_implementation
        TOKENIZER_CLASS = AutoProcessor
    elif model_name == "glm-4.5v":
        LLM_CLASS = Glm4vMoeForConditionalGeneration
        attn_implementation = hf_attn_implementation
        TOKENIZER_CLASS = AutoProcessor
    elif model_name in ["magicgui-cpt", "magicgui-rft"]:
        LLM_CLASS = Qwen2VLForConditionalGeneration
        attn_implementation = hf_attn_implementation
        TOKENIZER_CLASS = Qwen2VLProcessor
    else:
        raise ValueError(f"Model {model_name} not found.")

    return {
        'tokenizer': TOKENIZER_CLASS,
        'llm': LLM_CLASS,
        'attn': attn_implementation
    }
