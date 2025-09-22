import os
import torch
import traceback

from vllm import LLM
from guieval.utils.config_utils import model_config_handler

import guieval.models.run_agentcpm_gui as run_agentcpm_gui
import guieval.models.run_glm4p1v as run_glm4p1v
import guieval.models.run_glm4p5v as run_glm4p5v
import guieval.models.run_mimo_vl as run_mimo_vl
import guieval.models.run_qwen2p5_vl as run_qwen2p5_vl
import guieval.models.run_uitars_1p5 as run_uitars_1p5
import guieval.models.run_uitars_1 as run_uitars_1
import guieval.models.run_gui_owl as run_gui_owl
import guieval.models.run_uivenus_navi as run_uivenus_navi
import guieval.models.run_magicgui as run_magicgui
import guieval.utils.global_tokenizer as global_tokenizer


_llm = None


def init_llm(model_name, model_path, device_queue, use_vllm=True):
    global _llm
    model_config = model_config_handler(model_name)
    TOKENIZER_CLASS, LLM_CLASS, attn_implementation = (
        model_config["tokenizer"],
        model_config["llm"],
        model_config["attn"]
    )

    global_tokenizer._tokenizer = TOKENIZER_CLASS.from_pretrained(model_path, trust_remote_code=True)

    device_str = device_queue.get()

    tp = len(device_str.split(','))
    print('Start LLM on devices: ', device_str)
    if use_vllm:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        try:
            _llm = LLM(
                model=model_path,
                tensor_parallel_size=tp,
                gpu_memory_utilization=0.9,
                trust_remote_code=True,
                limit_mm_per_prompt={"image": 10},
            )
        except Exception:
            print('Traceback: ', traceback.format_exc())
    else:
        llm = LLM_CLASS.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation,
        )
        _llm = llm.to(f"cuda:{device_str}")


def shut_llm():
    global _llm
    del _llm
    return "Shutdown LLM..."


def dummy_task():
    return


def prepare_inputs(model_name, episode, episode_dir, episode_file, subset, dataset, use_vllm, no_think):
    if model_name == "agentcpm-gui-8b":
        return run_agentcpm_gui.prepare_task_inputs(episode, episode_dir, episode_file, subset, dataset, use_vllm)
    elif model_name in [
        "qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct"]:
        return run_qwen2p5_vl.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset, global_tokenizer._tokenizer, use_vllm)
    elif model_name == "ui-tars-1.5-7b":
        return run_uitars_1p5.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset, global_tokenizer._tokenizer, use_vllm)
    elif model_name in [
        "ui-tars-2b-sft", "ui-tars-7b-sft", "ui-tars-7b-dpo", "ui-tars-72b-sft", "ui-tars-72b-dpo"]:
        return run_uitars_1.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset, global_tokenizer._tokenizer, use_vllm)
    elif model_name in ["mimo-vl-7b-sft", "mimo-vl-7b-rl", "mimo-vl-7b-sft-2508", "mimo-vl-7b-rl-2508"]:
        return run_mimo_vl.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset, global_tokenizer._tokenizer, use_vllm, no_think)
    elif model_name == "glm-4.1v-9b-thinking":
        return run_glm4p1v.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset, global_tokenizer._tokenizer, use_vllm)
    elif model_name == "glm-4.5v":
        return run_glm4p5v.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset, global_tokenizer._tokenizer, use_vllm)
    elif model_name in ["gui-owl-7b", "gui-owl-32b"]:
        max_px, min_px = 10035200, 3136
        global_tokenizer._tokenizer.image_processor.max_pixels = max_px
        global_tokenizer._tokenizer.image_processor.min_pixels = min_px
        return run_gui_owl.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset,
            global_tokenizer._tokenizer, use_vllm, enable_think=not (no_think))
    elif model_name in ["ui-venus-navi-7b", "ui-venus-navi-72b"]:
        if model_name == "ui-venus-navi-7b":
            max_px, min_px = 937664, 830000
        else:
            max_px, min_px = 12845056, 3136
        global_tokenizer._tokenizer.image_processor.max_pixels = max_px
        global_tokenizer._tokenizer.image_processor.min_pixels = min_px
        return run_uivenus_navi.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset, global_tokenizer._tokenizer, use_vllm)
    elif model_name in ["magicgui-cpt", "magicgui-rft"]:
        global_tokenizer._tokenizer.image_processor.max_pixels = run_magicgui.MAX_PIXELS
        global_tokenizer._tokenizer.image_processor.min_pixels = run_magicgui.MIN_PIXELS
        return run_magicgui.prepare_task_inputs(
            episode, episode_dir, episode_file, subset, dataset, use_vllm)


def run_step_batch(model_name, batch_tasks, use_vllm):
    global _llm

    try:
        if model_name == "agentcpm-gui-8b":
            results = run_agentcpm_gui.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name in [
            "qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct", "qwen2.5-vl-32b-instruct", "qwen2.5-vl-72b-instruct"]:
            results = run_qwen2p5_vl.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name == "ui-tars-1.5-7b":
            results = run_uitars_1p5.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name in [
            "ui-tars-2b-sft", "ui-tars-7b-sft", "ui-tars-7b-dpo", "ui-tars-72b-sft", "ui-tars-72b-dpo"]:
            results = run_uitars_1.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name in ["mimo-vl-7b-sft", "mimo-vl-7b-rl", "mimo-vl-7b-sft-2508", "mimo-vl-7b-rl-2508"]:
            results = run_mimo_vl.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name == "glm-4.1v-9b-thinking":
            results = run_glm4p1v.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name == "glm-4.5v":
            results = run_glm4p5v.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name in ["gui-owl-7b", "gui-owl-32b"]:
            results = run_gui_owl.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name in ["ui-venus-navi-7b", "ui-venus-navi-72b"]:
            results = run_uivenus_navi.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        elif model_name in ["magicgui-cpt", "magicgui-rft"]:
            results = run_magicgui.run_task_batch(_llm, global_tokenizer._tokenizer, batch_tasks, use_vllm)
        else:
            results = list()
    except Exception as e:
        print(f"Batch inference error: {e}")
        import traceback
        print('traceback:', traceback.format_exc())
        results = [None] * len(batch_tasks)
    return results
