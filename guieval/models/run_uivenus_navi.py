import json
import numpy as np
import os
import re

from PIL import Image
from vllm import SamplingParams
from typing import Optional, Tuple
from qwen_vl_utils import process_vision_info, smart_resize

from guieval.utils.action_type import ActionType
from guieval.utils.action_utils import is_tap_action, get_direction


USER_PROMPT = (
    "\n"
    "**You are a GUI Agent.**  \n"
    "Your task is to analyze a given user task, review current screenshot and previous actions, "
    "and determine the next action to complete the task.\n"
    "\n"
    "### User Task\n"
    "{user_task}\n"
    "\n"
    "### Previous Actions\n"
    "{previous_actions}\n"
    "\n"
    "### Available Actions\n"
    "You may execute one of the following functions:\n"
    "Click(box=(x1, y1))\n"
    "Drag(start=(x1, y1), end=(x2, y2))\n"
    "Scroll(start=(x1, y1), end=(x2, y2), direction='down/up/right/left')\n"
    "Type(content='')\n"
    # "Launch(app='')\n"
    "Wait()\n"
    "Finished(content='')\n"
    "CallUser(content='')\n"
    "LongPress(box=(x1, y1))\n"
    "PressBack()\n"
    "PressHome()\n"
    "PressEnter()\n"
    "PressRecent()\n"
    "\n"
    "### Instruction\n"
    "- Make sure you understand the task goal to avoid wrong actions.\n"
    "- Make sure you carefully examine the the current screenshot. "
    "Sometimes the summarized history might not be reliable, over-claiming some effects.\n"
    "- For requests that are questions (or chat messages), "
    "remember to use the `CallUser` action to reply to user explicitly before finishing! "
    "Then, after you have replied, use the Finished action if the goal is achieved.\n"
    "- Consider exploring the screen by using the `scroll` action "
    "with different directions to reveal additional content.\n"
    "- To copy some text: first select the exact text you want to copy, "
    "which usually also brings up the text selection bar, then click the `copy` button in bar.\n"
    "- To paste text into a text box, first long press the text box, "
    "then usually the text selection bar will appear with a `paste` button in it.\n"
    "- You first thinks about the reasoning process in the mind, then provide the action. "
    "The reasoning and action are enclosed in <think></think> and <action></action> tags respectively. "
    "After providing action, summarize your action in <conclusion></conclusion> tags\n"
    )


def parse_coordinates(coord_str: str) -> Optional[Tuple[float, float]]:
    if not coord_str:
        return None, None

    coord_str_clean = coord_str.replace(" ", "")
    match = re.match(r"\(([\d.]+),([\d.]+)\)", coord_str_clean)
    if match:
        return float(match.group(1)), float(match.group(2))

    match = re.match(r"\(\s*([\d.]+)\s*,\s*([\d.]+)\s*\)", coord_str)
    if match:
        return float(match.group(1)), float(match.group(2))

    return None, None


def _split_parameters(params_str: str) -> list:
    param_parts = []
    current_part = ""

    in_quotes = False
    quote_char = None
    bracket_level = 0

    for char in params_str:
        if char in ['"', "'"] and not in_quotes:
            in_quotes = True
            quote_char = char
        elif char == quote_char and in_quotes:
            in_quotes = False
            quote_char = None
        elif not in_quotes:
            if char == '(':
                bracket_level += 1
            elif char == ')':
                bracket_level -= 1
            elif char == ',' and bracket_level == 0:
                param_parts.append(current_part.strip())
                current_part = ""
                continue
        current_part += char

    if current_part.strip():
        param_parts.append(current_part.strip())

    return param_parts


def parse_answer(action_str: str):
    pattern = r"^(\w+)\((.*)\)$"
    match = re.match(pattern, action_str.strip(), re.DOTALL)
    if not match:
        raise ValueError(f"Invalid action_str format: {action_str}")

    action_type = match.group(1)
    params_str = match.group(2).strip()
    params = {}

    if params_str:
        try:
            param_pairs = _split_parameters(params_str)
            for pair in param_pairs:
                if '=' in pair:
                    key, value = pair.split("=", 1)
                    value = value.strip("'").strip()
                    params[key.strip()] = value
                else:
                    params[pair.strip()] = None
        except Exception as e:
            print(f"Answer parse error: {e}")
    return action_type, params


def aitw_2_uivenus_action(aitw_action, resized_height, resized_width):

    ex_action_type = aitw_action['result_action_type']

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            click_y_min = int(touch_yx[0] * resized_height)
            click_x_min = int(touch_yx[1] * resized_width)
            click_y_max = int(lift_yx[0] * resized_height)
            click_x_max = int(lift_yx[1] * resized_width)
            return f"""Click(box=({(click_x_min + click_x_max) // 2}, {(click_y_min + click_y_max) // 2}))"""
        else:
            touch_xy_new = [touch_yx[1], touch_yx[0]]
            lift_xy_new = [lift_yx[1], lift_yx[0]]
            direction = get_direction(touch_xy_new, lift_xy_new)
            x1 = int(touch_yx[1] * resized_width)
            y1 = int(touch_yx[0] * resized_height)
            x2 = int(lift_yx[1] * resized_width)
            y2 = int(lift_yx[0] * resized_height)

            return f"""Scroll(start=({x1}, {y1}), end=({x2}, {y2}), direction='{direction}')"""

    elif ex_action_type == ActionType.PRESS_BACK:
        return f"""PressBack()"""

    elif ex_action_type == ActionType.PRESS_HOME:
        return f"""PressHome()"""

    elif ex_action_type == ActionType.PRESS_ENTER:
        return f"""PressEnter()"""

    elif ex_action_type == ActionType.TYPE:
        return f"""Type(content='{aitw_action['result_action_text']}')"""

    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        return f"""Finished(content='')"""

    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        return f"""Finished(content='')"""

    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        click_y_min = int(touch_yx[0] * resized_height)
        click_x_min = int(touch_yx[1] * resized_width)
        click_y_max = int(lift_yx[0] * resized_height)
        click_x_max = int(lift_yx[1] * resized_width)
        return f"""LongPress(box=({(click_x_min + click_x_max) // 2}, {(click_y_min + click_y_max) // 2}))"""

    elif ex_action_type == ActionType.NO_ACTION:
        return f"""Wait()"""

    elif ex_action_type == ActionType.OPEN_APP:
        return f"""Launch(app='{aitw_action['result_action_app_name']}')"""

    else:
        print('AiTW Action: ', aitw_action)
        raise NotImplementedError

    return ""


def uivenus_2_minicpm(output_text, size_params):

    answer_text = output_text.split('<action>')[1].split('</action>')[0].strip('\n')
    resized_width = size_params["resized_width"]
    resized_height = size_params["resized_height"]

    try:
        action_name, action_params = parse_answer(answer_text)
    except Exception as e:
        print(f"Error, JSON is NOT valid: {e}")
        return {}

    if action_name == "Click":
        x, y = parse_coordinates(action_params.get("box", ""))
        x = x / resized_width * 1000
        y = y / resized_height * 1000
        return {"POINT": [int(x), int(y)]}

    elif action_name == "LongPress":
        x, y = parse_coordinates(action_params.get("box", ""))
        x = x / resized_width * 1000
        y = y / resized_height * 1000
        time = 1000
        return {"POINT": [int(x), int(y)], "duration": time}

    elif action_name == "Scroll" or action_name == "Drag":
        x1, y1 = parse_coordinates(action_params.get("start", ""))
        x2, y2 = parse_coordinates(action_params.get("end", ""))
        direction = action_params.get("direction")
        if direction and None in (x1, y1, x2, y2):
            return {"POINT": [500, 500], "to": direction}

        x1 = x1 / resized_width * 1000
        y1 = y1 / resized_height * 1000
        x2 = x2 / resized_width * 1000
        y2 = y2 / resized_height * 1000
        if None not in (x1, y1, x2, y2):
            direction = get_direction([x1, y1], [x2, y2])
        return {"POINT": [int(x1), int(y1)], "to": direction}

    elif action_name == "Type":
        content = action_params.get("content", "")
        return {"TYPE": content}

    elif action_name == "PressBack":
        return {"PRESS": "BACK"}

    elif action_name == "PressHome":
        return {"PRESS": "HOME"}

    elif action_name == "PressEnter":
        return {"PRESS": "ENTER"}

    elif action_name == "Finished":
        return {"STATUS": "finish"}

    elif action_name == "Wait":
        return {"duration": 1000}

    elif action_name == "Launch":
        app = action_params.get("app", "")
        return {"OPEN_APP": app}

    print("Error, unrecognized action.")
    return {}


def prepare_task_input(step, image_path, history_actions, data_name, _tokenizer, use_vllm):
    query = step['instruction']
    if data_name == 'androidcontrol_low':
        query = step["low_instruction"]
    image = Image.open(image_path)
    images = [image]
    resized_height, resized_width = smart_resize(
    image.height, image.width,
    min_pixels=_tokenizer.image_processor.min_pixels,
    max_pixels=_tokenizer.image_processor.max_pixels)

    if len(history_actions) == 0:
        history_str = ""
    else:
        recent_history = history_actions
        history_entries = [
            (
            f"Step {i}: <think></think>"
            f"<action>{aitw_2_uivenus_action(aitw_action, resized_height, resized_width)}</action>")
            for i, aitw_action in enumerate(recent_history)
        ]
        history_str = "\n".join(history_entries)

    problem = USER_PROMPT.format(user_task=query, previous_actions=history_str)

    message = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": problem},
                    {
                        "type": "image",
                        "image": image_path,
                        "min_pixels": _tokenizer.image_processor.min_pixels,
                        "max_pixels": _tokenizer.image_processor.max_pixels,
                    }
                ],
            },
    ]

    return step, message, images


def prepare_task_inputs(episode, episode_dir, episode_file, subset, dataset, _tokenizer, use_vllm):

    res = []

    for index, step in enumerate(episode):
        step["category"] = subset
        history = []
        for prev_step in episode[:index]:
            image_path = os.path.join(episode_dir, episode_file, f"{episode_file}_{prev_step['step_id']}.jpeg")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".jpeg", ".png")
                if not os.path.exists(image_path):
                    image_path = image_path.replace(".png", ".jpg")
                    if not os.path.exists(image_path):
                        image_path = prev_step['image_path']
            history_action = {"instruction": prev_step['instruction'],
                               "observation": prev_step.get('observation', ''),
                              "result_action_type": prev_step['result_action_type'],
                              "result_action_text": prev_step['result_action_text'],
                              "result_touch_yx": prev_step['result_touch_yx'],
                              "result_lift_yx": prev_step['result_lift_yx'],
                              "low_instruction": prev_step.get("low_instruction", ''),
                              "image_path": image_path,
                              "result_action_app_name": prev_step.get('result_action_app_name', ''),
                              "bbox": prev_step.get('bbox', ''),
                              "description": prev_step.get('description', '')
                              }
            history.append(history_action)
        image_path = os.path.join(episode_dir, episode_file, f"{episode_file}_{step['step_id']}.jpeg")
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpeg", ".png")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".png", ".jpg")
                if not os.path.exists(image_path):
                    image_path = step['image_path']

        res.append(prepare_task_input(step, image_path, history, dataset, _tokenizer, use_vllm))

    return res


def run_task_batch(_llm, _tokenizer, batch_tasks, use_vllm):
    batch_steps = []
    batch_inputs = []
    batch_images = []
    for step, messages, images in batch_tasks:
        if use_vllm:
            text_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info([messages])
            batch_inputs.append({"prompt": text_prompt, "multi_modal_data": {"image": image_inputs}})
        else:
            text_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append(text_prompt)

        batch_steps.append(step)
        batch_images.append(images)

    if use_vllm:
        sampling_params = SamplingParams(
            max_tokens=2048,
            temperature=0,
            top_p=1.0,
            top_k=-1,
            repetition_penalty=1.05,
            n=1,
            stop_token_ids=[]
        )
        results = _llm.generate(batch_inputs, sampling_params, use_tqdm=False)
        predict_str = [result.outputs[0].text for result in results]
    else:
        generation_params = {
            'do_sample': True, 'top_p': 0.01, 'top_k': 1, 'temperature': 0.01, 'repetition_penalty': 1.0}
        inputs = _tokenizer(text=batch_inputs, images=batch_images, padding=True, return_tensors="pt")
        inputs = inputs.to(_llm.device)
        generated_ids = _llm.generate(**inputs, max_new_tokens=2048, **generation_params)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        predict_str = _tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    res_steps = []
    for step, str_res, images in zip(batch_steps, predict_str, batch_images):
        original_width, original_height = images[0].size
        resized_height, resized_width = smart_resize(
            original_height, original_width,
            min_pixels=_tokenizer.image_processor.min_pixels, max_pixels=_tokenizer.image_processor.max_pixels)
        size_params = {
            'original_width': original_width,
            'original_height': original_height,
            'resized_width': resized_width,
            'resized_height': resized_height,
        }
        try:
            step['pred'] = uivenus_2_minicpm(str_res, size_params)
        except Exception:
            print("Error, JSON is NOT valid.")
            step['pred'] = {}
        res_steps.append(step)
    return res_steps
