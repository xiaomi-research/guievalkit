import json
import numpy as np
import os

from qwen_vl_utils import smart_resize
from vllm import SamplingParams
from PIL import Image

from guieval.utils.action_type import ActionType
from guieval.utils.action_utils import is_tap_action, get_direction

SYSTEM_PROMPT = "You are a helpful assistant."

USER_QUERY_TEMPLATE = (
    "You are a GUI agent. You will be provided with a screenshot, a goal, and your action history. "
    "You need to perform the next action to complete the task.\n\n"
    "## Action Space\n"
    "{action_space}\n\n"
    "## Goal\n"
    "{goal}\n\n"
    "## Previous Actions\n"
    "{previous_actions}\n\n"
    "Now, output the next action in json format [{{\"action\": \"{{action_name}}\"}}, ...]."
)

ANDROIDCONTROL_LOW_TEMPLATE = (
    "You are a GUI agent. You will be provided with a screenshot, a goal, your action history, "
    "and an instruction for your next action. "
    "You need to perform the next action to complete the task.\n"
    "\n"
    "## Action Space\n"
    "{action_space}\n"
    "\n"
    "## Goal\n"
    "{goal}\n"
    "\n"
    "## Previous Actions\n"
    "{previous_actions}\n"
    "\n"
    "## Instruction for the next step\n"
    "{instruction}\n"
    "\n"
    "Now, output the next action in json format [{{\"action\": \"{{action_name}}\"}}, ...]."
)

action_space = [
    '{"action": "click", "start_point": [x,y]}', '{"action": "drag", "start_point": [x,y], "end_point": [x,y]}',
    '{"action": "input", "text": "text"}', '{"action": "press", "keys": [key1, key2, ...]}',
    '{"action": "wait"}', '{"action": "finished", "status": "status"}'
]
ACTION_SPACE = "\n".join(action_space)


def aitw_2_mimo_action(aitw_action, resized_height, resized_width):
    ex_action_type = aitw_action['result_action_type']
    mimo_action = {}

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            click_y, click_x = lift_yx[0], lift_yx[1]
            click_x = int(click_x * resized_width)
            click_y = int(click_y * resized_height)
            mimo_action["action_name"] = "click"
            mimo_action["start_point"] = [click_x, click_y]
        else:
            mimo_action["action_name"] = "drag"
            touch_y, touch_x = touch_yx[0], touch_yx[1]
            lift_y, lift_x = lift_yx[0], lift_yx[1]
            touch_x = int(touch_x * resized_width)
            touch_y = int(touch_y * resized_height)
            lift_x = int(lift_x * resized_width)
            lift_y = int(lift_y * resized_height)
            mimo_action["start_point"] = [touch_x, touch_y]
            mimo_action["end_point"] = [lift_x, lift_y]

    elif ex_action_type == ActionType.PRESS_BACK:
        mimo_action["action_name"] = "press"
        mimo_action["keys"] = ["back"]

    elif ex_action_type == ActionType.PRESS_HOME:
        mimo_action["action_name"] = "press"
        mimo_action["keys"] = ["home"]

    elif ex_action_type == ActionType.PRESS_ENTER:
        mimo_action["action_name"] = "press"
        mimo_action["keys"] = ["enter"]

    elif ex_action_type == ActionType.TYPE:
        mimo_action["action_name"] = "input"
        mimo_action["text"] = aitw_action['result_action_text']

    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE or ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        mimo_action["action_name"] = "finished"
        mimo_action["status"] = "success"

    elif ex_action_type == ActionType.LONG_POINT:
        mimo_action["action_name"] = "longpress"
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        click_y, click_x = touch_yx[0], touch_yx[1]
        click_x = int(click_x * resized_width)
        click_y = int(click_y * resized_height)
        mimo_action["start_point"] = [click_x, click_y]

    elif ex_action_type == ActionType.NO_ACTION:
        mimo_action["action_name"] = "wait"

    elif ex_action_type == ActionType.OPEN_APP:
        mimo_action["action_name"] = "open"
        mimo_action["app_name"] = aitw_action['result_action_app_name']

    else:
        print('AiTW Action: ', aitw_action)
        raise NotImplementedError

    return mimo_action


def mimo_2_minicpm(output_text, resized_height, resized_width):
    try:
        if "</think>" in output_text:
            output_text = output_text.split("</think>")[-1]
        start_idx = output_text.find("{\"action\":")
        end_idx = output_text.rfind("}")
        input_string = output_text[start_idx: end_idx + 1]
        action = json.loads(input_string)
        action_name = action['action']
    except Exception:
        print("Error, JSON is NOT valid.")
        return {}

    if action_name == "click":
        start_point = action.get('start_point', [0, 0])
        x, y = start_point[0], start_point[1]
        x = x / resized_width * 1000
        y = y / resized_height * 1000
        return {"POINT": [int(x), int(y)]}

    elif action_name == 'longpress':
        start_point = action.get('start_point', [0, 0])
        x, y = start_point[0], start_point[1]
        x = x / resized_width * 1000
        y = y / resized_height * 1000
        return {"POINT": [int(x), int(y)], "duration": 200}

    elif action_name == 'scroll':
        direction = action.get('direction', 'down')
        return {"POINT": [500, 500], "to": direction}

    elif action_name == 'drag':
        start_point = action.get('start_point', [0, 0])
        end_point = action.get('end_point', [0, 0])
        direction = get_direction([start_point[0], start_point[1]], [end_point[0], end_point[1]])
        return {"POINT": [500, 500], "to": direction}

    elif action_name == 'input':
        text = action.get('text', '')
        return {"TYPE": text}

    elif action_name == 'press':
        keys = action.get('keys', [])
        if keys == ['home']:
            return {"PRESS": "HOME"}
        elif keys == ['back']:
            return {"PRESS": "BACK"}
        elif keys == ['enter']:
            return {"PRESS": "ENTER"}

    elif action_name == 'finished':
        return {"STATUS": "finish"}

    elif action_name == 'wait':
        return {"duration": 200}

    elif action_name == 'open':
        app_name = action.get('app_name', '')
        return {"OPEN_APP": app_name}

    print("Error, unrecognized action.")

    return {}


def prepare_task_input(step, image_path, history_actions, data_name, _tokenizer, use_vllm, no_think):
    query = step['instruction']
    if data_name == 'androidcontrol_low':
        low_instruction = step["low_instruction"]

    image = Image.open(image_path)
    images = [image]
    resized_height, resized_width = smart_resize(
        image.height, image.width, factor=_tokenizer.image_processor.patch_size * _tokenizer.image_processor.merge_size,
        min_pixels=_tokenizer.image_processor.min_pixels, max_pixels=_tokenizer.image_processor.max_pixels)

    history_action_str = ""
    for idx, history_action in enumerate(history_actions):
        if history_action["low_instruction"]:
            history_action_str += (
                f"Step {idx}: {history_action['low_instruction']} "
                f"{aitw_2_mimo_action(history_action, resized_height, resized_width)}.")
        else:
            history_action_str += f"Step {idx}: {aitw_2_mimo_action(history_action, resized_height, resized_width)}."
    history_action_str = "None" if history_action_str == "" else history_action_str

    if data_name == 'androidcontrol_low':
        user_query = ANDROIDCONTROL_LOW_TEMPLATE.format(
            action_space=ACTION_SPACE, goal=query, previous_actions=history_action_str, instruction=low_instruction)
    else:
        user_query = USER_QUERY_TEMPLATE.format(action_space=ACTION_SPACE, goal=query,
                                                previous_actions=history_action_str)

    messages = []
    content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": user_query}
    ]
    if no_think:
        content.append({"type": "text", "text": " /no_think"})
    messages.append(
        {
            "role": "user",
            "content": content
        }
    )
    return step, messages, images


def prepare_task_inputs(episode, episode_dir, episode_file, subset, dataset, _tokenizer, use_vllm, no_think):
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
            history_action = {"result_action_type": prev_step['result_action_type'],
                              "result_action_text": prev_step['result_action_text'],
                              "result_touch_yx": prev_step['result_touch_yx'],
                              "result_lift_yx": prev_step['result_lift_yx'],
                              "low_instruction": prev_step.get(
                                "low_instruction", prev_step.get("coat_action_desc", "")),
                              "image_path": image_path,
                              "result_action_app_name": prev_step.get('result_action_app_name', '')
                              }
            history.append(history_action)
        image_path = os.path.join(episode_dir, episode_file, f"{episode_file}_{step['step_id']}.jpeg")
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpeg", ".png")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".png", ".jpg")
                if not os.path.exists(image_path):
                    image_path = step['image_path']
        res.append(prepare_task_input(step, image_path, history, dataset, _tokenizer, use_vllm, no_think))

    return res


def run_task_batch(_llm, _tokenizer, batch_tasks, use_vllm):
    batch_steps = []
    batch_inputs = []
    batch_images = []
    for step, messages, images in batch_tasks:
        if use_vllm:
            text_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append({"prompt": text_prompt, "multi_modal_data": {"image": images}})
        else:
            text_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append(text_prompt)

        batch_steps.append(step)
        batch_images.append(images)

    if use_vllm:
        sampling_params = SamplingParams(
            top_p=0.01, top_k=1, temperature=0.01, repetition_penalty=1.0, max_tokens=2048)
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
        resized_height, resized_width = smart_resize(
            images[0].height, images[0].width,
            factor=_tokenizer.image_processor.patch_size * _tokenizer.image_processor.merge_size,
            min_pixels=_tokenizer.image_processor.min_pixels, max_pixels=_tokenizer.image_processor.max_pixels)
        try:
            step['pred'] = mimo_2_minicpm(str_res, resized_height, resized_width)
        except Exception:
            print("Error, JSON is NOT valid.")
            step['pred'] = {}
        res_steps.append(step)

    return res_steps
