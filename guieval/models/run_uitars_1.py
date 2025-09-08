import copy
import json
import numpy as np
import os
import re

from qwen_vl_utils import process_vision_info
from vllm import SamplingParams

from guieval.utils.action_type import ActionType
from guieval.utils.action_utils import is_tap_action, get_direction

SYSTEM_PROMPT = "You are a helpful assistant."
USER_PROMPT = ("You are a GUI agent. You are given a task and your action history, with screenshots. "
               "You need to perform the next action to complete the task. \n\n"
               "## Output Format\n\n"
               "Thought: ...\n"
               "Action: ...\n\n\n"
               "## Action Space\n"
               "click(start_box=\'<|box_start|>(x1,y1)<|box_end|>\')\n"
               "long_press(start_box=\'<|box_start|>(x1,y1)<|box_end|>\', time=\'\')\n"
               "type(content=\'\')\n"
               "scroll(direction=\'down or up or right or left\')\n"
               # "open_app(app_name=\'\')\n"
               "press_back()\n"
               "press_home()\n"
               "wait()\n"
               "finished() # Submit the task regardless of whether it succeeds or fails.\n\n"
               "## Note\n"
               "- Use English in Thought part.\n\n"
               "- Summarize your next action (with its target element) in one sentence in Thought part.\n\n"
               "## User Instruction\n")


def extract_coords(s):
    pattern = re.compile(r'\((\d+),(\d+)\)')
    matches = pattern.findall(s)
    if len(matches) != 0:
        x, y = matches[0]
        return [int(x), int(y)]
    print(f"Cannot find coordinates in the string: {s}")
    return [0, 0]


def parse_scroll_command(command):
    pattern = r"drag\(start_box='(\(\d+,\d+\))', end_box='(\(\d+,\d+\))'\)"
    match = re.search(pattern, command)
    if match:
        start_box = match.group(1)
        end_box = match.group(2)
        start_box = tuple(map(int, start_box.strip("()").split(",")))
        end_box = tuple(map(int, end_box.strip("()").split(",")))
        x1, y1 = start_box
        x2, y2 = end_box
        if abs(y2 - y1) > abs(x2 - x1):
            if y2 > y1:
                _dir = 'up'
            else:
                _dir = 'down'
        else:
            if x2 > x1:
                _dir = 'left'
            else:
                _dir = 'right'
        return _dir
    else:
        print(f"Cannot find directions in the string: {command}")
        return 'no direction'


def uitars2minicpm(action_str):

    result = {"STATUS": "continue"}

    if "click(" in action_str:
        result["POINT"] = extract_coords(action_str)

    elif "long_press(" in action_str:
        result["POINT"] = extract_coords(action_str)
        if "time='" in action_str:
            time = action_str.split("time='")[1].split("'")[0]
            result["duration"] = int(time) if time else 1000

    elif "type(" in action_str:
        content = action_str.split("content='")[1].split("'")[0]
        result["TYPE"] = content

    elif "scroll(" in action_str:
        direction = action_str.split("direction='")[1].split("'")[0]
        result["POINT"] = [500, 500]
        # need reverse direction
        if direction == "down":
            direction = "up"
        elif direction == "up":
            direction = "down"
        elif direction == "right":
            direction = "left"
        elif direction == "left":
            direction = "right"
        result["to"] = direction

    elif "drag(" in action_str:
        _dir = parse_scroll_command(action_str)
        result["POINT"] = [500, 500]
        result["to"] = _dir

    elif "press_back()" in action_str:
        result["PRESS"] = "BACK"

    elif "press_home()" in action_str:
        result["PRESS"] = "HOME"

    elif "press_enter()" in action_str:
        result["PRESS"] = "ENTER"

    elif "wait(" in action_str:
        result["duration"] = 200

    elif "finished(" in action_str:
        result["STATUS"] = "finish"

    elif "open_app(app_name=" in action_str:
        result["OPEN_APP"] = action_str.split("app_name='")[1].split("'")[0]
    else:
        print(f"Error, invalid action: {action_str}")

    return result


def aitw_2_uitars(aitw_action: dict):

    ex_action_type = aitw_action['result_action_type']

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            # Click action
            click_y, click_x = lift_yx[0], lift_yx[1]
            click_x = int(click_x * 1000)
            click_y = int(click_y * 1000)
            return f"click(start_box=\'<|box_start|>({click_x},{click_y})<|box_end|>\')"
        else:
            touch_yx_new = [touch_yx[1], touch_yx[0]]
            lift_yx_new = [lift_yx[1], lift_yx[0]]
            direction = get_direction(touch_yx_new, lift_yx_new)

            if direction == 'up':
                inv_direction = 'down'
            elif direction == 'down':
                inv_direction = 'up'
            elif direction == 'right':
                inv_direction = 'left'
            elif direction == 'left':
                inv_direction = 'right'
            else:
                inv_direction = 'no direction'

            return f"scroll(direction='{inv_direction}')"

    elif ex_action_type == ActionType.PRESS_BACK:
        return f"press_back()"

    elif ex_action_type == ActionType.PRESS_HOME:
        return f"press_home()"

    elif ex_action_type == ActionType.PRESS_ENTER:
        return f"press_enter()"

    elif ex_action_type == ActionType.TYPE:
        return f"type(content='{aitw_action['result_action_text']}')"

    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        return f"finished()"

    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        return f"finished()"

    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        click_y, click_x = lift_yx[0], lift_yx[1]
        click_x = int(click_x * 1000)
        click_y = int(click_y * 1000)
        return f"long_press(start_box=\'<|box_start|>({click_x},{click_y})<|box_end|>\')"

    elif ex_action_type == ActionType.NO_ACTION:
        return f"wait()"

    elif ex_action_type == ActionType.OPEN_APP:
        return f"open(app_name='{aitw_action['result_action_app_name']}')"
    else:
        print('aitw_action:', aitw_action)
        raise NotImplementedError


def build_history_actions_str(history_list):
    history = []

    # Get indices of the last 4 image records
    image_indices = range(max(0, len(history_list) - 4), len(history_list))

    for i, step_history in enumerate(history_list):
        # If current index is in the last 4 image records, add the image
        if i in image_indices:
            image_history = {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": step_history["image_path"]
                    }
                ]
            }
            history.append(image_history)

        # Add action
        if i in image_indices:
            action = aitw_2_uitars(step_history)
            thought = step_history.get("low_instruction", "")
            text_history = {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": f"Thought: {thought}\nAction: {action}"}
                ]
            }
            history.append(text_history)
    return history


def prepare_task_input(step, image_path, data_name, episode_history, _tokenizer, use_vllm):
    images = None  # 读取image的部分可能可以做一点加速，目前会open两次

    query = step['instruction']

    history = build_history_actions_str(episode_history)

    text = USER_PROMPT + query
    conversation = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ],
        }
    ]

    conversation.extend(history)
    conversation.append({
        "role": "user",
        "content": [
            {"type": "image", "image": image_path}
        ],
    })

    if data_name == 'androidcontrol_low':
        thought = "Thought: " + step['low_instruction'] + "\nAction:"
        conversation.append({
            "role": "assistant",
            "content": [
                {"type": "text", "text": thought}
            ],
        })
        add_generation_prompt = False
    else:
        add_generation_prompt = True
    text_prompt = _tokenizer.apply_chat_template(conversation, tokenize=False,
                                                 add_generation_prompt=add_generation_prompt)
    if data_name == 'androidcontrol_low':
        text_prompt = text_prompt.rsplit("<|im_end|>", 1)[0].strip()
    if use_vllm:
        image_inputs, video_inputs = process_vision_info(conversation)
        inputs = {"prompt": text_prompt, "multi_modal_data": {"image": image_inputs}}
    else:
        inputs = (text_prompt, conversation)
    return step, inputs, images


def prepare_task_inputs(episode, episode_dir, episode_file, subset, dataset, _tokenizer, use_vllm):
    res = []
    for index, step in enumerate(episode):
        episode_history = []  # Create a separate history for each episode
        for prev_episode in episode[:index]:
            image_path = os.path.join(episode_dir, episode_file, f"{episode_file}_{prev_episode['step_id']}.jpeg")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".jpeg", ".png")
                if not os.path.exists(image_path):
                    image_path = image_path.replace(".png", ".jpg")
                    if not os.path.exists(image_path):
                        image_path = prev_episode['image_path']
            history_action = {
                "result_action_type": prev_episode['result_action_type'],
                "result_action_text": prev_episode['result_action_text'],
                "result_touch_yx": prev_episode['result_touch_yx'],
                "result_lift_yx": prev_episode['result_lift_yx'],
                "low_instruction": prev_episode.get("low_instruction", ""),
                "image_path": image_path,
                "result_action_app_name": prev_episode.get('result_action_app_name', ''),
            }
            episode_history.append(history_action)

        step["category"] = subset
        image_path = os.path.join(episode_dir, episode_file, f"{episode_file}_{step['step_id']}.jpeg")
        if not os.path.exists(image_path):
            image_path = image_path.replace(".jpeg", ".png")
            if not os.path.exists(image_path):
                image_path = image_path.replace(".png", ".jpg")
                if not os.path.exists(image_path):
                    image_path = step['image_path']
        episode_history_copy = copy.deepcopy(episode_history)

        res.append(prepare_task_input(step, image_path, dataset, episode_history_copy, _tokenizer, use_vllm))

    return res


def run_task_batch(_llm, _tokenizer, batch_tasks, use_vllm):
    batch_steps = []
    batch_inputs = []
    for step, inputs, _ in batch_tasks:
        batch_inputs.append(inputs)
        batch_steps.append(step)
    if use_vllm:
        sampling_params = SamplingParams(temperature=0.1, max_tokens=2048)
        results = _llm.generate(batch_inputs, sampling_params, use_tqdm=False)
        predict_str = [result.outputs[0].text for result in results]

    else:
        texts = [msg[0] for msg in batch_inputs]
        conversations = [msg[1] for msg in batch_inputs]
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = _tokenizer(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(_llm.device)

        generated_ids = _llm.generate(**inputs, temperature=0.1, max_new_tokens=256, do_sample=True)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = _tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        predict_str = output_texts

    res_steps = []
    for step, str_res in zip(batch_steps, predict_str):
        try:
            step['pred'] = uitars2minicpm(str_res, )
        except Exception:
            print("Error, JSON is NOT valid.")
            step['pred'] = {}
        res_steps.append(step)
    return res_steps
