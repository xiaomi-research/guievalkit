import json
import numpy as np
import os

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt, Message, ContentItem
from qwen_vl_utils import smart_resize
from PIL import Image
from vllm import SamplingParams

from guieval.models.utils.qwen2p5_vl_agent_function_call import MobileUse
from guieval.utils.action_type import ActionType
from guieval.utils.action_utils import is_tap_action

USER_QUERY_TEMPLATE = '''The user query:  {user_request}
Task progress (You have done the following operation on the current device): {history_actions}'''

ANDROIDCONTROL_LOW_TEMPLATE = (
    "The user query:  {user_request} \n"
    "Current step query: {low_instruction}\n"
    "Task progress (You have done the following operation on the current device): {history_actions}"
)

GUI_ODYSSEY_TEMPLATE = (
    "The user query: {user_request}\n"
    "Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, "
    "and insert them before the <tool_call></tool_call> XML tags.\n"
    "After answering, summarize your action in <conclusion></conclusion> tags, "
    "and insert them after the <tool_call></tool_call> XML tags.\n"
    "Task progress (You have done the following operation on the current device):\n"
    "{history_actions}"
)


def aitw_2_qwen2_5_action(aitw_action, resized_height, resized_width):
    ex_action_type = aitw_action['result_action_type']
    qwen_action = {"name": "mobile_use", "arguments": {}}

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            click_y, click_x = lift_yx[0], lift_yx[1]
            click_x = int(click_x * resized_width)
            click_y = int(click_y * resized_height)
            qwen_action["arguments"] = {"action": "click", "coordinate": [click_x, click_y]}
        else:
            qwen_action["arguments"] = {"action": "swipe",
                                        "coordinate": [int(touch_yx[1] * resized_width),
                                                       int(touch_yx[0] * resized_height)],
                                        "coordinate2": [int(lift_yx[1] * resized_width),
                                                        int(lift_yx[0] * resized_height)]}

    elif ex_action_type == ActionType.PRESS_BACK:
        button = "Back"
        qwen_action["arguments"] = {"action": "system_button", "button": button}

    elif ex_action_type == ActionType.PRESS_HOME:
        button = "Home"
        qwen_action["arguments"] = {"action": "system_button", "button": button}

    elif ex_action_type == ActionType.PRESS_ENTER:
        button = "Enter"
        qwen_action["arguments"] = {"action": "system_button", "button": button}

    elif ex_action_type == ActionType.TYPE:
        qwen_action["arguments"] = {"action": "type", "text": aitw_action['result_action_text']}

    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        qwen_action["arguments"] = {"action": "terminate", "status": "success"}

    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        qwen_action["arguments"] = {"action": "terminate", "status": "failure"}

    elif ex_action_type == ActionType.LONG_POINT:
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        qwen_action["arguments"] = {"action": "long_press",
                                    "coordinate": [int(touch_yx[1] * resized_width),
                                                   int(touch_yx[0] * resized_height)],
                                    "time": 2}

    elif ex_action_type == ActionType.NO_ACTION:
        qwen_action["arguments"] = {"action": "wait", "time": 2}

    elif ex_action_type == ActionType.OPEN_APP:
        qwen_action["arguments"] = {"action": "open", "text": aitw_action['result_action_app_name']}

    else:
        print('AiTW Action: ', aitw_action)
        raise NotImplementedError

    return json.dumps(qwen_action)


def qwen2p5_2_minicpm(output_text, resized_height, resized_width):
    try:
        input_string = output_text.split('<tool_call>\n')[1].split('\n</tool_call>')[0].strip()
        action = json.loads(input_string)
        qwen_action = action['arguments']
        action_name = qwen_action['action']
    except Exception:
        print("Error, JSON is NOT valid.")
        return {}

    if action_name == "click":
        x, y = qwen_action["coordinate"]
        x = x / resized_width * 1000
        y = y / resized_height * 1000
        return {"POINT": [int(x), int(y)]}

    elif action_name == "long_press":
        x, y = qwen_action["coordinate"]
        x = x / resized_width * 1000
        y = y / resized_height * 1000
        time = qwen_action["time"]
        time = time * 1000  # convert time to milliseconds
        return {"POINT": [int(x), int(y)], "duration": time}

    elif action_name == "swipe":
        x1, y1 = qwen_action["coordinate"]
        x2, y2 = qwen_action["coordinate2"]
        x1 = x1 / resized_width * 1000
        y1 = y1 / resized_height * 1000
        x2 = x2 / resized_width * 1000
        y2 = y2 / resized_height * 1000
        if abs(x2 - x1) > abs(y2 - y1):
            direction = "right" if x2 > x1 else "left"
        else:
            direction = "down" if y2 > y1 else "up"
        return {"POINT": [int(x1), int(y1)], "to": direction}

    elif action_name == "type":
        return {"TYPE": qwen_action["text"]}

    elif action_name == "system_button":
        button = qwen_action["button"]
        if button == "Back":
            return {"PRESS": "BACK"}
        elif button == "Home":
            return {"PRESS": "HOME"}
        elif button == "Enter":
            return {"PRESS": "ENTER"}

    elif action_name == "terminate":
        return {"STATUS": "finish"}

    elif action_name == "wait":
        time = qwen_action["time"]
        time = time * 1000  # convert time to milliseconds
        return {"duration": time}

    elif action_name == "open":
        return {"OPEN_APP": qwen_action["text"]}

    print("Error, unrecognized action.")

    return {}


def prepare_task_input(step, image_path, history_actions, data_name, _tokenizer, use_vllm):
    query = step['instruction']
    if data_name == 'androidcontrol_low':
        low_instruction = step["low_instruction"]

    image = Image.open(image_path)
    images = [image]
    resized_height, resized_width = smart_resize(
        image.height, image.width, factor=_tokenizer.image_processor.patch_size * _tokenizer.image_processor.merge_size,
        min_pixels=_tokenizer.image_processor.min_pixels, max_pixels=_tokenizer.image_processor.max_pixels)

    mobile_use = MobileUse(cfg={"display_width_px": resized_width, "display_height_px": resized_height})

    if history_actions:
        history_actions_str = "".join(
            [f"Step {i + 1}: {aitw_2_qwen2_5_action(action, resized_height, resized_width).strip()}; " for i, action in
             enumerate(history_actions)])
    else:
        history_actions_str = ""

    prompt = NousFnCallPrompt()
    if data_name == 'androidcontrol_low':
        message = prompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(text=ANDROIDCONTROL_LOW_TEMPLATE.format(user_request=query,
                                                                        history_actions=history_actions_str,
                                                                        low_instruction=low_instruction)),
                    ContentItem(image=f"file://{image_path}")
                ]),
            ],
            functions=[mobile_use.function],
            lang=None,
        )
    elif data_name == 'gui_odyssey':
        message = prompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(
                        text=GUI_ODYSSEY_TEMPLATE.format(user_request=query, history_actions=history_actions_str)),
                    ContentItem(image=f"file://{image_path}")
                ]),
            ],
            functions=[mobile_use.function],
            lang=None,
        )
    else:
        message = prompt.preprocess_fncall_messages(
            messages=[
                Message(role="system", content=[ContentItem(text="You are a helpful assistant.")]),
                Message(role="user", content=[
                    ContentItem(
                        text=USER_QUERY_TEMPLATE.format(user_request=query, history_actions=history_actions_str)),
                    ContentItem(image=f"file://{image_path}")
                ]),
            ],
            functions=[mobile_use.function],
            lang=None,
        )
    message = [msg.model_dump() for msg in message]

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
            history_action = {"result_action_type": prev_step['result_action_type'],
                              "result_action_text": prev_step['result_action_text'],
                              "result_touch_yx": prev_step['result_touch_yx'],
                              "result_lift_yx": prev_step['result_lift_yx'],
                              "low_instruction": prev_step.get("low_instruction", ""),
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
        res.append(prepare_task_input(step, image_path, history, dataset, _tokenizer, use_vllm))

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
            step['pred'] = qwen2p5_2_minicpm(str_res, resized_height, resized_width)
        except Exception:
            print("Error, JSON is NOT valid.")
            step['pred'] = {}
        res_steps.append(step)
    return res_steps
