import json
import numpy as np
import os
import re

from PIL import Image
from vllm import SamplingParams

from guieval.utils.action_type import ActionType
from guieval.utils.action_utils import is_tap_action, get_direction


def get_mobile_prompt(task, history):

    prompt = (
        f"""You are an agent who can operate an Android phone on behalf of a user. """
        f"""Based on user's goal/request, you may\n"""
        # f"""- Answer back if the request/goal is a question (or a chat message), """
        # f"""like user asks "What is my schedule for today?".\n"""
        f"""- Complete some tasks described in the requests/goals """
        f"""by performing actions (step by step) on the phone.\n"""
        f"""\n"""
        f"""When given a user request, you will try to complete it step by step. At each step, """
        f"""you will be given the current screenshot (including the original screenshot """
        f"""and the same screenshot with bounding boxes and numeric indexes added to some UI elements) """
        f"""and a history of what you have done (in text). Based on these pieces of information and the goal, """
        f"""you must choose to perform one of the action in the following list """
        f"""(action description followed by the JSON format) by outputting the action in the correct JSON format.\n"""
        f"""- If you think the task has been completed, """
        f"""finish the task by using the status action with complete as goal_status: """
        f"""`{{"action_type": "status", "goal_status": "complete"}}`\n"""
        f"""- If you think the task is not feasible """
        f"""(including cases like you don't have enough information or can not perform some necessary actions), """
        f"""finish by using the `status` action with infeasible as goal_status: """
        f"""`{{"action_type": "status", "goal_status": "infeasible"}}`\n"""
        # f"""- Answer user's question: `{{"action_type": "answer", "text": "<answer_text>"}}`\n"""
        # f"""-- You should only answer once in one command. If you needs multiple pieces of information to answer """
        # f"""the """
        # f"""question, you should gather the information in "Memory" and answer the question when you have enough """
        # f"""information.\n"""
        f"""- Click/tap on an element on the screen. Use the box_2d to indicate which element you want to click: """
        f"""`{{"action_type": "click", "box_2d": [[,,,]]}}`. """
        f"""The box_2d should be [[xmin,ymin,xmax,ymax]] normalized to 0-999, """
        f"""indicating the position of the element.\n"""
        f"""- Long press on an element on the screen, similar with the click action above, """
        f"""use the box_2d to indicate which element you want to long press: """
        f"""`{{"action_type": "long_press", "box_2d": [[,,,]]}}`.\n"""
        f"""- Type text into a text field (this action contains clicking the text field, """
        f"""typing in the text and pressing the enter, so no need to click on the target field to start), """
        f"""use the box_2d to indicate the target text field. The text to be input can be from the command, """
        f"""the memory, or the current screen: """
        f"""`{{"action_type": "input_text", "text": <text_input>, "box_2d": [[,,,]], "override": true/false}}`. """
        f"""If override is True, the text field will be cleared before typing.\n"""
        f"""- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n"""
        f"""- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n"""
        f"""- Navigate back: `{{"action_type": "navigate_back"}}`\n"""
        f"""- Swipe the screen or a scrollable UI element in one of the four directions, """
        f"""use the box_2d as above if you want to swipe a specific UI element, """
        f"""leave it empty when swipe the whole screen: """
        f"""`{{"action_type": "swipe", "direction": <up, down, left, right>, "box_2d": [[,,,]](optional)}}`. \n"""
        # f"""- Open an app (nothing will happen if the app is not installed): `{{"action_type": "open_app", """
        # f""""app_name": <name>}}`\n"""
        # f"""-- supported app_names: {",".join(app_names)}\n"""
        f"""- Wait for the screen to update: `{{"action_type": "wait"}}`\n"""
        f"""\n"""
        f"""The current user goal/request is: {task}\n"""
        f"""\n"""
        f"""Here is a history of what you have done so far:\n"""
        )

    history_str = ""
    if len(history) == 0:
        history_str = "You just started, no action has been performed yet."
    else:
        for idx, glm_history in enumerate(history):
            history_str += f"Step {idx}:\n{glm_history}\n\n"

    prompt += history_str + "\n"

    prompt += (
        f"""The current screenshot is given to you. \n"""
        f"""Here are some useful guidelines you need to follow:\n"""
        f"""General:\n"""
        f"""- Usually there will be multiple ways to complete a task, pick the easiest one. """
        f"""Also when something does not work as expected (due to various reasons), """
        f"""sometimes a simple retry can solve the problem, but if it doesn't (you can see that from the history), """
        f"""SWITCH to other solutions.\n"""
        # f"""- Sometimes you may need to navigate the phone to gather information needed to complete the task, """
        # f"""for example if user asks "what is my schedule tomorrow", then you may want to open the calendar app """
        # f"""(using the `open_app` action), look up information there, answer user's question (using the `answer` """
        # f"""action) and finish (using the `status` action with complete as goal_status).\n"""
        # f"""- For requests that are questions (or chat messages), remember to use the `answer` action to """
        # f"""reply to user explicitly before finish! Merely displaying the answer on the screen is NOT sufficient """
        # f"""(unless the goal is something like "show me ...").\n"""
        f"""- If the desired state is already achieved (e.g., enabling Wi-Fi when it's already on), """
        f"""you can just complete the task.\n"""
        f"""- If we say that two items are duplicated, """
        f"""in most cases we require that all of their attributes are exactly the same, not just the name.\n"""
        f"""Text Related Operations:\n"""
        f"""- Normally to select certain text on the screen: """
        f"""<i> Enter text selection mode by long pressing the area where the text is, """
        f"""then some of the words near the long press point will be selected """
        f"""(highlighted with two pointers indicating the range) """
        f"""and usually a text selection bar will also appear with options like `copy`, `paste`, `select all`, etc. """
        f"""<ii> Select the exact text you need. """
        f"""Usually the text selected from the previous step is NOT the one you want, """
        f"""you need to adjust the range by dragging the two pointers. """
        f"""If you want to select all text in the text field, simply click the `select all` button in the bar.\n"""
        f"""- To delete some text: first select the text you want to delete (if you want to delete all texts, """
        f"""just long press the text field and click the `clear all` button in the text selection bar), """
        f"""then click the backspace button in the keyboard.\n"""
        f"""- To copy some text: first select the exact text you want to copy, """
        f"""which usually also brings up the text selection bar, then click the `copy` button in bar.\n"""
        f"""- To paste text into a text box, first long press the text box, """
        f"""then usually the text selection bar will appear with a `paste` button in it.\n"""
        f"""- When typing into a text field, sometimes an auto-complete dropdown list will appear. """
        f"""This usually indicating this is a enum field """
        f"""and you should try to select the best match by clicking the corresponding one in the list.\n"""
        f"""Action Related:\n"""
        f"""- Use the `input_text` action whenever you want to type something (including password) """
        f"""instead of clicking characters on the keyboard one by one. """
        f"""Sometimes there is some default text in the text field you want to type in, """
        f"""remember to delete them before typing.\n"""
        f"""- Consider exploring the screen by using the `swipe` action with different directions """
        f"""to reveal additional content.\n"""
        f"""- The direction parameter for the `swipe` action can be confusing sometimes """
        f"""as it's opposite to swipe, for example, to view content at the bottom, """
        f"""the `swipe` direction should be set to "up". """
        f"""It has been observed that you have difficulties in choosing the correct direction, """
        f"""so if one does not work, try the opposite as well.\n"""
        f"""- To open an app if you can not find its icon, """
        f"""you can first press home (if necessary) and swipe up to the app drawer.\n"""
        f"""- Swipe up means swiping from bottom to top, swipe down means swiping from top to bottom, """
        f"""swipe left means swiping from right to left, swipe right means swiping from left to right.\n"""
        f"""- Use the `navigate_back` action to close/hide the soft keyboard.\n"""
        f"""\n"""
        f"""Now output: \n"""
        f"""1. Memory: important information you want to remember for the future actions. """
        f"""The memory should be only contents on the screen that will be used in the future actions. """
        f"""It should satisfy that: you cannot determine one or more future actions without this memory. \n"""
        f"""2. Reason: the reason for the action and the memory. """
        f"""Your reason should include, but not limited to:- the content of the GUI, """
        f"""especially elements that are tightly related to the user goal- the step-by-step thinking """
        f"""process of how you come up with the new action. \n"""
        f"""3. Action: the action you want to take, in the correct JSON format. """
        f"""The action should be one of the above list.\n"""
        f"""\n"""
        f"""Your answer should look like:\n"""
        f"""Memory: ...\n"""
        f"""Reason: ...\n"""
        f"""Action: {{"action_type":...}}"""
        )

    return prompt


def parse_mobile_response(response):
    pattern = r"Memory:(.*?)Reason:(.*?)Action:(.*)"
    match = re.search(pattern, response, re.DOTALL)
    if not match:
        return None

    memory = match.group(1).strip()
    reason = match.group(2).strip()
    action = match.group(3).strip()

    if "<|begin_of_box|>" in action:
        action = action[
            action.index("<|begin_of_box|>") + len("<|begin_of_box|>"): action.rindex(
                "<|end_of_box|>"
            )
        ]

    parsed_action = None
    if action.startswith("{"):
        parsed_action = json.loads(action)

    return {
        "memory": memory,
        "reason": reason,
        "action": action,
        "parsed_action": parsed_action,
    }


def aitw_2_glm4p1v_action(aitw_action):

    ex_action_type = aitw_action['result_action_type']
    glm_action = {"Action": {}}

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            lift_yx = json.loads(aitw_action['result_lift_yx'])
            touch_yx = json.loads(aitw_action['result_touch_yx'])
            click_y_min = int(touch_yx[0] * 999)
            click_x_min = int(touch_yx[1] * 999)
            click_y_max = int(lift_yx[0] * 999)
            click_x_max = int(lift_yx[1] * 999)
            glm_action["Action"] = {"action_type": "click",
                                    "box_2d": [[click_x_min, click_y_min, click_x_max, click_y_max]]}
        else:
            touch_yx_new = [touch_yx[1], touch_yx[0]]
            lift_yx_new = [lift_yx[1], lift_yx[0]]
            direction = get_direction(touch_yx_new, lift_yx_new)
            glm_action["Action"] = {"action_type": "swipe",
                                           "direction": direction,
                                        "box_2d": [[int(touch_yx[1] * 999), int(touch_yx[0] * 999),
                                                      int(lift_yx[1] * 999), int(lift_yx[0] * 999)]]}

    elif ex_action_type == ActionType.PRESS_BACK:
        glm_action["Action"] = {"action_type": "navigate_back"}

    elif ex_action_type == ActionType.PRESS_HOME:
        glm_action["Action"] = {"action_type": "navigate_home"}

    elif ex_action_type == ActionType.PRESS_ENTER:
        glm_action["Action"] = {"action_type": "keyboard_enter"}

    elif ex_action_type == ActionType.TYPE:
        bbox = aitw_action.get('bbox', "")
        if bbox:
            bbox = json.loads(aitw_action['bbox'])
            bbox = bbox[0]
            click_y_min = int(bbox[0] * 999)
            click_x_min = int(bbox[1] * 999)
            click_y_max = int(bbox[2] * 999)
            click_x_max = int(bbox[3] * 999)
        else:
            lift_yx = json.loads(aitw_action['result_lift_yx'])
            touch_yx = json.loads(aitw_action['result_touch_yx'])
            click_y_min = int(touch_yx[0] * 999)
            click_x_min = int(touch_yx[1] * 999)
            click_y_max = int(lift_yx[0] * 999)
            click_x_max = int(lift_yx[1] * 999)

        glm_action["Action"] = {
                                "action_type": "input_text",
                                "text": aitw_action['result_action_text'],
                                "box_2d": [[click_x_min, click_y_min, click_x_max, click_y_max]],
                                "override": "true"}

    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        glm_action["Action"] = {"action_type": "status", "goal_status": "complete"}

    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        glm_action["Action"] = {"action_type": "status", "goal_status": "infeasible"}

    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        click_y_min = int(touch_yx[0] * 999)
        click_x_min = int(touch_yx[1] * 999)
        click_y_max = int(lift_yx[0] * 999)
        click_x_max = int(lift_yx[1] * 999)
        glm_action["Action"] = {"action_type": "long_press",
                                "box_2d": [[click_x_min, click_y_min, click_x_max, click_y_max]]}

    elif ex_action_type == ActionType.NO_ACTION:
        glm_action["Action"] = {"action_type": "wait"}

    elif ex_action_type == ActionType.OPEN_APP:
        glm_action["Action"] = {"action_type": "open_app", "app_name": aitw_action['result_action_app_name']}

    else:
        print('AiTW Action: ', aitw_action)
        raise NotImplementedError

    return f"""Action: {json.dumps(glm_action["Action"], ensure_ascii=False)}"""


def glm4p1v_2_minicpm(output_text):

    try:
        parsed_response = parse_mobile_response(output_text)
        glm_action = parsed_response["parsed_action"]
        action_name = glm_action['action_type']
    except Exception as e:
        print(f"Error, JSON is NOT valid: {e}")
        return {}

    if action_name == "click":
        box_2d = glm_action["box_2d"]
        x_min, y_min, x_max, y_max = box_2d[0]
        x = (x_min + x_max) // 2
        y = (y_min + y_max) // 2
        x = x / 999 * 1000
        y = y / 999 * 1000
        return {"POINT": [int(x), int(y)]}

    elif action_name == "long_press":
        box_2d = glm_action["box_2d"]
        x_min, y_min, x_max, y_max = box_2d[0]
        x = (x_min + x_max) // 2
        y = (y_min + y_max) // 2
        x = x / 999 * 1000
        y = y / 999 * 1000
        time = 1000
        return {"POINT": [int(x), int(y)], "duration": time}

    elif action_name == "swipe":
        direction = glm_action['direction']
        box_2d = glm_action.get("box_2d", "")
        if box_2d:
            x_min, y_min, x_max, y_max = box_2d[0]
            x = (x_min + x_max) // 2
            y = (y_min + y_max) // 2
            x = x / 999 * 1000
            y = y / 999 * 1000
            return {"POINT": [int(x), int(y)], "to": direction}
        return {"POINT": [500, 500], "to": direction}  # screen center point

    elif action_name == "input_text":
        box_2d = glm_action["box_2d"]
        x_min, y_min, x_max, y_max = box_2d[0]
        x = (x_min + x_max) // 2
        y = (y_min + y_max) // 2
        x = x / 999 * 1000
        y = y / 999 * 1000
        return {"INPUT": {"text": glm_action["text"], "point": [int(x), int(y)]}}

    elif action_name == "navigate_back":
        return {"PRESS": "BACK"}

    elif action_name == "navigate_home":
        return {"PRESS": "HOME"}

    elif action_name == "keyboard_enter":
        return {"PRESS": "ENTER"}

    elif action_name == "status":
        return {"STATUS": "finish"}

    elif action_name == "wait":
        return {"duration": 1000}

    elif action_name == "open_app":
        return {"OPEN_APP": glm_action["app_name"]}
    print("Error, unrecognized action.")
    return {}


def prepare_task_input(step, image_path, history_actions, data_name, _tokenizer, use_vllm):
    query = step['instruction']
    if data_name == 'androidcontrol_low':
        query = step["low_instruction"]
    image = Image.open(image_path)
    images = [image]
    glm_history = []
    for action in history_actions:
        glm_history.append(aitw_2_glm4p1v_action(action))

    prompt = get_mobile_prompt(query, glm_history)
    message = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"<image></image>"},
                {"type": "text", "text": prompt},
            ],
        }
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
            batch_inputs.append({"prompt": text_prompt, "multi_modal_data": {"image": images}})
        else:
            text_prompt = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            batch_inputs.append(text_prompt)

        batch_steps.append(step)
        batch_images.append(images)

    if use_vllm:
        sampling_params = SamplingParams(temperature=0.01, max_tokens=8192)
        results = _llm.generate(batch_inputs, sampling_params, use_tqdm=False)
        predict_str = [result.outputs[0].text for result in results]
    else:
        generation_params = {'temperature': 0.01}
        inputs = _tokenizer(text=batch_inputs, images=batch_images, padding=True, return_tensors="pt")
        inputs = inputs.to(_llm.device)
        generated_ids = _llm.generate(**inputs, max_new_tokens=8192, **generation_params)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        predict_str = _tokenizer.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    res_steps = []
    for step, str_res, images in zip(batch_steps, predict_str, batch_images):
        try:
            step['pred'] = glm4p1v_2_minicpm(str_res)
        except Exception as e:
            print(f"Error, JSON is NOT valid: {e}")
            step['pred'] = {}
        res_steps.append(step)
    return res_steps
