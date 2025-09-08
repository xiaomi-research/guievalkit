import itertools
import json
import math
import numpy as np
import os
import random
import re

from typing import Any, Literal, Union, Optional, Dict

from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt, Message, ContentItem
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_vl_utils import smart_resize, process_vision_info
from vllm import SamplingParams
from PIL import Image

from guieval.utils.action_type import ActionType
from guieval.utils.action_utils import is_tap_action, get_direction


class KeyMismatchError(Exception):
    """Raised when block keys don't match the expected keys."""


@register_tool("mobile_for_gui_owl")
class MobileUse(BaseTool):
    @property
    def description(self):
        return ("Use a touchscreen to interact with a mobile device, and take screenshots.\n"
                "* This is an interface to a mobile device with touchscreen. "
                "You can perform actions like clicking, typing, swiping, etc.\n"
                "* Some applications may take time to start or process actions, "
                "so you may need to wait and take successive screenshots to see the results of your actions.\n"
                f"* The screen's resolution is {self.display_width_px}x{self.display_height_px}.\n"
                "* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. "
                "Don't click boxes on their edges unless asked.").strip()

    parameters = {
        "properties": {
            "action": {
                "description": """
The action to perform. The available actions are:
* `key`: Perform a key event on the mobile device.
    - This supports adb's `keyevent` syntax.
    - Examples: "volume_up", "volume_down", "power", "camera", "clear".
* `click`: Click the point on the screen with coordinate (x, y).
* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.
* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).
* `type`: Input the specified text into the activated input box.
* `system_button`: Press the system button.
* `open`: Open an app on the device.
* `wait`: Wait specified seconds for the change to happen.
* `terminate`: Terminate the current task and report its completion status.
""".strip(),
                "enum": [
                    "key",
                    "click",
                    "long_press",
                    "swipe",
                    "type",
                    "system_button",
                    "open",
                    "wait",
                    "terminate",
                ],
                "type": "string",
            },
            "coordinate": {
                "description": ("(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                                "coordinates to move the mouse to. "
                                "Required only by `action=click`, `action=long_press`, and `action=swipe`."),
                "type": "array",
            },
            "coordinate2": {
                "description": ("(x, y): The x (pixels from the left edge) and y (pixels from the top edge) "
                                "coordinates to move the mouse to. "
                                "Required only by `action=swipe`."),
                "type": "array",
            },
            "text": {
                "description": "Required only by `action=key`, `action=type`, and `action=open`.",
                "type": "string",
            },
            "time": {
                "description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.",
                "type": "number",
            },
            "button": {
                "description": ("Back means returning to the previous interface, Home means returning to the desktop, "
                                "Menu means opening the application background menu, "
                                "and Enter means pressing the enter. "
                                "Required only by `action=system_button`"),
                "enum": [
                    "Back",
                    "Home",
                    "Menu",
                    "Enter",
                ],
                "type": "string",
            },
            "status": {
                "description": "The status of the task. Required only by `action=terminate`.",
                "type": "string",
                "enum": ["success", "failure"],
            },
        },
        "required": ["action"],
        "type": "object"}

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params, **kwargs):
        return 'function_holder'


def build_system_messages(resized_width, resized_height):
    mobile_use = MobileUse(
        cfg={"display_width_px": resized_width, "display_height_px": resized_height}
    )
    query_messages = [
        Message(
            role="system", content=[ContentItem(text="You are a helpful assistant.")]
        )
    ]
    messages = NousFnCallPrompt().preprocess_fncall_messages(
        messages=query_messages,
        functions=[mobile_use.function],
        lang=None,
    )
    messages = [m.model_dump() for m in messages]
    system_prompt_part = {'role': 'system', 'content': []}
    system_prompt_part['content'].append(
        {"type": "text", "text": messages[0]['content'][0]['text'] + messages[0]['content'][1]['text']})
    return system_prompt_part


def build_user_messages(step: int, instruction, images,
                        min_pixels, max_pixels,
                        history=list(), enable_think=False,
                        think_tag_begin='<thinking>', think_tag_end='</thinking>', *,
                        dataset: Optional[str] = None):
    if step > 0 and dataset == 'androidcontrol_high':
        user_prompt = f''''''
        history = ''.join([f'Step {si + 1}: {_}; 'for si, _ in enumerate(history)]) if history else 'None'
        user_prompt += f'\nTask progress (You have done the following operation on the current device): {history}.\n'
    else:
        user_prompt = f'''The user query: \n{instruction}'''
        if dataset != 'androidcontrol_high':
            history = ''.join([f'Step {si + 1}: {_}; 'for si, _ in enumerate(history)]) if history else 'None'
            user_prompt += (f'\nTask progress (You have done the following operation'
                            f' on the current device): {history}.\n')
    if enable_think:
        user_prompt += ('\nBefore answering, explain your reasoning step-by-step in '
                        f'{think_tag_begin}{think_tag_end} tags, '
                        'and insert them before the <tool_call></tool_call> XML tags.')
        user_prompt += ('\nAfter answering, summarize your action in <conclusion></conclusion> tags, '
                        'and insert them after the <tool_call></tool_call> XML tags.')
    image_contents = [{"type": "image", "image": image_path,
                       "min_pixels": min_pixels, "max_pixels": max_pixels}
                       for image_path in images]
    query_contents = ([{"type": "text", "text": user_prompt + '\n'}]
                      if (step == 0 or dataset != 'androidcontrol_high') else
                      list())
    return [{"role": "user", "content": [*query_contents, *image_contents]}]


def _fetch_matched_exterior_braces(dstr: str):
    pairs = list()
    box = list()
    left_braces_count = dstr.count(r'{')
    right_braces_count = dstr.count(r'}')
    if left_braces_count * right_braces_count == 0:
        return list()
    elif left_braces_count <= right_braces_count:
        _start = 0
        for i, element in enumerate(dstr):
            if not box and i > _start:
                pairs.append((_start, i))
                continue
            if element == '{':
                if not box:
                    _start = i
                box.append(element)
            elif element == '}':
                if box:
                    box.pop()
    else:
        _stop = -1
        for i, element in enumerate(reversed(dstr)):
            if element == '}':
                if not box:
                    _stop = -1 - i
                box.append(element)
            elif element == '{':
                if box:
                    box.pop()
                else:
                    pairs.append((-i, ((_stop + 1) if _stop < -1 else None)))
    return pairs


def _try_parse_json_block(resp: str) -> Optional[Dict]:
    json_str = re.search(r'<tool_call>.*({.*}).*</tool_call>', resp, re.DOTALL).group(1)
    # * switch to AttributeError processing
    for pair in _fetch_matched_exterior_braces(json_str):
        try:
            result: Dict = json.loads(json_str[pair[0]:pair[1]])
            break
        except json.JSONDecodeError:
            pass
    else:
        raise json.JSONDecodeError('No parseable braced json block')  # * switch to json.JSONDecodeError processing
    if set.issuperset(set(result.keys()), {"name", "arguments"}) and 'action' in result['arguments']:
        return result
    elif 'action' in result:
        return dict(name="mobile_for_gui_owl",
                    arguments=result)
    else:
        raise KeyMismatchError('Keys not match')  # * switch to KeyMismatchError processing


def _try_extract_fields(resp: str) -> Optional[Dict]:
    try:
        function_name = re.search(r'"name".*:.*"(.*)".*"arguments"', resp, re.DOTALL).group(1)
        arguments = re.search(r'arguments".*:(.*)}', resp, re.DOTALL).group(1)
        arguments = json.loads(arguments.strip())
        if 'action' not in arguments:
            raise KeyMismatchError('Arguments does not contain requested field action')
    except AttributeError:
        raise KeyMismatchError('Keys not match')  # * switch to KeyMismatchError processing
    except json.JSONDecodeError:
        raise KeyMismatchError('Arguments not parseable')  # * switch to KeyMismatchError processing
    return dict(name=function_name, arguments=arguments)


def _enhanced_strip(dstr: str) -> str:
    try:
        stripped = re.search(r"^['\"\n\s\t]*(.*?)['\"\n\s\t]*$", dstr, re.DOTALL).group(1)
        return stripped if stripped else dstr
    except Exception:
        return dstr


def parse_response(resp: str):
    # * parse action
    try:
        decision = _try_parse_json_block(resp=resp)
    except (AttributeError, json.JSONDecodeError):
        decision = _try_extract_fields(resp=resp)
    # * parse thinking if necessary
    thinking, conclusion = '', ''
    try:
        thinking = re.search(r'<thinking>(.*)</thinking>', resp, re.DOTALL).group(1)
        conclusion = re.search(r'<conclusion>(.*)</conclusion>', resp, re.DOTALL).group(1)
    except Exception:
        pass
    decision: Dict[Literal['name', 'arguments'],
                    Union[str,
                          Dict[Union[Literal['action'], str],
                               Union[str, Any]]]]
    action = decision['arguments']['action']
    arguments = dict(_item for _item in decision['arguments'].items() if 'action' not in _item)
    return dict(action=action,
                arguments=arguments,
                thinking=_enhanced_strip(thinking),
                conclusion=_enhanced_strip(conclusion))


def gui_owl_2_minicpm(output_text, width, height):
    resized_width = width
    resized_height = height

    try:
        parsed_response: Dict[Literal['action', 'arguments', 'thinking', 'conclusion'],
                              Union[Dict, Any]] = parse_response(output_text)
    except Exception as e:
        print(f"Error, JSON is NOT valid: {e}")
        return {}

    if parsed_response['action'] == "click":
        x, y = parsed_response['arguments'].get('coordinate')
        x = x / resized_width * 1000
        y = y / resized_height * 1000
        return {"POINT": [int(x), int(y)]}

    elif parsed_response['action'] == 'long_press':
        x, y = parsed_response['arguments'].get('coordinate')
        x = x / resized_width * 1000
        y = y / resized_height * 1000
        time = 1000
        return {"POINT": [int(x), int(y)], "duration": time}

    elif parsed_response['action'] == 'swipe':
        x1, y1 = parsed_response['arguments'].get('coordinate')
        x2, y2 = parsed_response['arguments'].get('coordinate2')

        x1 = x1 / resized_width * 1000
        y1 = y1 / resized_height * 1000
        x2 = x2 / resized_width * 1000
        y2 = y2 / resized_height * 1000
        direction = get_direction([x1, y1], [x2, y2]) if None not in (x1, y1, x2, y2) else None
        return {"POINT": [int(x1), int(y1)], "to": direction}

    elif parsed_response['action'] == "type":
        content = parsed_response['arguments'].get('text')
        return {"TYPE": content}

    elif parsed_response['action'] == 'system_button':
        button = parsed_response['arguments'].get('button')
        if button == 'Back':
            return {"PRESS": "BACK"}
        elif button == 'Home':
            return {"PRESS": "HOME"}
        elif button == 'Enter':
            return {"PRESS": "ENTER"}
        elif button == 'Menu':
            return {"PRESS": "MENU"}

    elif parsed_response['action'] == 'open':
        app = parsed_response['arguments'].get('text')
        return {"OPEN_APP": app}

    elif parsed_response['action'] == 'terminate':
        return {"STATUS": "finish"}

    elif parsed_response['action'] == 'wait':
        return {"duration": 1000}

    print("Error, unrecognized action.")
    return {}


def _formulate_history_action(action: str, **arguments):
    decision = json.dumps(dict(name="mobile_for_gui_owl",
                               arguments=dict(action=action,
                                              **arguments)), ensure_ascii=False)
    response_without_thinking_simulation = ("<tool_call>\n"
                                            f"{decision}\n"
                                            "</tool_call>")
    return {"role": "assistant", "content": [{'type': 'text',
                                              "text": response_without_thinking_simulation}]}


def _formulate_history_user_message(aitw_action: dict, memorize_image: bool,
                                    min_pixels: int, max_pixels: int):
    step_id = aitw_action["step_id"]
    user_message_contents = list()
    if step_id == 0:
        user_message_contents.append({'type': 'text', 'text': f'''The user query: \n{aitw_action['instruction']}'''})
    if memorize_image:
        user_message_contents.append({"type": "image", "image": aitw_action["image_path"],
                                      "min_pixels": min_pixels, "max_pixels": max_pixels})
    return {"role": "user", "content": user_message_contents} if user_message_contents else None


def _formulate_history_messages(aitw_action: dict, memorize_image: bool,
                                min_pixels: int, max_pixels: int,
                                action: str, **arguments):
    user_message = _formulate_history_user_message(aitw_action=aitw_action,
                                                   memorize_image=memorize_image,
                                                   min_pixels=min_pixels, max_pixels=max_pixels)
    assistant_response_message = _formulate_history_action(action=action, **arguments)
    return [assistant_response_message] if user_message is None else [user_message, assistant_response_message]


def aitw_2_gui_owl_action(aitw_action, resized_height, resized_width, *,
                          min_pixels: int,
                          max_pixels: int,
                          memorize_image: bool = False):

    ex_action_type = aitw_action['result_action_type']

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            click_y_min = int(touch_yx[0] * resized_height)
            click_x_min = int(touch_yx[1] * resized_width)
            click_y_max = int(lift_yx[0] * resized_height)
            click_x_max = int(lift_yx[1] * resized_width)
            coordinate = [math.ceil((click_x_max + click_x_min) / 2),
                          math.ceil((click_y_max + click_y_min) / 2)]
            messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                                   min_pixels=min_pixels, max_pixels=max_pixels,
                                                   action='click',
                                                   coordinate=coordinate)
            progress = f'Click {coordinate}'
            return messages, progress
        else:
            touch_xy_new = [touch_yx[1], touch_yx[0]]
            lift_xy_new = [lift_yx[1], lift_yx[0]]
            direction = get_direction(touch_xy_new, lift_xy_new)
            x1 = int(touch_yx[1] * resized_width)
            y1 = int(touch_yx[0] * resized_height)
            x2 = int(lift_yx[1] * resized_width)
            y2 = int(lift_yx[0] * resized_height)
            messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                                   min_pixels=min_pixels, max_pixels=max_pixels,
                                                   action='swipe',
                                                   coordinate=[x1, y1],
                                                   coordinate2=[x2, y2])
            progress = f'Swipe {direction}'
            return messages, progress

    elif ex_action_type == ActionType.PRESS_BACK:
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='system_button',
                                               button='Back')
        progress = f'Back'
        return messages, progress

    elif ex_action_type == ActionType.PRESS_HOME:
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='system_button',
                                               button='Home')
        progress = f'Home'
        return messages, progress

    elif ex_action_type == ActionType.PRESS_ENTER:
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='system_button',
                                               button='Enter')
        progress = f'Enter'
        return messages, progress

    elif ex_action_type == ActionType.TYPE:
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='type',
                                               text=aitw_action['result_action_text'])
        progress = f"""type '{aitw_action['result_action_text']}')"""
        return messages, progress

    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='terminate',
                                               status='success')
        progress = f'Terminate with success'
        return messages, progress

    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='terminate',
                                               status='failure')
        progress = f'Terminate with failure'
        return messages, progress

    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        click_y_min = int(touch_yx[0] * resized_height)
        click_x_min = int(touch_yx[1] * resized_width)
        click_y_max = int(lift_yx[0] * resized_height)
        click_x_max = int(lift_yx[1] * resized_width)
        coordinate = [math.ceil((click_x_max + click_x_min) / 2),
                      math.ceil((click_y_max + click_y_min) / 2)]
        time = random.randint(3, 10)
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='long_press',
                                               coordinate=coordinate,
                                               time=time)
        progress = f'Long press point {coordinate} for {time} seconds'
        return messages, progress

    elif ex_action_type == ActionType.NO_ACTION:
        time = random.randint(3, 10)
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='wait',
                                               time=time)
        progress = f'Wait for {time} seconds'
        return messages, progress

    elif ex_action_type == ActionType.OPEN_APP:
        messages = _formulate_history_messages(aitw_action=aitw_action, memorize_image=memorize_image,
                                               min_pixels=min_pixels, max_pixels=max_pixels,
                                               action='open',
                                               text=aitw_action['result_action_app_name'])
        app = aitw_action['result_action_app_name']
        progress = f'Open App {app}'
        return messages, progress

    else:
        print('GUI_OWL Action: ', aitw_action)
        raise NotImplementedError

    return ""


def prepare_task_input(step, image_path, history_actions: list, data_name, _tokenizer, use_vllm, *,
                       enable_think: bool = True, image_memory: int = 2):
    query = step['instruction']
    if data_name == 'androidcontrol_low':
        query = step["low_instruction"]
    image = Image.open(image_path)
    resized_height, resized_width = smart_resize(
    image.height, image.width,
    min_pixels=_tokenizer.image_processor.min_pixels,
    max_pixels=_tokenizer.image_processor.max_pixels)
    images = [image]

    step_id = len(history_actions)
    recent_history = history_actions.copy()
    history = [
        aitw_2_gui_owl_action(aitw_action, resized_height, resized_width,
                              min_pixels=_tokenizer.image_processor.min_pixels,
                              max_pixels=_tokenizer.image_processor.max_pixels,
                              memorize_image=(-i <= image_memory - 1))
        for i, aitw_action in enumerate(recent_history, start=-step_id)
    ]
    history_messages, progresses = (list(), list()) if not history else list(zip(*history))
    history_messages = ([]
                        if not history_messages or data_name != 'androidcontrol_high' else
                        list(itertools.chain(*history_messages)))

    system_message = build_system_messages(resized_height=resized_height, resized_width=resized_width)
    user_messages = build_user_messages(step_id, query, [image_path],
                                        _tokenizer.image_processor.min_pixels,
                                        _tokenizer.image_processor.max_pixels,
                                        enable_think=enable_think, history=progresses,
                                        dataset=data_name)

    message = [
        system_message,
        *history_messages,
        *user_messages
    ]

    return step, message, images


def prepare_task_inputs(episode, episode_dir, episode_file, subset, dataset, _tokenizer, use_vllm, *,
                        enable_think: bool = True):

    res = []
    files_dir = os.path.join(episode_dir, episode_file)

    for index, step in enumerate(episode):
        step["category"] = subset
        history = []
        for prev_step in episode[:index]:
            if os.path.exists(prev_step['image_path']):
                image_path = (prev_step['image_path']
                              if os.path.isabs(prev_step['image_path']) else
                              os.path.abspath(prev_step['image_path']))
            else:
                image_suffices = {"jpeg", "png", "jpg"}
                suffix_pattern = '|'.join(r'_{step}\.({suffix})$'.format(step=prev_step['step_id'], suffix=suffix)
                                          for suffix in image_suffices)
                for _file in os.listdir(files_dir):
                    if re.search(suffix_pattern, _file):
                        break
                else:
                    raise FileNotFoundError(f'No image found step of episode')
                image_path = os.path.join(episode_dir, episode_file, _file)
                if os.path.exists(image_path):
                    image_path = os.path.abspath(image_path)
                else:
                    raise FileNotFoundError(f'Located image does not exist: {image_path}')

            history_action = {"step_id": prev_step['step_id'],
                              "instruction": prev_step['instruction'],
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

        res.append(prepare_task_input(step, image_path, history, dataset, _tokenizer, use_vllm,
                                      enable_think=enable_think))

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
            temperature=0.1,
            top_p=0.001,
            top_k=1,
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
        width, height = images[0].size
        try:
            step['pred'] = gui_owl_2_minicpm(str_res, width, height)
        except Exception:
            print("Error, JSON is NOT valid.")
            step['pred'] = {}
        res_steps.append(step)
    return res_steps
