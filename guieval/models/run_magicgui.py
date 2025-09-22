import math
import re
import json
import os
import logging
import numpy as np
from vllm import SamplingParams
from PIL import Image
from qwen_vl_utils import process_vision_info
from typing import Sequence
from guieval.utils.action_type import ActionType
from guieval.utils.action_utils import is_tap_action, get_direction

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(name=__name__)


def format_history_queries(history_queries: Sequence[str]) -> str:
    return '\n'.join(history_queries) + '\n'


BASE_PIXELS = 28 ** 2
MIN_PIXELS = 4 * BASE_PIXELS
MAX_PIXELS = 768 * BASE_PIXELS
INSTRUCTION_TEMPLATE = "\"{query}\"\nAction History: \"{history_queries}\""
PROMPT_TEMPLATE = ("You are a well-trained mobile intelligent agent "
                   "capable of assisting users with step-by-step navigation tasks. "
                   "Given the current smartphone screenshot <image> and the user instruction:\n"
                   "Instruction: {instruction}\n\n"
                   "Please output the correct function call to accomplish the user instruction. "
                   "Besides the function call, you should not output any other content.\n"
                   "You can call the following functions to control the smartphone.\n"
                   "- UI Basic Operations:\n"
                   "    1. tap(x: float,y: float) This function is used to click on a "
                          "specific point on the smartphone screen. "
                          "The coordinates x and y indicate the click position.  \n"
                   "    2. scroll(x: float, y: float,direction: str) "
                          "This function is used to swipe from the "
                          "starting coordinates (x, y) in the specified direction. "
                          "The coordinates x and y represent the center position of the control to be swiped. "
                          "The direction can be \"up\", \"down\", \"left\", or \"right\".\n"
                   "    3. text(x: float,y: float,text_input: str) "
                          "This function is used to input the specified text at the given coordinates. "
                          "The coordinates x and y represent the center position of the control to be clicked.\n"
                   "- Phone Key Operations:\n"
                   "    4. navigate_back() This function is used to return to the previous screen on the smartphone.\n"
                   "    5. navigate_home() This function is used to return to the home screen of the phone.\n"
                   "- Other Operations:\n"
                   "    6. long_press(x: float,y: float) This function is used to perform a "
                          "long press action at a specific point on the smartphone screen. "
                          "The coordinates x and y indicate the long press position.\n"
                   "    7. wait() This function is to wait at current page.\n"
                   "    8. finish() The user task is finished.\n    ")
PROMPT_TEMPLATE_CN = ('已知用户在界面<image>，提出了要求\n\"{instruction}"\n你认为合理的单步操作是什么？除了函数调用之外，你不能输出任何其他内容。你可以调用以下函数来控制智能手机：\n'
                      'UI基础操作：\n'
                      '1.tap(x,y) 该函数用于在智能手机屏幕上点击特定点，坐标 x 和 y 表示待点击控件中心位置。\n'
                      '2.scroll(x,y,direction) '
                        '该函数用于从起始坐标 (x,y) 开始在智能手机屏幕上滑动操作，'
                        'direction为手指滑动的方向，可以是 "up"、"down"、"left" 或 "right"。\n'
                      '3.text(x,y,text_input) 该函数用于在智能手机屏幕上输入指定的文本text_input。坐标 x 和 y 表示待点击控件的中心位置。\n'
                      '手机按键操作：\n'
                      '4. navigate_back() 该函数用于返回智能手机的上一个屏幕。\n'
                      '5. navigate_home() 该函数用于返回手机的home screen。\n其他操作：\n'
                      '6. long_press(x,y) 该函数用于在智能手机屏幕上的特定点执行长按操作。坐标 x 和 y 表示待点击控件的中心位置。\n'
                      '7. wait() 该函数表示在当前页面等候。\n'
                      '8. enter() 该函数表示按下enter键。\n'
                      '9. take_over(message) 该函数用于提示用户接管智能手机，其中 message 是提示用户接管手机的原因。如果原因不确定，请填写“请您接管当前界面”。\n'
                      '10. drag(x1,y1,x2,y2) 该函数执行一个对起始和终点敏感的拖动操作，表示手指从点(x1,y1)拖到点(x2,y2)。常见的场景包括滑块拖动、滚动选择器拖动和图片裁剪。\n'
                      '11. screen_shot() 该函数用于截图。\n'
                      '12. long_screen_shot() 该函数用于长截图。\n'
                      '13. call_api(api_name,operation) 对指定的APP进行操作。'
                          'api_name是API的名称。operation可以选择open或者kill。例如，call_api(Amazon, open)意味着打开亚马逊APP。\n'
                      '如果你发现当前指令无法在当前页面上执行，你需要输出no_answer()。如果你发现当前指令已完成，你需要输出action_completed()。')


def build_user_messages(instruction, images, min_pixels, max_pixels) -> list[dict]:
    image_contents = [{"type": "image", "image": image_path,
                       "min_pixels": min_pixels, "max_pixels": max_pixels}
                       for image_path in images]
    user_prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    query_contents = [{"type": "text", "text": user_prompt + '\n'}]
    contents = [*image_contents, *query_contents]
    return [{"role": "user", "content": contents}] if contents else list()


def parse_response(resp: str) -> tuple[str, list[str]]:
    pattern = re.compile(r'^([^\(\)]*)\((.*)\)$')
    action_str, arguments_str = pattern.search(resp).groups()
    action_name = action_str.strip().lower()
    arguments = [arg.strip() for arg in arguments_str.split(',')]
    return action_name, arguments


def magicgui_2_minicpm(output_text):
    try:
        action, arguments = parse_response(output_text)
        if action == 'tap':
            x, y = map(float, arguments)
            return {"POINT": [int(x), int(y)]}
        elif action == 'scroll':
            x, y, direction = arguments
            direction_choices = {"up", "down", "left", "right"}
            direction = direction.lower()
            if direction in direction_choices:
                raise ValueError(f'Predicted direction `{direction}` not in choices')
            x1, y1 = map(float, arguments[:2])
            return {"POINT": [int(x1), int(y1)],
                    "to": arguments[2]}
        elif action == 'drag':
            x1, y1, x2, y2 = map(float, arguments)
            direction = get_direction([x1, y1], [x2, y2])
            return {"POINT": [int(arguments[0]), int(arguments[1])],
                    "to": arguments[2]}
        elif action == 'text':
            content = ', '.join(arguments[2:])
            return {"TYPE": content}
        elif action == 'navigate_back':
            return {"PRESS": "BACK"}
        elif action in {'navigate_home', 'call_home'}:
            return {"PRESS": "HOME"}
        elif action == 'enter':
            return {"PRESS": "ENTER"}
        elif action == 'long_press':
            x, y = map(float, arguments)
            return {"POINT": [int(x), int(y)], "duration": 1000}
        elif action == 'wait':
            return {"duration": 1000}
        elif action in {'finish', 'action_completed', 'no_answer'}:
            return {"STATUS": "finish"}
        elif action == 'call_api':
            return {"OPEN_APP": arguments[0]}
    except AttributeError:
        logger.warning(msg=('Action Parsing for the following resp failed:\n'
                            f'\t\t\"{output_text}\"\n'
                            '\tReason:\n'
                            f'\t\tDoes not match pattern `action(arg1,arg2,...)`'))
        return dict()
    except Exception as e:
        logger.warning(msg=('Action Parsing for the following resp failed:\n'
                            f'\t\t\"{output_text}\"\n'
                            '\tReason:\n'
                            f'\t\t{repr(e)}'))
        return dict()


def aitw_2_magicgui_action(aitw_action):

    ex_action_type = aitw_action['result_action_type']

    if ex_action_type == ActionType.DUAL_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
            click_y_min = int(touch_yx[0] * 1000)
            click_x_min = int(touch_yx[1] * 1000)
            click_y_max = int(lift_yx[0] * 1000)
            click_x_max = int(lift_yx[1] * 1000)
            x, y = [math.ceil((click_x_max + click_x_min) / 2),
                    math.ceil((click_y_max + click_y_min) / 2)]
            return f'tap({int(x)},{int(y)})'
        else:
            touch_xy_new = [touch_yx[1], touch_yx[0]]
            lift_xy_new = [lift_yx[1], lift_yx[0]]
            direction = get_direction(touch_xy_new, lift_xy_new)
            x1 = int(touch_yx[1] * 1000)
            y1 = int(touch_yx[0] * 1000)
            x2 = int(lift_yx[1] * 1000)
            y2 = int(lift_yx[0] * 1000)
            center_x = math.ceil((x1 + x2) / 2)
            center_y = math.ceil((y1 + y2) / 2)
            return f'scroll({int(center_x)},{int(center_y)},{direction})'
    elif ex_action_type == ActionType.PRESS_BACK:
        return 'navigate_back()'
    elif ex_action_type == ActionType.PRESS_HOME:
        return 'navigate_home()'
    elif ex_action_type == ActionType.PRESS_ENTER:
        return 'enter()'
    elif ex_action_type == ActionType.TYPE:
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        try:
            touch_xy_new = [touch_yx[1], touch_yx[0]]
            x = int(touch_yx[1] * 1000)
            y = int(touch_yx[0] * 1000)
        except Exception:
            x, y = 500, 500
        text = aitw_action['result_action_text']
        return f'text({x},{y},{text})'
    elif ex_action_type == ActionType.STATUS_TASK_COMPLETE:
        return 'finish()'
    elif ex_action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
        return 'no_answer()'
    elif ex_action_type == ActionType.LONG_POINT:
        lift_yx = json.loads(aitw_action['result_lift_yx'])
        touch_yx = json.loads(aitw_action['result_touch_yx'])
        click_y_min = int(touch_yx[0] * 1000)
        click_x_min = int(touch_yx[1] * 1000)
        click_y_max = int(lift_yx[0] * 1000)
        click_x_max = int(lift_yx[1] * 1000)
        coordinate = [math.ceil((click_x_max + click_x_min) / 2),
                      math.ceil((click_y_max + click_y_min) / 2)]
        x, y = map(int, coordinate)
        return f'long_press({x},{y})'
    elif ex_action_type == ActionType.NO_ACTION:
        return 'wait()'
    elif ex_action_type == ActionType.OPEN_APP:
        app = aitw_action['result_action_app_name']
        return f'call_api({app},open)'
    else:
        print('MagicGUI Action: ', aitw_action)
        raise NotImplementedError

    return ""


def prepare_task_input(step, image_path, history_actions: list, data_name, use_vllm):
    query = step['instruction']
    if data_name == 'androidcontrol_low':
        query = step["low_instruction"]

    images = [image_path, ]
    history_queries = [(_action.get("low_instruction")
                        if _action.get("low_instruction") else
                        aitw_2_magicgui_action(_action))
                       for _action in history_actions]
    dedupped_history_queries = list()
    for _query in history_queries:
        if not dedupped_history_queries or dedupped_history_queries[-1] != _query:
            dedupped_history_queries.append(_query)
    instruction = INSTRUCTION_TEMPLATE.format(query=query,
                                              history_queries=format_history_queries(dedupped_history_queries))
    messages = build_user_messages(instruction, images, MIN_PIXELS, MAX_PIXELS)

    return step, messages, images


def prepare_task_inputs(episode, episode_dir, episode_file, subset, dataset, use_vllm):

    res = []
    files_dir = os.path.join(episode_dir, episode_file)

    for index, step in enumerate(episode):
        step["category"] = subset
        history = []
        for idx, prev_step in enumerate(episode[:index]):
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

            history_action = {"step_id": idx,
                              "instruction": prev_step['instruction'],
                               "observation": prev_step.get('observation', ''),
                              "result_action_type": prev_step['result_action_type'],
                              "result_action_text": prev_step['result_action_text'],
                              "result_touch_yx": prev_step['result_touch_yx'],
                              "result_lift_yx": prev_step['result_lift_yx'],
                              "low_instruction": prev_step.get("low_instruction"),
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

        res.append(prepare_task_input(step, image_path, history, dataset, use_vllm))

    return res


def run_task_batch(_llm, _tokenizer, batch_tasks, use_vllm):
    batch_steps = []
    batch_inputs = []
    batch_images: list[list[Image.Image]] = []
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
            repetition_penalty=1.0,
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
        step['pred'] = magicgui_2_minicpm(str_res)
        res_steps.append(step)
    return res_steps
