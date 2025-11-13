import json
import math
import numpy as np
import random
import re
import logging
import json

from string import Template
from typing import Any, Literal, Sequence, Union, Dict
from vllm import SamplingParams

# subsec internal
from guieval.main import StepTaskModel
from guieval.utils import ActionType
from guieval.models.utils import *
from guieval.models.utils.abcmodel import *
from guieval.utils.action_utils import is_tap_action, get_direction
from guieval.models.resources.qwen_vl.sys_prompt_builder import build as build_sys_prompt

# section struct
logger = logging.getLogger(__name__)

MIN_PIXELS, MAX_PIXELS = 256 * 28 * 28, 1280 * 28 * 28

ACTIONS = {'click', 'long_press', 'swipe', 'type', 'answer', 'system_button', 'wait', 'terminate'}
FIELDS = {'coordinate', 'coordinate2', 'text', 'time', 'button', 'status'}

USER_QUERY_TEMPLATE = '''The user query:  ${instruction}
Task progress (You have done the following operation on the current device): ${history}'''

ANDROIDCONTROL_LOW_TEMPLATE = (
    "The user query:  ${instruction} \n"
    "Current step query: ${low_instruction}\n"
    "Task progress (You have done the following operation on the current device): ${history}"
)

GUI_ODYSSEY_TEMPLATE = (
    "The user query: ${instruction}\n"
    "Before answering, explain your reasoning step-by-step in <thinking></thinking> tags, "
    "and insert them before the <tool_call></tool_call> XML tags.\n"
    "After answering, summarize your action in <conclusion></conclusion> tags, "
    "and insert them after the <tool_call></tool_call> XML tags.\n"
    "Task progress (You have done the following operation on the current device):\n"
    "${history}"
)


class KeyMismatchError(Exception):
    """Raised when block keys don't match the expected keys."""


_SAMPLING_PARAMS = SamplingParams(
    max_tokens=2048,
    top_p=0.01,
    top_k=1,
    temperature=0.01,
    repetition_penalty=1.0)


def build_user_message(step_task: StepTaskModel, history_contents: Sequence[str],
                       height: int, width: int) -> dict:
    history = ' '.join(f'Step {idx + 1}: {_content};' for idx, _content in enumerate(history_contents))
    if step_task.dataset == 'gui_odyssey':
        query_template = Template(GUI_ODYSSEY_TEMPLATE)
    elif step_task.dataset == 'androidcontrol_low':
        query_template = Template(ANDROIDCONTROL_LOW_TEMPLATE)
    else:
        query_template = Template(USER_QUERY_TEMPLATE)
    user_query = query_template.safe_substitute(instruction=step_task.instruction,
                                                low_instruction=step_task.low_instruction,
                                                history=history)
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": build_sys_prompt(enable_conclude=step_task.enable_conclude,
                                                          enable_think=step_task.enable_think,
                                                          height=height, width=width)},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {"type": "image", "image": step_task.image_abspaths[0]},
            ],
        }
    ]


# section main
@ModelRegistry.register()
class Qwen2_5VL(ABCModel):
    NAMES = ("qwen2.5-vl-3b-instruct", "qwen2.5-vl-7b-instruct")
    MODEL_PATTERNS = ModelPatterns(answer_pattern=r'<tool_call>(.*)</tool_call>',
                                   answer_flags=[re.DOTALL, ],
                                   thinking_pattern=r'Thought:(.*?)(?=Action|<tool_call>)',
                                   thinking_flags=[re.DOTALL, ],
                                   conclusion_pattern=r'Action:(.*?)(?=<tool_call>)',
                                   conclusion_flags=[re.DOTALL, ])

    DEFAULT_SAMPLING_PARAMS = dict.fromkeys(NAMES, _SAMPLING_PARAMS)

    @first_level_parser.validate_patterns(MODEL_PATTERNS)
    def parse_response(self, parsed_matches):
        try:
            answer_str: str = parsed_matches['answer'].group(1)
            answer_str = answer_str.strip()

            try:
                answer_json_block = re.search(r'^[^{}]*({.*})[^{}]*$', answer_str, re.DOTALL).group(1)
                answer = json.loads(answer_json_block)
            except json.JSONDecodeError:
                answer = ParserTools.parse_json_dict_block(answer_str)
            except AttributeError:
                try:
                    function_name = re.search(r'"name".*:.*"(.*)".*"arguments"', answer_str, re.DOTALL).group(1)
                    arguments = re.search(r'arguments".*:(.*)}', answer_str, re.DOTALL).group(1)
                    arguments = json.loads(arguments.strip())
                    if 'action' not in arguments:
                        raise KeyMismatchError('Arguments does not contain requested field action')
                except AttributeError:
                    raise KeyMismatchError('Keys not match')
                except json.JSONDecodeError:
                    raise KeyMismatchError('Arguments not parseable')
                answer = dict(name=function_name, arguments=arguments)

            if set.issuperset(set(answer.keys()), {"name", "arguments"}) and 'action' in answer['arguments']:
                pass
            elif 'action' in answer:
                answer = dict(name="mobile_use", arguments=answer)
            else:
                raise KeyMismatchError('Keys not match')

            action = answer['arguments']['action']
            arguments = dict(_item for _item in answer['arguments'].items() if 'action' not in _item)
            return dict(action=action,
                        arguments=arguments,
                        answer=answer_str,
                        thinking=(None if parsed_matches["thinking"] is None else
                                  ParserTools.enhanced_strip(parsed_matches["thinking"].group(1))),
                        conclusion=(None if parsed_matches["conclusion"] is None else
                                    ParserTools.enhanced_strip(parsed_matches["conclusion"].group(1))))
        except Exception:
            raise parsed_matches['error']

    def model_2_minicpm(self, output_text, width, height) -> MINICPM_ACTION:
        try:
            parsed_response: Dict[Literal['action', 'arguments', 'thinking', 'conclusion'],
                                  Union[Dict, Any]] = self.parse_response(output_text)
        except Exception as err:
            logger.debug(f"Error. No valid `ModelPatterns` Extraction:\n\t{err}")
            return {'action': None,
                    'arguments': dict()}
        if parsed_response['action'] == "click" or parsed_response['action'] == "left_click":
            x, y = parsed_response['arguments'].get('coordinate')
            x = x / width * 1000
            y = y / height * 1000
            return {'action': 'CLICK',
                    'arguments': {"POINT": (int(x), int(y))}}
        elif parsed_response['action'] == 'long_press':
            x, y = parsed_response['arguments'].get('coordinate')
            x = x / width * 1000
            y = y / height * 1000
            time = 1000
            return {'action': 'LONG_POINT',
                    'arguments': {"POINT": (int(x), int(y)), "duration": time}}
        elif parsed_response['action'] == 'swipe':
            x1, y1 = parsed_response['arguments'].get('coordinate')
            x2, y2 = parsed_response['arguments'].get('coordinate2')

            x1 = x1 / width * 1000
            y1 = y1 / height * 1000
            x2 = x2 / width * 1000
            y2 = y2 / height * 1000
            x1, x2, y1, y2 = map(int, (x1, y1, x2, y2))
            direction = get_direction([x1, y1], [x2, y2]) if None not in (x1, y1, x2, y2) else None
            return {'action': 'SCROLL',
                    'arguments': {"POINT": (int(x1), int(y1)), "to": direction}}
        elif parsed_response['action'] == "type":
            content = parsed_response['arguments'].get('text')
            return {'action': 'TYPE',
                    'arguments': {"TYPE": content}}
        elif parsed_response['action'] == 'system_button':
            button = parsed_response['arguments'].get('button')
            if button == 'Back':
                return {'action': 'PRESS',
                        'arguments': {"PRESS": "BACK"}}
            elif button == 'Home':
                return {'action': 'PRESS',
                        'arguments': {"PRESS": "HOME"}}
            elif button == 'Enter':
                return {'action': 'PRESS',
                        'arguments': {"PRESS": "ENTER"}}
            elif button == 'Menu':
                return {'action': None,
                        'arguments': {"PRESS": "MENU"}}

            # action with predefined choices actually means special action for each.
            # which means predefined choice not in unified action space, is indeed a special action.
            # Thus we assign None to the action type during parsing
            # to avoid confusion between `match action` and `match arguments with matched action`
            # sample with action type assigned to None would not be included in the final evaluation.
            # Action still reserved for symmetry.

        elif parsed_response['action'] == 'open':
            app = parsed_response['arguments'].get('text')
            return {'action': 'OPEN',
                    'arguments': {"OPEN_APP": app}}
        elif parsed_response['action'] == 'terminate':
            return {'action': 'STOP',
                    'arguments': {"STATUS": "finish"}}
        elif parsed_response['action'] == 'wait':
            return {'action': 'WAIT',
                    'arguments': {"duration": 1000}}
        elif parsed_response['action'] == 'answer':
            content = parsed_response['arguments'].get('text')
            return {'action': None,
                    'arguments': {'TYPE': content}}
        else:
            logger.error(f"Unrecognized action: {parsed_response['action']}")
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def aitw_2_model_action(step_task, resized_height, resized_width) -> MODEL_ACTION:
        action_type = step_task.result_action_type
        if action_type == ActionType.DUAL_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
                click_y_min = int(touch_yx[0] * resized_height)
                click_x_min = int(touch_yx[1] * resized_width)
                click_y_max = int(lift_yx[0] * resized_height)
                click_x_max = int(lift_yx[1] * resized_width)
                coordinate = [math.ceil((click_x_max + click_x_min) / 2),
                            math.ceil((click_y_max + click_y_min) / 2)]
                return {'action': 'click',
                        'arguments': dict(coordinate=coordinate)}
            else:
                touch_xy_new = [touch_yx[1], touch_yx[0]]
                lift_xy_new = [lift_yx[1], lift_yx[0]]
                direction = get_direction(touch_xy_new, lift_xy_new)
                x1 = int(touch_yx[1] * resized_width)
                y1 = int(touch_yx[0] * resized_height)
                x2 = int(lift_yx[1] * resized_width)
                y2 = int(lift_yx[0] * resized_height)
                return {'action': 'swipe',
                        'arguments': dict(direction=direction, coordinate=[x1, y1], coordinate2=[x2, y2])}
        elif action_type == ActionType.PRESS_BACK:
            return {'action': 'system_button',
                    'arguments': dict(button='Back')}
        elif action_type == ActionType.PRESS_HOME:
            return {'action': 'system_button',
                    'arguments': dict(button='Home')}
        elif action_type == ActionType.PRESS_ENTER:
            return {'action': 'system_button',
                    'arguments': dict(button='Enter')}
        elif action_type == ActionType.TYPE:
            return {'action': 'type',
                    'arguments': dict(text=step_task.result_action_text)}
        elif action_type == ActionType.STATUS_TASK_COMPLETE:
            return {'action': 'terminate',
                    'arguments': dict(status='success')}
        elif action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
            return {'action': 'terminate',
                    'arguments': dict(status='failure')}
        elif action_type == ActionType.LONG_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            click_y_min = int(touch_yx[0] * resized_height)
            click_x_min = int(touch_yx[1] * resized_width)
            click_y_max = int(lift_yx[0] * resized_height)
            click_x_max = int(lift_yx[1] * resized_width)
            coordinate = [math.ceil((click_x_max + click_x_min) / 2),
                          math.ceil((click_y_max + click_y_min) / 2)]
            time = random.randint(3, 10)
            return {'action': 'long_press',
                    'arguments': dict(coordinate=coordinate, time=time)}
        elif action_type == ActionType.NO_ACTION:
            return {'action': 'wait',
                    'arguments': dict(time=random.randint(3, 10))}
        else:
            logger.info(f'Task {step_task.episode_id} step {step_task.step_id} '
                        f'Action type `{ActionType(action_type).name}` not supported for Qwen2.5-VL.')
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def model_2_contents(step_task, action, *, online: bool = False) -> HistoryContent:
        if online and step_task.evaluation.exact_match:
            return {'content': step_task.answer,  # history content required by qwen2.5vl is the answer.
                    'source': 'online'}
        else:
            answer = json.dumps(dict(name="mobile_use",
                                     arguments=dict(action=action.get('action'),
                                                    **action.get('arguments'))))
            return {'content': answer,
                    'source': 'offline_rule'}

    def prepare_task_input(self, step_task: StepTaskModel, *,
                           image_memory: int = 1,
                           **kwargs) -> StepTaskModel:
        raw_input = super().prepare_task_input(step_task=step_task)

        formulated_messages = build_user_message(step_task=step_task,
                                                 history_contents=raw_input['history_contents'],
                                                 height=raw_input['fetched_step_image_height'],
                                                 width=raw_input['fetched_step_image_width'])

        step_task.formulated_messages = formulated_messages
        step_task.history_content_srcs = raw_input['history_content_srcs']
        step_task.images = raw_input['step_images'][-image_memory:]

        return step_task
