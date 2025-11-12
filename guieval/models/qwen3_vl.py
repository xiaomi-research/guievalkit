import json
import math
import numpy as np
import random
import re
import logging
import json

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

QUERY_TEMPLATE = ("The user query: {instruction}.\n"
                  "Task progress (You have done the following operation on the current device): {history}.\n")
ACTIONS = {'click', 'long_press', 'swipe', 'type', 'answer', 'system_button', 'wait', 'terminate'}
FIELDS = {'coordinate', 'coordinate2', 'text', 'time', 'button', 'status'}

_INSTRUCT_SAMPLING_PARAMS = SamplingParams(
    top_p=0.8,
    top_k=20,
    temperature=0.7,
    repetition_penalty=1.0,
    presence_penalty=1.5,
    max_tokens=2048,
    n=1)
_THINKING_SAMPLING_PARMAS = SamplingParams(
    top_p=0.95,
    top_k=20,
    repetition_penalty=1.0,
    presence_penalty=0.0,
    temperature=1.0,
    max_tokens=2048,
    n=1)


class KeyMismatchError(Exception):
    """Raised when block keys don't match the expected keys."""


def build_user_message(instruction: str, conclusions: Sequence[str], image_url: str, *,
                       enable_conclude: bool = True, enable_think: bool = True) -> dict:
    history = ' '.join(f'Step {idx + 1}: {_conclusion};' for idx, _conclusion in enumerate(conclusions))
    user_query = QUERY_TEMPLATE.format(instruction=instruction, history=history)
    return [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": build_sys_prompt(enable_conclude=enable_conclude,
                                                          enable_think=enable_think)},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_query},
                {"type": "image", "image": image_url},
            ],
        }
    ]


# section main
@ModelRegistry.register()
class Qwen3VL(ABCModel):
    NAMES = ("qwen3-vl-4b-instruct", "qwen3-vl-4b-thinking", "qwen3-vl-8b-instruct", "qwen3-vl-8b-thinking")
    MODEL_PATTERNS = ModelPatterns(answer_pattern=r'<tool_call>(.*)</tool_call>',
                                   answer_flags=[re.DOTALL, ],
                                   thinking_pattern=r'Thought:(.*?)(?=Action|<tool_call>)',
                                   thinking_flags=[re.DOTALL, ],
                                   conclusion_pattern=r'Action:(.*?)(?=<tool_call>)',
                                   conclusion_flags=[re.DOTALL, ])
    DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = {
        "qwen3-vl-4b-instruct": _INSTRUCT_SAMPLING_PARAMS,
        "qwen3-vl-4b-thinking": _THINKING_SAMPLING_PARMAS,
        "qwen3-vl-8b-instruct": _INSTRUCT_SAMPLING_PARAMS,
        "qwen3-vl-8b-thinking": _THINKING_SAMPLING_PARMAS
    }

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

    def model_2_minicpm(self, output_text, *args) -> MINICPM_ACTION:
        try:
            parsed_response: Dict[Literal['action', 'arguments', 'thinking', 'conclusion'],
                                  Union[Dict, Any]] = self.parse_response(output_text)
        except Exception as err:
            logger.debug(f"Error. No valid `ModelPatterns` Extraction:\n\t{err}")
            return {'action': None,
                    'arguments': dict()}

        if parsed_response['action'] == "click":
            x, y = parsed_response['arguments'].get('coordinate')
            return {'action': 'CLICK',
                    'arguments': {"POINT": (int(x), int(y))}}
        elif parsed_response['action'] == 'long_press':
            x, y = parsed_response['arguments'].get('coordinate')
            time = 1000
            return {'action': 'LONG_POINT',
                    'arguments': {"POINT": (int(x), int(y)), "duration": time}}
        elif parsed_response['action'] == 'swipe':
            x1, y1 = parsed_response['arguments'].get('coordinate')
            x2, y2 = parsed_response['arguments'].get('coordinate2')
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
            # Thus we assign None to the action type during parsing,
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
            logger.info(f"Unrecognized action during remormulation: {parsed_response['action']}")
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def aitw_2_model_action(step_task: StepTaskModel, *args) -> MODEL_ACTION:
        action_type = step_task.result_action_type
        if action_type == ActionType.DUAL_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
                click_y_min = int(touch_yx[0])
                click_x_min = int(touch_yx[1])
                click_y_max = int(lift_yx[0])
                click_x_max = int(lift_yx[1])
                coordinate = [math.ceil((click_x_max + click_x_min) / 2),
                            math.ceil((click_y_max + click_y_min) / 2)]
                return {'action': 'click',
                        'arguments': dict(coordinate=coordinate)}
            else:
                touch_xy_new = [touch_yx[1], touch_yx[0]]
                lift_xy_new = [lift_yx[1], lift_yx[0]]
                direction = get_direction(touch_xy_new, lift_xy_new)
                x1 = int(touch_yx[0])
                y1 = int(touch_yx[1])
                x2 = int(lift_yx[0])
                y2 = int(lift_yx[1])
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
            click_y_min = int(touch_yx[0])
            click_x_min = int(touch_yx[1])
            click_y_max = int(lift_yx[0])
            click_x_max = int(lift_yx[1])
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
                        f'Action type `{ActionType(action_type).name}` not supported for Qwen3-VL.')
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def model_2_contents(step_task, action, *, online: bool = False) -> HistoryContent:
        if online and step_task.evaluation.exact_match:
            return {'content': step_task.conclusion,  # history content required by Qwen3-VL is the action conclusion.
                    'source': 'online'}
        if action['action'] == 'click':
            coordinate = action['arguments'].get('coordinate')
            return {'content': f'Click {coordinate}' if coordinate else 'Click Unknown coordinate',
                    'source': 'offline_rule'}
        elif action['action'] == 'swipe':
            direction = action['arguments'].get('direction')
            return {'content': f'Swipe {direction}' if direction else 'Swipe Unknown direction',
                    'source': 'offline_rule'}
        elif action['action'] == 'system_button':
            button = action['arguments'].get('button')
            return {'content': f'{button}' if button else 'Unknown button',
                    'source': 'offline_rule'}
        elif action['action'] == 'type':
            text = action['arguments'].get('text')
            return {'content': f'Type {text}' if text else 'Type Unknown text',
                    'source': 'offline_rule'}
        elif action['action'] == 'terminate':
            status = action['arguments'].get('status')
            return {'content': f'Terminate with {status}' if status else 'Terminate with Unknown status',
                    'source': 'offline_rule'}
        elif action['action'] == 'long_press':
            coordinate = action['arguments'].get('coordinate')
            time = action['arguments'].get('time')
            return {'content': (f'Long press {coordinate} for {time} seconds'
                                if coordinate and time else
                                'Long press Unknown coordinate for Unknown time'),
                    'source': 'offline_rule'}
        elif action['action'] == 'wait':
            time = action['arguments'].get('time')
            return {'content': f'Wait {time} seconds' if time else 'Wait Unknown time',
                    'source': 'offline_rule'}
        elif action['action'] is None:
            return {'content': 'Unknown action',
                    'source': 'offline_rule'}

    def prepare_task_input(self, step_task: StepTaskModel, *,
                           image_memory: int = 1,
                           **kwargs) -> StepTaskModel:
        raw_input = super().prepare_task_input(step_task=step_task)

        formulated_messages = build_user_message(instruction=raw_input['instruction'],
                                                 conclusions=raw_input['history_contents'],
                                                 image_url=step_task.image_abspaths[0],
                                                 enable_conclude=step_task.enable_conclude,
                                                 enable_think=step_task.enable_think)

        step_task.formulated_messages = formulated_messages
        step_task.history_content_srcs = raw_input['history_content_srcs']
        step_task.images = raw_input['step_images'][-image_memory:]

        return step_task
