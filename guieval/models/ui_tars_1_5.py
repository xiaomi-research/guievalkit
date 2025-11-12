# section import
import itertools
import json
import re
import logging
import json
import numpy as np

from typing import Any, Literal, Union, Dict
from vllm import SamplingParams

# internal
from guieval.main import StepTaskModel
from guieval.utils import ActionType
from guieval.models.utils import *
from guieval.models.utils.abcmodel import *
from guieval.utils.action_utils import is_tap_action, get_direction
from guieval.models.resources.ui_tars_1_5.instruction_message_builder import build as build_instruction_message


# section struct
logger = logging.getLogger(__name__)

ACTIONS = {"click", "long_press", "type", "scroll", "drag", "press_back", "press_home", "wait", "finished", "open_app"}
FIELDS = {"start_box", "end_box", "direction", "content", "app_name"}
MIN_PIXELS = None  # 100 * 28 * 28
MAX_PIXELS = None  # 4096 * 28 * 28


# section main
@ModelRegistry.register()
class UITars1_5(ABCModel):
    NAMES = ("ui-tars-1.5-7b", )
    MODEL_PATTERNS = ModelPatterns(answer_pattern=r'Action:(.*)',
                                   answer_flags=[re.DOTALL, ],
                                   thinking_pattern=r'.*Thought:(.*)Action',
                                   thinking_flags=[re.DOTALL, ],
                                   conclusion_pattern=None)
    FIELD_PATTERNS = {
        "start_box": re.compile(r'(start_box)[^=]*=\'\((.*)\)\''),
        "end_box": re.compile(r'(end_box)[^=]*=\'\((.*)\)\''),
        "direction": re.compile(r'(direction)[^=]*=\'(down|up|right|left)\''),
        "content": re.compile(r'(content)[^=]*=\'(.*)\''),
        "app_name": re.compile(r'(app_name)[^=]*=\'(.*)\''),
    }
    DEFAULT_SAMPLING_PARAMS = {
        "ui-tars-1.5-7b": SamplingParams(
        max_tokens=512,
        temperature=0.1)
    }

    @first_level_parser.validate_patterns(MODEL_PATTERNS)
    def parse_response(self, parsed_matches):
        def parse_field(field: str, values: str):
            if field in {"start_box", "end_box"}:
                coordinates = re.findall(r'\d+', values)
                assert len(coordinates) >= 2
                return tuple(map(float, coordinates[:2]))
            else:
                return values.strip()

        try:
            answer_str: str = parsed_matches['answer'].group(1)
            answer_str = ParserTools.enhanced_strip(answer_str, extra=['`', ])
            action, arg_str = re.search(r'([^\(\)]*)\((.*)\)', answer_str).groups()
            action = action.strip()
            if action not in ACTIONS:
                raise ValueError(f'Invalid action: {action}')
            matches = filter(lambda _item: bool(_item[1]),
                            (
                                (field, pattern.search(arg_str))
                                for field, pattern in self.FIELD_PATTERNS.items()))
            arguments = dict((field, parse_field(field, values=match.group(2)))
                            for field, match in matches)
            if 'press' not in action and action != 'wait' and not arguments:
                raise ValueError(f'No valid arg parsing for: {arg_str}')
        except Exception as err:
            logger.error(f"Error occurred during parsing resp:\n"
                         f"\t{repr(err)}")
            raise parsed_matches['error']  # we recommend reraise the parsing error for more clear error handling <save>

        return dict(action=action,
                    arguments=arguments,
                    answer=answer_str,
                    thinking=(None if parsed_matches["thinking"] is None else
                              ParserTools.enhanced_strip(parsed_matches["thinking"].group(1), extra=['`', ])),
                    conclusion=None)

    def model_2_minicpm(self, output_text, width, height) -> MINICPM_ACTION:
        try:
            parsed_response: Dict[Literal['action', 'arguments', 'thinking', 'conclusion'],
                                Union[Dict, Any]] = self.parse_response(output_text)
        except Exception as err:
            logger.error(f"Error. No valid `ModelPatterns` Extraction:\n\t{err}")
            return {'action': None,
                    'arguments': dict()}
        if parsed_response['action'] == "click":
            x, y = parsed_response['arguments'].get('start_box')
            x = x / width * 1000
            y = y / height * 1000
            return {'action': 'CLICK',
                    'arguments': {"POINT": (int(x), int(y))}}
        elif parsed_response['action'] == "long_press":
            x, y = parsed_response['arguments'].get('start_box')
            x = x / width * 1000
            y = y / height * 1000
            time = 1000
            return {'action': 'LONG_POINT',
                    'arguments': {"POINT": (int(x), int(y)), "duration": time}}
        elif parsed_response['action'] == "type":
            content = parsed_response['arguments'].get('content')
            return {'action': 'TYPE',
                    'arguments': {"TYPE": content}}
        elif parsed_response['action'] == "scroll":
            x, y = parsed_response['arguments'].get('start_box')
            x = x / width * 1000
            y = y / height * 1000
            direction = parsed_response['arguments'].get('direction')
            reverse_map = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}
            return {'action': 'SCROLL',
                    'arguments': {"POINT": (int(x), int(y)), "to": reverse_map.get(direction)}}
        elif parsed_response['action'] == "drag":
            x, y = parsed_response['arguments'].get('start_box')
            x = x / width * 1000
            y = y / height * 1000
            x2, y2 = parsed_response['arguments'].get('end_box')
            x2 = x2 / width * 1000
            y2 = y2 / height * 1000
            direction = get_direction([x2, y2], [x, y])
            return {'action': 'SCROLL',
                    'arguments': {"POINT": (int(x), int(y)), "to": direction}}
        elif parsed_response['action'] == "press_back":
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "BACK"}}
        elif parsed_response['action'] == "press_home":
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "HOME"}}
        elif parsed_response['action'] == "wait":
            return {'action': 'WAIT',
                    'arguments': {"duration": 1000}}
        elif parsed_response['action'] == "finished":
            return {'action': "STOP",
                    'arguments': {"STATUS": "finish"}}
        elif parsed_response['action'] == "open_app":
            app_name = parsed_response['arguments'].get('app_name')
            return {'action': 'OPEN',
                    'arguments': {"OPEN_APP": app_name}}
        else:
            logger.info(f"Unrecognized action during remormulation: {parsed_response['action']}")
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def aitw_2_model_action(step_task: StepTaskModel, resized_height: int, resized_width: int) -> MODEL_ACTION:
        action_type = step_task.result_action_type

        if action_type == ActionType.DUAL_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
                click_y_min = int(touch_yx[0] * resized_height)
                click_x_min = int(touch_yx[1] * resized_width)
                click_y_max = int(lift_yx[0] * resized_height)
                click_x_max = int(lift_yx[1] * resized_width)
                return {'action': 'click',
                        'arguments': dict(start_box=(f'({(click_x_min + click_x_max) // 2},'
                                                     f'{(click_y_min + click_y_max) // 2})'))}
            else:
                touch_xy_new = [touch_yx[1], touch_yx[0]]
                lift_xy_new = [lift_yx[1], lift_yx[0]]
                direction = get_direction(touch_xy_new, lift_xy_new)
                x1 = int(touch_yx[1] * resized_width)
                y1 = int(touch_yx[0] * resized_height)
                x2 = int(lift_yx[1] * resized_width)
                y2 = int(lift_yx[0] * resized_height)
                return {'action': 'scroll',
                        'arguments': dict(start_box=f'({x1},{y1})', end_box=f'({x2},{y2})', direction=direction)}
        elif action_type == ActionType.LONG_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            click_y_min = int(touch_yx[0] * resized_height)
            click_x_min = int(touch_yx[1] * resized_width)
            click_y_max = int(lift_yx[0] * resized_height)
            click_x_max = int(lift_yx[1] * resized_width)
            box = ((click_x_min + click_x_max) // 2, (click_y_min + click_y_max) // 2)
            return {'action': 'long_press',
                    'arguments': dict(start_box=f'({box[0]},{box[1]})')}
        elif action_type == ActionType.PRESS_BACK:
            return {'action': 'press_back'}
        elif action_type == ActionType.PRESS_HOME:
            return {'action': 'press_home'}
        elif action_type == ActionType.PRESS_ENTER:
            return {'action': 'press_enter'}
        elif action_type == ActionType.TYPE:
            return {'action': 'type',
                    'arguments': dict(content=step_task.result_action_text)}
        elif action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
            return {'action': 'finished'}
        elif action_type == ActionType.OPEN_APP:
            return {'action': 'open_app',
                    'arguments': dict(app_name=step_task.result_action_app_name)}
        else:
            logger.info(f'Task {step_task.episode_id} step {step_task.step_id} '
                        f'Action type `{ActionType(action_type).name}` not supported for UI-Tars-1.5.')
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def _wrap_answer(answer: str, thought: str | None) -> dict:
        content = f'Thought: {thought if thought else None}\nAction: {answer}'
        return dict(role='assistant',
                    content=content)

    def model_2_contents(self,
                         step_task: StepTaskModel,
                         action: MODEL_ACTION, *,
                         online: bool = False) -> HistoryContent:
        if online and step_task.evaluation.exact_match:
            return {'content': dict(role='assistant', content=step_task.response),
                    # history content required by uivenus_navi is the action/raw_answer.
                    'source': 'online'}
        if action['action'] == 'click':
            box = action['arguments'].get('start_box')
            return {'content': self._wrap_answer(answer=(f'click(start_box=\'{box}\')'
                                                         if box else
                                                         'click unknown start_box'),
                                                 thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        elif action['action'] == 'long_press':
            box = action['arguments'].get('start_box')
            return {'content': self._wrap_answer(answer=(f'long_press(start_box=\'{box}\')' if box
                                                         else
                                                         'long_press unknown start_box'),
                                                 thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        elif action['action'] == 'scroll':
            start_box = action['arguments'].get('start_box')
            direction = action['arguments'].get('direction')
            return {'content': self._wrap_answer(answer=(f'scroll(start_box=\'{start_box}\', direction=\'{direction}\')'
                                                         if start_box and direction else
                                                         'scroll unknown start_box or direction'),
                                                 thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        elif action['action'] == 'type':
            content = action['arguments'].get('content')
            return {'content': self._wrap_answer(answer=(f'type(content=\'{content}\')'
                                                         if content else
                                                         'type unknown content'),
                                                 thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        elif action['action'] == 'press_back':
            return {'content': self._wrap_answer(answer=f'press_back()', thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        elif action['action'] == 'press_home':
            return {'content': self._wrap_answer(answer=f'press_home()', thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        elif action['action'] == 'finished':
            content = action['arguments'].get('content')
            return {'content': self._wrap_answer(answer=(f'finished(content=\'{content}\')'
                                                         if content else
                                                         'finished without content'),
                                                 thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        elif action['action'] == 'open_app':
            app_name = action['arguments'].get('app_name')
            return {'content': self._wrap_answer(answer=(f'open_app(app_name=\'{app_name}\')'
                                                         if app_name else
                                                         'open_app unknown app_name'),
                                                 thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        elif action['action'] == 'drag':
            start_box = action['arguments'].get('start_box')
            end_box = action['arguments'].get('end_box')
            return {'content': self._wrap_answer(answer=(f'drag(start_box=\'{start_box}\', end_box=\'{end_box}\')'
                                                         if start_box and end_box else
                                                         'drag unknown start_box or end_box'),
                                                 thought=step_task.low_instruction),
                    'source': 'offline_rule'}
        else:
            return {'content': self._wrap_answer(answer='Unknown Action', thought=step_task.low_instruction),
                    'source': 'offline_rule'}

    def prepare_task_input(self, step_task: StepTaskModel, image_memory: int = 5, **kwargs) -> StepTaskModel:
        raw_input = super().prepare_task_input(step_task=step_task,
                                               min_pixels=MIN_PIXELS,
                                               max_pixels=MAX_PIXELS)
        filtered_history_indices = [_i
                                    for _i, _history_step in enumerate(raw_input['filled_history'])
                                    if _history_step.images]

        truncated_history_indices = filtered_history_indices[-4:]

        selected_history = [raw_input['filled_history'][_i] for _i in truncated_history_indices]
        selected_history_contents = [raw_input['history_contents'][_i] for _i in truncated_history_indices]
        selected_history_content_srcs = [raw_input['history_content_srcs'][_i] for _i in truncated_history_indices]

        history_observations = [dict(role="user", content=[dict(type="image",
                                                                image=_history_step_task.image_abspaths[0])])
                                for _history_step_task in selected_history]
        history_images = raw_input['history_images']

        image_message = {'role': 'user',
                         'content': [{
                             'type': 'image',
                             'image': step_task.image_abspaths[0]
                         }]}
        history_messages = itertools.chain.from_iterable(zip(history_observations, selected_history_contents))

        formulated_messages = [build_instruction_message(instruction=raw_input['instruction'],
                                                         enable_think=step_task.enable_think),
                               *history_messages,
                               image_message]

        step_task.formulated_messages = formulated_messages
        step_task.history_content_srcs = selected_history_content_srcs
        step_task.images = [*history_images, *raw_input['step_images']][-image_memory:]

        return step_task
