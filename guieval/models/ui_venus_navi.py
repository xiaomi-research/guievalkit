import json
import numpy as np
import re
import logging

from typing import Any, Literal, Union, Dict
from vllm import SamplingParams

from guieval.main import StepTaskModel
from guieval.utils import ActionType
from guieval.models.utils import *
from guieval.models.utils.abcmodel import *
from guieval.utils.action_utils import is_tap_action, get_direction
from guieval.models.resources.ui_venus_navi.prompt_builder import build as build_prompt

# section struct
logger = logging.getLogger(__name__)

VISION_CONFIG = {"ui-venus-navi-7b": dict(MAX_PIXELS=937664,
                                          MIN_PIXELS=830000),
                 "ui-venus-navi-72b": dict(MAX_PIXELS=12845056,
                                           MIN_PIXELS=3136)}
ACTIONS = {"Click", "Drag", "Scroll", "Type",  "Finished", "CallUser", "Wait",
           "LongPress", "PressBack", "PressHome", "PressEnter", "PressRecent", "Launch"}
FIELDS = {"box", "start", "end", "direction", "content", "app"}
_SAMPLING_PARAMS = SamplingParams(
    max_tokens=2048,
    temperature=0,
    top_p=1.0,
    top_k=-1,
    repetition_penalty=1.05,
    n=1,
    stop_token_ids=[])


# section main
@ModelRegistry.register()
class UI_Venus_Navi(ABCModel):
    NAMES = ("ui-venus-navi-7b", "ui-venus-navi-72b")
    MODEL_PATTERNS = ModelPatterns(answer_pattern=r'<action>(.*)</action>',
                                   answer_flags=[re.DOTALL, ],
                                   thinking_pattern=r'<think>(.*)</think>',
                                   thinking_flags=[re.DOTALL, ],
                                   conclusion_pattern=r'<conclusion>(.*)</conclusion>',
                                   conclusion_flags=[re.DOTALL, ])
    FIELD_PATTERNS = {
        "box": re.compile(r'(box)[^\)]*=\(([^\)]*)\)'),
        "start": re.compile(r'(start)[^\)]*=\(([^\)]*)\)'),
        "end": re.compile(r'(end)[^\)]*=\(([^\)]*)\)'),
        "direction": re.compile(r'(direction)[^\)]*=\'([^\']*)\''),
        "content": re.compile(r'(content)[^\)]*=\'([^\']*)\''),
        "app": re.compile(r'(app)[^\)]*=\'([^\']*)\''),
    }
    DEFAULT_SAMPLING_PARAMS = dict.fromkeys(NAMES, _SAMPLING_PARAMS)

    @first_level_parser.validate_patterns(MODEL_PATTERNS)
    def parse_response(self, parsed_matches):
        def parse_field(field: str, values: str):
            if field in {"box", "start", "end"}:
                return tuple(map(float, values.split(',')))
            elif field == 'direction':
                direction = values.strip()
                if direction not in {"down", "up", "right", "left"}:
                    raise AttributeError(f'Invalid direction: {direction}')
                return direction
            else:
                return values.strip()
        try:
            answer_str: str = parsed_matches['answer'].group(1)
            action, arguments = re.search(r'([^\(\)]*)\((.*)\)', answer_str).groups()
            lowered_actions = dict((_action.lower(), _action) for _action in ACTIONS)
            action = lowered_actions[action.strip().lower()]
            matches = filter(lambda _item: bool(_item[1]),
                            (
                                (field, pattern.search(arguments))
                                for field, pattern in self.FIELD_PATTERNS.items()))
            arguments = dict((field, parse_field(field, values=match.group(2)))
                            for field, match in matches)
            return dict(action=action,
                        arguments=arguments,
                        answer=(None if parsed_matches['answer'] is None else
                                parsed_matches["answer"].group(1)),
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
            logger.debug(str(err))
            return {'action': None,
                    'arguments': dict()}
        if parsed_response['action'] == "Click":
            x, y = parsed_response['arguments'].get('box')
            x = x / width * 1000
            y = y / height * 1000
            return {'action': 'CLICK',
                    'arguments': {"POINT": (int(x), int(y))}}
        elif parsed_response['action'] == "LongPress":
            x, y = parsed_response['arguments'].get('box')
            x = x / width * 1000
            y = y / height * 1000
            time = 1000
            return {'action': 'LONG_POINT',
                    'arguments': {"POINT": (int(x), int(y)), "duration": time}}
        elif parsed_response['action'] == "Scroll" or parsed_response['action'] == "Drag":
            x1, y1 = parsed_response['arguments'].get('start')
            x2, y2 = parsed_response['arguments'].get('end')
            direction = parsed_response['arguments'].get('direction')
            if direction and None in (x1, y1, x2, y2):
                return {'action': 'SCROLL',
                        'arguments': {"POINT": (500, 500), "to": direction}}

            x1 = x1 / width * 1000
            y1 = y1 / height * 1000
            x2 = x2 / width * 1000
            y2 = y2 / height * 1000
            direction = get_direction([x1, y1], [x2, y2])
            return {'action': 'SCROLL',
                    'arguments': {"POINT": (int(x1), int(y1)), "to": direction}}
        elif parsed_response['action'] == "Type":
            content = parsed_response['arguments'].get('content')
            return {'action': 'TYPE',
                    'arguments': {"TYPE": content}}
        elif parsed_response['action'] == "PressBack":
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "BACK"}}
        elif parsed_response['action'] == "PressHome":
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "HOME"}}
        elif parsed_response['action'] == "PressEnter":
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "ENTER"}}
        elif parsed_response['action'] == "PressRecent":
            return {'action': None,
                    'arguments': {"PRESS": "RECENT"}}

            # action with predefined choices actually means special action for each.
            # which means predefined choice not in unified action space, is indeed a special action.
            # Thus we assign None to the action type during parsing,
            # to avoid confusion between `match action` and `match arguments with matched action`
            # sample with action type assigned to None would not be included in the final evaluation.
            # Action still reserved for symmetry.

        elif parsed_response['action'] == "Finished":
            return {'action': 'STOP',
                    'arguments': {"STATUS": "finish"}}
        elif parsed_response['action'] == "Wait":
            return {'action': 'WAIT',
                    'arguments': {"duration": 1000}}
        elif parsed_response['action'] == "Launch":
            return {'action': 'OPEN',
                    'arguments': {"OPEN_APP": parsed_response['arguments'].get('app')}}
        else:
            logger.debug("Unrecognized action when parsing unified form prediction: "
                         f"{parsed_response['action']}")
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
                return {'action': 'Click',
                        'arguments': dict(box=((click_x_min + click_x_max) // 2, (click_y_min + click_y_max) // 2))}
            else:
                touch_xy_new = [touch_yx[1], touch_yx[0]]
                lift_xy_new = [lift_yx[1], lift_yx[0]]
                direction = get_direction(touch_xy_new, lift_xy_new)
                x1 = int(touch_yx[1] * resized_width)
                y1 = int(touch_yx[0] * resized_height)
                x2 = int(lift_yx[1] * resized_width)
                y2 = int(lift_yx[0] * resized_height)
                return {'action': 'Scroll',
                        'arguments': dict(start=(x1, y1), end=(x2, y2), direction=direction)}
        elif action_type == ActionType.PRESS_BACK:
            return {'action': 'PressBack',
                    'arguments': dict()}
        elif action_type == ActionType.PRESS_HOME:
            return {'action': 'PressHome',
                    'arguments': dict()}
        elif action_type == ActionType.PRESS_ENTER:
            return {'action': 'PressEnter',
                    'arguments': dict()}
        elif action_type == ActionType.TYPE:
            return {'action': 'Type',
                    'arguments': dict(content=step_task.result_action_text)}
        elif action_type == ActionType.STATUS_TASK_COMPLETE:
            return {'action': 'Finished',
                    'arguments': dict(content='')}
        elif action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
            return {'action': 'Finished',
                    'arguments': dict(content='')}
        elif action_type == ActionType.LONG_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            click_y_min = int(touch_yx[0] * resized_height)
            click_x_min = int(touch_yx[1] * resized_width)
            click_y_max = int(lift_yx[0] * resized_height)
            click_x_max = int(lift_yx[1] * resized_width)
            return {'action': 'LongPress',
                    'arguments': dict(box=((click_x_min + click_x_max) // 2, (click_y_min + click_y_max) // 2))}
        elif action_type == ActionType.NO_ACTION:
            return {'action': 'Wait',
                    'arguments': dict()}
        elif action_type == ActionType.OPEN_APP:
            return {'action': 'Launch',
                    'arguments': dict(app=step_task.result_action_app_name)}
        else:
            logger.debug(f'Task {step_task.episode_id} step {step_task.step_id} '
                         f'Action type {action_type} not supported for UIVENUS-NAVI.')
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def model_2_contents(step_task, action, *, online: bool = False) -> HistoryContent:
        if online and step_task.evaluation.exact_match:
            return {'content': step_task.answer,  # history content required by uivenus_navi is the answer content.
                    'source': 'online'}
        if action['action'] == 'Click':
            box = action['arguments'].get('box')
            return {'content': f'Click(box={box})' if box else 'Click Unknown Box',
                    'source': 'offline_rule'}
        elif action['action'] == 'Drag':
            start = action['arguments'].get('start')
            end = action['arguments'].get('end')
            return {'content': f'Drag(start={start}, end={end})' if start and end else 'Drag Unknown',
                    'source': 'offline_rule'}
        elif action['action'] == 'LongPress':
            box = action['arguments'].get('box')
            return {'content': f'LongPress(box={box})' if box else 'LongPress Unknown Box',
                    'source': 'offline_rule'}
        elif action['action'] == 'Scroll':
            start = action['arguments'].get('start')
            end = action['arguments'].get('end')
            direction = action['arguments'].get('direction')
            return {'content': (f'Scroll(start={start}, end={end}, direction=\'{direction}\')'
                                if start and end and direction else
                                'Scroll Unknown'),
                    'source': 'offline_rule'}
        elif action['action'] == 'Type':
            content = action['arguments'].get('content')
            return {'content': f'Type(content=\'{content}\')' if content else 'Type Unknown Content',
                    'source': 'offline_rule'}
        elif action['action'] == 'PressBack':
            return {'content': 'PressBack()',
                    'source': 'offline_rule'}
        elif action['action'] == 'PressHome':
            return {'content': 'PressHome()',
                    'source': 'offline_rule'}
        elif action['action'] == 'PressEnter':
            return {'content': 'PressEnter()',
                    'source': 'offline_rule'}
        elif action['action'] == 'PressRecent':
            # this action is actually not in unified action space. reserved for symmetry.
            return {'content': 'PressRecent()',
                    'source': 'offline_rule'}
        elif action['action'] == 'Finished':
            content = action['arguments'].get('content')
            return {'content': f'Finished(content=\'{content}\')' if content else 'Finished without content',
                    'source': 'offline_rule'}
        elif action['action'] == 'CallUser':
            # this action is actually not in unified action space. reserved for symmetry.
            content = action['arguments'].get('content')
            return {'content': f'CallUser(content=\'{content}\')' if content else 'CallUser without content',
                    'source': 'offline_rule'}
        elif action['action'] == 'Wait':
            return {'content': 'Wait()',
                    'source': 'offline_rule'}
        elif action['action'] == 'Launch':
            app = action['arguments'].get('app')
            return {'content': f'Launch(app=\'{app}\')' if app else 'Launch Unknown App',
                    'source': 'offline_rule'}
        else:
            return {'content': 'Unknown Action',
                    'source': 'offline_rule'}

    def prepare_task_input(self, step_task: StepTaskModel, **kwargs):
        raw_input = super().prepare_task_input(step_task=step_task)

        history_thoughts = [(_history_step_task.thinking
                             if (step_task.mode == 'semi_online' and
                                 _history_step_task.evaluation.exact_match and
                                 _history_step_task.thinking is not None) else
                             "")
                            for _history_step_task in raw_input['filled_history']]
        history_entries = [
            (
                f"Step {i}: <think>{thinking}</think>"
                f"<action>{content}</action>")
            for i, (thinking, content) in enumerate(zip(history_thoughts, raw_input['history_contents']))
        ]
        history_str = "\n".join(history_entries)

        problem = build_prompt(instruction=raw_input['instruction'],
                               previous_actions=history_str,
                               enable_think=step_task.enable_think)

        step_task.history_content_srcs = raw_input['history_content_srcs']
        step_task.formulated_messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": problem},
                        {
                            "type": "image",
                            "image": step_task.image_abspaths[0],
                            "min_pixels": VISION_CONFIG.get(step_task.model)["MIN_PIXELS"],
                            "max_pixels": VISION_CONFIG.get(step_task.model)["MAX_PIXELS"],
                        }
                    ],
                },
        ]
        step_task.images = raw_input['step_images']

        return step_task
