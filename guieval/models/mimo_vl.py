import json
import numpy as np
import re
import logging
import json

from typing import Any, Literal, Union, Dict
from vllm import SamplingParams

# subsec internal
from guieval.main import StepTaskModel
from guieval.utils import ActionType
from guieval.models.utils import *
from guieval.models.utils.abcmodel import *
from guieval.utils.action_utils import is_tap_action, get_direction
from guieval.models.resources.mimo_vl.prompt_builder import build as build_prompt


# section struct
logger = logging.getLogger(__name__)

ACTIONS = {"click", "longpress", "input", "drag", "press", "wait", "finished"}
FIELDS = {"start_point", "end_point", "text", "keys", "status"}

_SAMPLING_PARAMS = SamplingParams(
    max_tokens=2048,
    temperature=0.01,
    top_p=0.01,
    top_k=1,
    repetition_penalty=1.0,
    n=1
)


# section main
@ModelRegistry.register()
class MiMo_VL(ABCModel):
    NAMES = ("mimo-vl-7b-sft", "mimo-vl-7b-rl", "mimo-vl-7b-sft-2508", "mimo-vl-7b-rl-2508")
    MODEL_PATTERNS = ModelPatterns(answer_pattern=r'.*',
                                   answer_flags=[re.DOTALL, ],
                                   thinking_pattern=r'<think>(.*)</think>',
                                   thinking_flags=[re.DOTALL],
                                   conclusion_pattern=None)
    DEFAULT_SAMPLING_PARAMS = dict.fromkeys(NAMES, _SAMPLING_PARAMS)

    @first_level_parser.validate_patterns(MODEL_PATTERNS)
    def parse_response(self, parsed_matches):
        try:
            answer_str: str = parsed_matches['answer'].group()
            answer = ParserTools.parse_json_dict_block(answer_str)
            action = answer.pop('action', None)
            arguments = answer
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
        if parsed_response['action'] == "click":
            x, y = parsed_response['arguments'].get('start_point')
            x = x / width * 1000
            y = y / height * 1000
            return {'action': 'CLICK',
                    'arguments': {"POINT": (int(x), int(y))}}
        elif parsed_response['action'] == "longpress":
            x, y = parsed_response['arguments'].get('start_point')
            x = x / width * 1000
            y = y / height * 1000
            time = 1000
            return {'action': 'LONG_POINT',
                    'arguments': {"POINT": (int(x), int(y)), "duration": time}}
        elif parsed_response['action'] == "input":
            content = parsed_response['arguments'].get('text')
            return {'action': 'TYPE',
                    'arguments': {"TYPE": content}}
        elif parsed_response['action'] == "scroll":
            direction = parsed_response['arguments'].get('direction', 'no_direction')
            return {'action': 'SCROLL',
                    'arguments': {"POINT": (500, 500), "to": direction}}
        elif parsed_response['action'] == "drag":
            x, y = parsed_response['arguments'].get('start_point')
            x = x / width * 1000
            y = y / height * 1000
            x2, y2 = parsed_response['arguments'].get('end_point')
            x2 = x2 / width * 1000
            y2 = y2 / height * 1000
            direction = get_direction([x2, y2], [x, y])
            return {'action': 'SCROLL',
                    'arguments': {"POINT": (int(x), int(y)), "to": direction}}
        elif parsed_response['action'] == 'press':
            button = parsed_response['arguments'].get('keys', list())
            if 'back' in button:
                return {'action': 'PRESS',
                        'arguments': {"PRESS": "BACK"}}
            elif 'home' in button:
                return {'action': 'PRESS',
                        'arguments': {"PRESS": "HOME"}}
            elif 'enter' in button:
                return {'action': 'PRESS',
                        'arguments': {"PRESS": "ENTER"}}
        elif parsed_response['action'] == "wait":
            return {'action': 'WAIT',
                    'arguments': {"duration": 1000}}
        elif parsed_response['action'] == "finished":
            return {'action': 'STOP',
                    'arguments': {"STATUS": "finish"}}
        elif parsed_response['action'] == "open":
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
                        'arguments': dict(start_point=[(click_x_min + click_x_max) // 2,
                                                       (click_y_min + click_y_max) // 2])}
            else:
                x1 = int(touch_yx[1] * resized_width)
                y1 = int(touch_yx[0] * resized_height)
                x2 = int(lift_yx[1] * resized_width)
                y2 = int(lift_yx[0] * resized_height)
                return {'action': 'drag',
                        'arguments': dict(start_point=[x1, y1], end_point=[x2, y2])}
        elif action_type == ActionType.LONG_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            click_y_min = int(touch_yx[0] * resized_height)
            click_x_min = int(touch_yx[1] * resized_width)
            click_y_max = int(lift_yx[0] * resized_height)
            click_x_max = int(lift_yx[1] * resized_width)
            box = ((click_x_min + click_x_max) // 2, (click_y_min + click_y_max) // 2)
            return {'action': 'longpress',
                    'arguments': dict(start_point=[box[0], box[1]])}
        elif action_type == ActionType.PRESS_BACK:
            return {'action': 'press',
                    'arguments': dict(keys=["back"])}
        elif action_type == ActionType.PRESS_HOME:
            return {'action': 'press',
                    'arguments': dict(keys=["home"])}
        elif action_type == ActionType.PRESS_ENTER:
            return {'action': 'press',
                    'arguments': dict(keys=["enter"])}
        elif action_type == ActionType.TYPE:
            return {'action': 'input',
                    'arguments': dict(text=step_task.result_action_text)}
        elif action_type == ActionType.STATUS_TASK_COMPLETE:
            return {'action': 'finished',
                    'arguments': dict(status='success')}
        elif action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
            return {'action': 'finished',
                    'arguments': dict(status='failed')}
        elif action_type == ActionType.OPEN_APP:
            return {'action': 'open',
                    'arguments': dict(app_name=step_task.result_action_app_name)}
        else:
            logger.info(f'Task {step_task.episode_id} step {step_task.step_id} '
                        f'Action type `{ActionType(action_type).name}` not supported for MiMo-VL.')
            return {'action': None,
                    'arguments': dict()}

    def model_2_contents(self,
                         step_task: StepTaskModel,
                         action: MODEL_ACTION, *,
                         online: bool = False) -> HistoryContent:
        # history content required by ui-tars-1.5 is the original answer message
        if online and step_task.evaluation.exact_match:
            answer = step_task.response.split('</think>')[-1]
            return {'content': answer.strip(),
                    'source': 'online'}
        else:
            return {'content': str(action),
                    'source': 'offline_rule'}

    def prepare_task_input(self, step_task: StepTaskModel, **kwargs):
        raw_input = super().prepare_task_input(step_task=step_task)

        # todo integrate to jinja2 template
        history_action_str = ""
        for idx, (history_action, history_content) in enumerate(zip(raw_input['filled_history'],
                                                                    raw_input['history_contents'])):
            if history_action.low_instruction:
                history_action_str += (
                    f"Step {idx}: {history_action.low_instruction} "
                    f"{history_content}.")
            else:
                history_action_str += f"Step {idx}: {history_content}."
        history_action_str = "None" if history_action_str == "" else history_action_str

        prompt = build_prompt(step_task=step_task, history=history_action_str)

        # todo wrap
        messages = []
        content = [
            {"type": "image", "image": step_task.image_abspaths[0]},
            {"type": "text", "text": prompt}
        ]
        if not step_task.enable_think:
            content.append({"type": "text", "text": " /no_think"})
        messages.append(
            {
                "role": "user",
                "content": content
            }
        )

        step_task.formulated_messages = messages
        step_task.history_content_srcs = raw_input['history_content_srcs']
        step_task.images = raw_input['step_images']

        return step_task
