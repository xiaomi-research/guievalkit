import json
import math
import numpy as np
import re
import json

from numbers import Number
from typing import Any, Literal, Union, Dict
from vllm import SamplingParams

from guieval.main import StepTaskModel
from guieval.utils import ActionType
from guieval.models.utils import *
from guieval.models.abcmodel import *
from guieval.utils.action_utils import is_tap_action, get_direction
from guieval.models.resources.glm_4_5v.prompt_builder import build as build_prompt

# section struct
ACTIONS = {'click', 'long_press', 'swipe', 'input_text', 'answer', 'keyboard_enter',
           'navigate_home', 'wait', 'status', "open_app"}
FIELDS = {'box_2d', 'text', 'override', "direction", "app_name"}


# section main
@ModelRegistry.register()
class GLM4_5v(ABCModel):
    NAMES = ("glm-4.5v", "glm-4.6v")
    MODEL_PATTERNS = ModelPatterns(conclusion_pattern=r'Memory:(.*?)Reason:',
                                   conclusion_flags=[re.DOTALL, ],
                                   thinking_pattern=r'<think>(.*)</think>|Reason:(.*?)Action:',
                                   thinking_flags=[re.DOTALL, ],
                                   answer_pattern=r'Action:(.*)',
                                   answer_flags=[re.DOTALL, ])
    DEFAULT_SAMPLING_PARAMS = {
        "glm-4.5v": SamplingParams(
            max_tokens=8192,
            temperature=0.001,
            skip_special_tokens=False),
        "glm-4.6v": SamplingParams(
            top_k=2,
            top_p=0.6,
            temperature=0.8,
            repetition_penalty=1.1,
            max_tokens=8192 * 2
        )}

    @first_level_parser.validate_patterns(MODEL_PATTERNS)
    def parse_response(self, parsed_matches):
        if parsed_matches['answer'] is not None:
            answer_str: str = parsed_matches['answer'].group(1)
            answer_str = answer_str.strip()

            if "<|begin_of_box|>" in answer_str:
                answer_str = answer_str[
                    answer_str.index("<|begin_of_box|>") + len("<|begin_of_box|>"): answer_str.rindex(
                        "<|end_of_box|>"
                    )
                ]

            answer = ParserTools.parse_json_dict_block(answer_str)
            action = answer.pop('action_type', None)
            thought = (None if parsed_matches['thought'] is None else
                       ParserTools.enhanced_strip(parsed_matches['thought'].group(1))
                       if parsed_matches['thought'].group(1) else
                       ParserTools.enhanced_strip(parsed_matches['thought'].group(2)))
            arguments = answer
            return dict(action=action,
                        arguments=arguments,
                        answer=answer_str,
                        thought=thought,
                        conclusion=(None if parsed_matches["conclusion"] is None else
                                    ParserTools.enhanced_strip(parsed_matches["conclusion"].group(1))))
        else:
            raise parsed_matches['error']

    def model_2_minicpm(self, output_text, *args) -> MINICPM_ACTION:
        try:
            parsed_response: Dict[Literal['action', 'arguments', 'thought', 'conclusion'],
                                    Union[Dict, Any]] = self.parse_response(output_text)
        except Exception as err:
            self._logger.debug(f"Error. No valid `ModelPatterns` Extraction:\n\t{err}")
            return {'action': None,
                    'arguments': dict()}
        if parsed_response['action'] == "click":
            [[xmin, ymin, xmax, ymax]] = parsed_response['arguments'].get('box_2d')
            x = math.ceil((xmin + xmax) / 2)
            y = math.ceil((ymin + ymax) / 2)
            return {'action': 'CLICK',
                    'arguments': {"POINT": (x, y)}}
        elif parsed_response['action'] == 'long_press':
            [[xmin, ymin, xmax, ymax]] = parsed_response['arguments'].get('box_2d')
            x = math.ceil((xmin + xmax) / 2)
            y = math.ceil((ymin + ymax) / 2)
            time = 1000
            return {'action': 'LONG_POINT',
                    'arguments': {"POINT": (x, y), "duration": time}}
        elif parsed_response['action'] == 'swipe':
            try:
                [[xmin, ymin, xmax, ymax]] = parsed_response['arguments'].get('box_2d')
                if all(isinstance(coordinate, Number) for coordinate in (xmin, ymin, xmax, ymax)):
                    point = (math.ceil((xmin + xmax) / 2), math.ceil((ymin + ymax) / 2))
                    direction = get_direction([xmin, ymin], [xmax, ymax])
                else:
                    raise ValueError()
            except Exception:
                point = (500, 500)
                direction = parsed_response['arguments'].get("direction")
            return {'action': 'SCROLL',
                    'arguments': {"POINT": point, "to": direction}}
        elif parsed_response['action'] == "input_text":
            content = parsed_response['arguments'].get('text')
            return {'action': 'TYPE',
                    'arguments': {"TYPE": content}}
        elif parsed_response['action'] == "keyboard_enter":
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "ENTER"}}
        elif parsed_response['action'] == "navigate_home":
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "HOME"}}
        elif parsed_response['action'] == "navigate_back":
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "BACK"}}
        elif parsed_response['action'] == "open_app":
            app = parsed_response['arguments'].get("app_name")
            return {'action': 'OPEN',
                    'arguments': {"OPEN_APP": app}}
        elif parsed_response['action'] == 'status':
            return {'action': 'STOP',
                    'arguments': {"STATUS": "finish"}}
        elif parsed_response['action'] == 'wait':
            return {'action': 'WAIT',
                    'arguments': {"duration": 1000}}
        elif parsed_response['action'] == 'answer':
            content = parsed_response['arguments'].get('text')
            return {'action': None,
                    'arguments': {'TYPE': content}}

        # We assign None to the action type during parsing.
        # sample with action type assigned to None would not be included in the final evaluation.
        # Action still reserved for symmetry.

        else:
            self._logger.info(f"Unrecognized action identified: {parsed_response['action']}")
            return {'action': None,
                    'arguments': dict()}

    def aitw_2_model_action(self, step_task: StepTaskModel, *args) -> MODEL_ACTION:
        action_type = step_task.result_action_type
        if action_type == ActionType.DUAL_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
                click_y_min = int(touch_yx[0])
                click_x_min = int(touch_yx[1])
                click_y_max = int(lift_yx[0])
                click_x_max = int(lift_yx[1])
                return {'action': 'click',
                        'arguments': dict(box_2d=[[click_x_min, click_y_min,
                                                   click_x_max, click_y_max]])}
            else:
                touch_xy_new = [touch_yx[1], touch_yx[0]]
                lift_xy_new = [lift_yx[1], lift_yx[0]]
                direction = get_direction(touch_xy_new, lift_xy_new)
                click_y_min = int(touch_yx[0])
                click_x_min = int(touch_yx[1])
                click_y_max = int(lift_yx[0])
                click_x_max = int(lift_yx[1])
                return {'action': 'swipe',
                        'arguments': dict(direction=direction,
                                          box_2d=[[click_x_min, click_y_min,
                                                   click_x_max, click_y_max]])}
        elif action_type == ActionType.PRESS_BACK:
            return {'action': 'navigate_back',
                    'arguments': dict()}
        elif action_type == ActionType.PRESS_HOME:
            return {'action': 'navigate_home',
                    'arguments': dict()}
        elif action_type == ActionType.PRESS_ENTER:
            return {'action': 'keyboard_enter',
                    'arguments': dict()}
        elif action_type == ActionType.TYPE:
            return {'action': 'input_text',
                    'arguments': dict(text=step_task.result_action_text,
                                      box_2d=[[0, 0, 999, 999]],
                                      override=False)}
        elif action_type == ActionType.STATUS_TASK_COMPLETE:
            return {'action': 'status',
                    'arguments': dict(goal_status='complete')}
        elif action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
            return {'action': 'status',
                    'arguments': dict(goal_status='infeasible')}
        elif action_type == ActionType.LONG_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            click_y_min = int(touch_yx[0])
            click_x_min = int(touch_yx[1])
            click_y_max = int(lift_yx[0])
            click_x_max = int(lift_yx[1])
            return {'action': 'long_press',
                    'arguments': dict(box_2d=[[click_x_min, click_y_min,
                                               click_x_max, click_y_max]])}
        elif action_type == ActionType.NO_ACTION:
            return {'action': 'wait',
                    'arguments': dict()}
        elif action_type == ActionType.OPEN_APP:
            return {'action': 'open_app',
                    'arguments': dict(app_name=step_task.result_action_app_name)}
        else:
            self._logger.info(f'Task {step_task.episode_id} step {step_task.step_id} '
                        f'Action type `{ActionType(action_type).name}` not supported for GLM-4.5V.')
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def model_2_contents(history_step_task, current_step_task, action, *,
                         expected_content_source) -> HistoryContent:
        def _reformulate_contents(conclusion: str | None, thought: str | None,
                                model_action: dict[Literal['action', 'arguments'], str | dict]) -> str:
            template = 'Memory: {memory}\nReason: {reason}\nAction: <|begin_of_box|>{action}<|end_of_box|>'
            action = dict(action_type=model_action.get('action'),
                        **model_action.get('arguments'))
            action_str = json.dumps(action, ensure_ascii=False)
            return template.format(memory=conclusion,
                                reason=thought,
                                action=action_str)
        # history content required by GLM-4.5v is the <think>.*</think> excluded entire repsonse.
        # see https://github.com/zai-org/GLM-V/blob/main/examples/gui-agent/glm-45v/agent.md for more details

        if expected_content_source == "online_pos" and history_step_task.evaluate().exact_match:
            response = history_step_task.response.split('</think>')[-1].strip()
            return {'content': response,
                    'source': 'online_pos'}
        elif (expected_content_source == "online_neg"
              and not history_step_task.evaluate().exact_match
              and history_step_task.pred_action is not None):
            response = history_step_task.response.split('</think>')[-1].strip()
            return {'content': response,
                    'source': 'online_neg'}
        elif ((expected_content_source == "low_instruction"
               or current_step_task.history_content_source_choices[1] == "low_instruction")
               and history_step_task.low_instruction):
            conclusion = history_step_task.low_instruction
            response = _reformulate_contents(conclusion=conclusion, thought=None, model_action=action)
            return {'content': response,
                    'source': 'low_instruction'}

        if action['action'] == 'click':
            coordinate = action['arguments'].get('box_2d')
            conclusion = f'Click {coordinate}' if coordinate else 'Click Unknown coordinate'
        elif action['action'] == 'swipe':
            direction = action['arguments'].get('direction')
            conclusion = f'Swipe {direction}' if direction else 'Swipe Unknown direction'
        elif action['action'] == 'navigate_back':
            conclusion = 'Back'
        elif action['action'] == 'navigate_home':
            conclusion = 'Go to home page'
        elif action['action'] == 'keyboard_enter':
            conclusion = 'Press keyboard enter'
        elif action['action'] == 'input_text':
            text = action['arguments'].get('text')
            conclusion = f'Type \'{text}\'' if text else 'Type \'\''
        elif action['action'] == 'status':
            status = action['arguments'].get('goal_status')
            conclusion = f'Terminate since task {status}' if status else 'Terminate with Unknown status'
        elif action['action'] == 'long_press':
            coordinate = action['arguments'].get('box_2d')
            conclusion = f'Long press {coordinate}' if coordinate else 'Long press Unknown coordinate for Unknown time'
        elif action['action'] == 'wait':
            conclusion = 'Wait for a while'
        elif action['action'] == 'open_app':
            app = action['arguments'].get('app_name')
            conclusion = f'Open App {app}' if app else 'Open Unknown app'
        else:
            conclusion = 'Unknown action'

        return {'content': _reformulate_contents(conclusion=conclusion,
                                                      thought=None,
                                                      model_action=action),
                'source': 'offline_rule'}

    def prepare_task_input(self, step_task: StepTaskModel, **kwargs):
        raw_input = super().prepare_task_input(step_task=step_task)

        prompt = build_prompt(instruction=raw_input['instruction'],
                              history=raw_input['history_contents'])
        formulated_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": step_task.image_abspaths[0]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        step_task.formulated_messages = formulated_messages
        step_task.history_content_srcs = raw_input['history_content_srcs']
        step_task.images = raw_input['step_images']

        return step_task
