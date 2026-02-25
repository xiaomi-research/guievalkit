import json
import numpy as np
import re
import math
import logging
from vllm import SamplingParams

from guieval.main import StepTaskModel
from guieval.utils import ActionType
from guieval.models.utils import *
from guieval.models.abcmodel import *
from guieval.utils.action_utils import is_tap_action, get_direction
from guieval.models.resources.magicgui.prompt_builder import build as build_prompt


BASE_PIXELS = 28 ** 2
MIN_PIXELS = 4 * BASE_PIXELS
MAX_PIXELS = 768 * BASE_PIXELS

logger = logging.getLogger(__name__)


# section main
@ModelRegistry.register()
class MagicGUI(ABCModel):
    NAMES = ("magicgui-cpt", "magicgui-rft")
    MODEL_PATTERNS = ModelPatterns(answer_pattern=r'^([^\(\)]*)\((.*)\)$',
                                   answer_flags=[re.DOTALL, ],
                                   thinking_pattern=None,
                                   thinking_flags=[],
                                   conclusion_pattern=None,
                                   conclusion_flags=[])
    DEFAULT_SAMPLING_PARAMS = dict.fromkeys(NAMES, SamplingParams(n=1,
                                                                  temperature=0.1,
                                                                  top_p=0.001,
                                                                  top_k=1,
                                                                  repetition_penalty=1.0,
                                                                  stop_tokens_ids=[]))

    def parse_response(self, resp: str) -> tuple[str, list[str]]:
        pattern = re.compile(r'^([^\(\)]*)\((.*)\)$')
        action_str, arguments_str = pattern.search(resp).groups()
        action_name = action_str.strip().lower()
        arguments = [arg.strip() for arg in arguments_str.split(',')]
        return action_name, arguments

    def model_2_minicpm(self, output_text):
        try:
            action, arguments = self.parse_response(output_text)
        except Exception as err:
            logger.debug(str(err))
            return {'action': None,
                    'arguments': dict()}

        if action == 'tap':
            x, y = map(float, arguments)
            return {'action': 'CLICK',
                    'arguments': {"POINT": (int(x), int(y))}}
        elif action == 'scroll':
            x, y, direction = arguments
            direction_choices = {"up", "down", "left", "right"}
            direction = direction.lower()
            if direction not in direction_choices:
                raise ValueError(f'Predicted direction `{direction}` not in choices')
            x1, y1 = map(float, arguments[:2])
            return {'action': 'SCROLL',
                    'arguments': {"POINT": (int(x1), int(y1)),
                                  "to": arguments[2]}}
        elif action == 'drag':
            x1, y1, x2, y2 = map(float, arguments)
            direction = get_direction([x1, y1], [x2, y2])
            return {'action': 'DRAG',
                    'arguments': {"POINT": (int(x1), int(y1)),
                                  "to": arguments[2]}}
        elif action == 'text':
            content = ', '.join(arguments[2:])
            return {'action': 'TYPE',
                    'arguments': {"TYPE": content}}
        elif action == 'navigate_back':
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "BACK"}}
        elif action in {'navigate_home', 'call_home'}:
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "HOME"}}
        elif action == 'enter':
            return {'action': 'PRESS',
                    'arguments': {"PRESS": "ENTER"}}
        elif action == 'long_press':
            x, y = map(float, arguments)
            return {'action': 'LONG_POINT',
                    'arguments': {"POINT": (int(x), int(y)), "duration": 1000}}
        elif action == 'wait':
            return {'action': 'WAIT',
                    'arguments': {"duration": 1000}}
        elif action in {'finish', 'action_completed', 'no_answer'}:
            return {'action': 'STOP',
                    'arguments': {"STATUS": "finish"}}
        elif action == 'call_api':
            return {'action': 'OPEN',
                    'arguments': {"OPEN_APP": arguments[0]}}
        else:
            logger.debug("Unrecognized action when parsing unified form prediction: "
                         f"{action}")
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def aitw_2_model_action(step_task: StepTaskModel) -> MODEL_ACTION:
        action_type = step_task.result_action_type

        if action_type == ActionType.DUAL_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            if is_tap_action(np.array(touch_yx), np.array(lift_yx)):
                click_y_min = int(touch_yx[0] * 1000)
                click_x_min = int(touch_yx[1] * 1000)
                click_y_max = int(lift_yx[0] * 1000)
                click_x_max = int(lift_yx[1] * 1000)
                return {'action': 'tap',
                        'arguments': ((click_x_max + click_x_min) // 2, (click_y_max + click_y_min) // 2)}
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
                return {'action': 'scroll',
                        'arguments': (center_x, center_y, direction)}
        elif action_type == ActionType.PRESS_BACK:
            return {'action': 'navigate_back',
                    'arguments': tuple()}
        elif action_type == ActionType.PRESS_HOME:
            return {'action': 'navigate_home',
                    'arguments': tuple()}
        elif action_type == ActionType.PRESS_ENTER:
            return {'action': 'enter',
                    'arguments': tuple()}
        elif action_type == ActionType.TYPE:
            touch_yx = json.loads(step_task.result_touch_yx)
            try:
                touch_xy_new = [touch_yx[1], touch_yx[0]]
                x = int(touch_yx[1] * 1000)
                y = int(touch_yx[0] * 1000)
            except Exception:
                x, y = 500, 500
            text = step_task.result_action_text
            return {'action': 'text',
                    'arguments': (x, y, text)}
        elif action_type == ActionType.STATUS_TASK_COMPLETE:
            return {'action': 'finish',
                    'arguments': tuple()}
        elif action_type == ActionType.STATUS_TASK_IMPOSSIBLE:
            return {'action': 'no_answer',
                    'arguments': tuple()}
        elif action_type == ActionType.LONG_POINT:
            lift_yx = json.loads(step_task.result_lift_yx)
            touch_yx = json.loads(step_task.result_touch_yx)
            click_y_min = int(touch_yx[0] * 1000)
            click_x_min = int(touch_yx[1] * 1000)
            click_y_max = int(lift_yx[0] * 1000)
            click_x_max = int(lift_yx[1] * 1000)
            center_x = math.ceil((click_x_max + click_x_min) / 2)
            center_y = math.ceil((click_y_max + click_y_min) / 2)
            return {'action': 'long_press',
                    'arguments': (center_x, center_y)}
        elif action_type == ActionType.NO_ACTION:
            return {'action': 'wait',
                    'arguments': tuple()}
        elif action_type == ActionType.OPEN_APP:
            return {'action': 'call_api',
                    'arguments': (step_task.result_action_text, 'open')}
        else:
            logger.debug("Unrecognized action when converting AITW action to model action: "
                         f"{action_type}")
            return {'action': None,
                    'arguments': tuple()}

    @staticmethod
    def model_2_contents(history_step_task, current_step_task, action, *,
                         expected_content_source) -> HistoryContent:
        if expected_content_source == "online_pos" and history_step_task.evaluate().exact_match:
            return {'content': history_step_task.answer,  # history content required by magicgui is the answer.
                    'source': 'online_pos'}
        elif (expected_content_source == "online_neg"
              and not history_step_task.evaluate().exact_match
              and history_step_task.pred_action is not None):
            return {'content': history_step_task.answer,
                    'source': 'online_neg'}
        elif history_step_task.low_instruction:
            return {'content': history_step_task.low_instruction,
                    'source': 'low_instruction'}

        return {'content': f"{action['action']}({','.join(action['arguments'])})",
                'source': 'offline_rule'}

    def prepare_task_input(self, step_task: StepTaskModel, **kwargs):
        raw_input = super().prepare_task_input(step_task=step_task)

        prompt = build_prompt(instruction=raw_input['instruction'],
                              previous_actions=raw_input['history_contents'],
                              language='Chinese')

        step_task.history_content_srcs = raw_input['history_content_srcs']

        image_contents = [{"type": "image", "image": image_path,
                           "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}
                          for image_path in step_task.image_abspaths]
        query_contents = [{"type": "text", "text": prompt + '\n'}]
        contents = [*image_contents, *query_contents]

        formulated_messages = [
                {"role": "user", "content": contents}
            ]
        step_task.formulated_messages = formulated_messages
        step_task.images = raw_input['step_images']

        return step_task
