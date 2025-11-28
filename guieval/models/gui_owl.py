# section import
import itertools
import json
import math
import numpy as np
import re
import logging

from string import Template
from typing import Any, Literal, Union, Optional, Dict
from qwen_agent.llm.fncall_prompts.nous_fncall_prompt import NousFnCallPrompt, Message, ContentItem
from qwen_agent.tools.base import register_tool
from vllm import SamplingParams

# subsec internal
from guieval.main import StepTaskModel
from guieval.utils import ActionType
from guieval.models.utils import *
from guieval.models.utils.abcmodel import *
from guieval.utils.action_utils import is_tap_action, get_direction
from guieval.models.utils.qwen2p5_vl_agent_function_call import MobileUse

# section struct
logger = logging.getLogger(__name__)

MAX_PIXELS, MIN_PIXELS = 10035200, 3136
_SAMPLING_PARAMS = SamplingParams(
    max_tokens=2048,
    temperature=0.1,
    top_p=0.001,
    top_k=1,
    repetition_penalty=1.05,
    n=1,
    stop_token_ids=[])


class KeyMismatchError(Exception):
    """Raised when block keys don't match the expected keys."""


@register_tool("mobile_for_gui_owl")
class MobileUse_GUIOwl(MobileUse):
    name: str = "mobile_for_gui_owl"

    def __init__(self, cfg=None):
        self.display_width_px = cfg["display_width_px"]
        self.display_height_px = cfg["display_height_px"]
        super().__init__(cfg)

    def call(self, params, **kwargs):
        return 'function_holder'


def build_system_messages(resized_width, resized_height):
    mobile_use = MobileUse_GUIOwl(
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


# section main
@ModelRegistry.register()
class GUIOwl(ABCModel):
    NAMES = ("gui-owl-7b", "gui-owl-32b")
    MODEL_PATTERNS = ModelPatterns(answer_pattern=r'<tool_call>(.*)</tool_call>',
                                   answer_flags=[re.DOTALL, ],
                                   thinking_pattern=r'<thinking>(.*)</thinking>',
                                   thinking_flags=[re.DOTALL, ],
                                   conclusion_pattern=r'<conclusion>(.*)</conclusion>',
                                   conclusion_flags=[re.DOTALL, ])
    DEFAULT_SAMPLING_PARAMS = dict.fromkeys(NAMES, _SAMPLING_PARAMS)

    @first_level_parser.validate_patterns(MODEL_PATTERNS)
    def parse_response(self, parsed_matches: PARSED_MATCHES):
        try:
            answer_str: str = parsed_matches['answer'].group(1)
            answer_str = answer_str.strip()

            try:
                answer_json_block = re.search(r'^[^{}]*({.*})[^{}]*$', answer_str, re.DOTALL).group(1)
                try:
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
                answer = dict(name="mobile_for_gui_owl", arguments=answer)
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
            logger.error(f"Error. No valid `ModelPatterns` Extraction: {err}")
            return {'action': None,
                    'arguments': dict()}
        if parsed_response['action'] == "click":
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
            # Action parsing still reserved for symmetry.

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
            logger.error(f"Error, unrecognized action: {parsed_response['action']}")
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
            # time = random.randint(3, 10)
            time = 1000
            return {'action': 'long_press',
                    'arguments': dict(coordinate=coordinate, time=time)}
        elif action_type == ActionType.NO_ACTION:
            # time = random.randint(3, 10)
            time = 1000
            return {'action': 'wait',
                    'arguments': dict(time=time)}
        elif action_type == ActionType.OPEN_APP:
            return {'action': 'open',
                    'arguments': dict(text=step_task.result_action_app_name)}
        else:
            logger.info(f'Task {step_task.episode_id} step {step_task.step_id} '
                        f'Action type `{ActionType(action_type).name}` not supported for GUI-OWL.')
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def model_2_contents(step_task, action, *, online: bool = False) -> HistoryContent:
        if online and step_task.evaluation.exact_match:
            return {'content': step_task.conclusion,  # history content required by gui owl is the action conclusion.
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
            return {'content': f'Type \'{text}\'' if text else 'Type \'\'',
                    'source': 'offline_rule'}
        elif action['action'] == 'terminate':
            status = action['arguments'].get('status')
            return {'content': f'Terminate with {status}' if status else 'Terminate with Unknown status',
                    'source': 'offline_rule'}
        elif action['action'] == 'long_press':
            coordinate = action['arguments'].get('coordinate')
            time = action['arguments'].get('time')
            return {'content': (f'Long press point {coordinate} for {time} seconds'
                                if coordinate and time else
                                'Long press Unknown coordinate for Unknown time'),
                    'source': 'offline_rule'}
        elif action['action'] == 'wait':
            time = action['arguments'].get('time')
            return {'content': f'Wait for {time} seconds' if time else 'Wait',
                    'source': 'offline_rule'}
        elif action['action'] == 'open':
            app = action['arguments'].get('text')
            return {'content': f'Open App {app}' if app else 'Open Unknown app',
                    'source': 'offline_rule'}
        else:
            logger.error(f'Action {action["action"]} not supported for GUI-OWL.')
            return {'content': 'Unknown action',
                    'source': 'offline_rule'}

    @staticmethod
    def _history_action_2_message(step_task: StepTaskModel, action: MODEL_ACTION, *,
                                  enable_think: bool = False, online: bool = False):
        template = '<tool_call>\n${raw_answer}\n</tool_call>'
        template = (Template(template)
                    if not enable_think else
                    Template('\n'.join(['<thinking>\n${thinking}\n\n</thinking>', template,
                                        '<conclusion>\n${conclusion}\n</conclusion>'])))
        if online and step_task.evaluation.exact_match:
            response = template.safe_substitute(thinking=step_task.thinking,
                                                conclusion=step_task.conclusion,
                                                raw_answer=step_task.answer)
        else:
            raw_answer = json.dumps(dict(name="mobile_for_gui_owl",
                                         arguments=dict(action=action['action'],
                                                        **action['arguments'])), ensure_ascii=False)
            response = template.safe_substitute(thinking='null',
                                                conclusion='null',
                                                raw_answer=raw_answer)

        return {"role": "assistant", "content": [{"type": "text",
                                                  "text": response}]}

    @staticmethod
    def _formulate_history_user_message(step_task: StepTaskModel, memorize_image: bool):
        step_id = step_task.step_id
        user_message_contents = list()
        if step_id == 0:
            user_message_contents.append({'type': 'text', 'text': f'''The user query: \n{step_task.instruction}'''})
        if memorize_image:
            user_message_contents.append({"type": "image", "image": step_task.image_abspaths[0],
                                          "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS})
        return {"role": "user", "content": user_message_contents} if user_message_contents else None

    def _formulate_history_messages(self, step_task: StepTaskModel, action: dict, memorize_image: bool, *,
                                    enable_think: bool = False, online: bool = False):
        user_message = self._formulate_history_user_message(step_task=step_task,
                                                            memorize_image=memorize_image)
        assistant_response_message = self._history_action_2_message(step_task=step_task, action=action,
                                                                    enable_think=enable_think, online=online)
        return [assistant_response_message] if user_message is None else [user_message, assistant_response_message]

    def prepare_task_input(self, step_task: StepTaskModel, *,
                           image_memory: int = 2) -> StepTaskModel:
        raw_input = super().prepare_task_input(step_task=step_task,
                                               min_pixels=MIN_PIXELS,
                                               max_pixels=MAX_PIXELS)

        history_length = len(raw_input['filled_history'])
        if history_length >= 1 and step_task.dataset == 'androidcontrol_high':
            history_image_memory = (1 - image_memory)
            history_messages = [self._formulate_history_messages(step_task=_history_step_task, action=_action,
                                                                 memorize_image=(
                                                                     (i - history_length) >= history_image_memory),
                                                                 enable_think=step_task.enable_think,
                                                                 online=(step_task.mode == 'semi_online'))
                                for i, (_history_step_task, _action) in enumerate(zip(raw_input['filled_history'],
                                                                                      raw_input['history_actions']))]
            history_messages = list(itertools.chain.from_iterable(history_messages))
            history_images = raw_input['filled_history'][-1].images
        else:
            history_messages = list()
            history_images = list()

        system_message = build_system_messages(resized_height=raw_input['fetched_step_image_height'],
                                               resized_width=raw_input['fetched_step_image_width'])
        user_messages = build_user_messages(step_task.step_id, raw_input['instruction'],
                                            step_task.image_abspaths,
                                            MIN_PIXELS, MAX_PIXELS,
                                            enable_think=step_task.enable_think,
                                            history=raw_input['history_contents'],
                                            dataset=step_task.dataset)

        formulated_messages = [
            system_message,
            *history_messages,
            *user_messages
        ]
        images = [*history_images, *raw_input['step_images']][-image_memory:]

        step_task.formulated_messages = formulated_messages
        step_task.history_content_srcs = raw_input['history_content_srcs']
        step_task.images = images

        return step_task
