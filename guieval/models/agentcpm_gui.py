import re
import logging
from PIL import Image
from typing import Any, Literal, Union, Dict
from vllm import SamplingParams

# subsec internal
from guieval.main import StepTaskModel
from guieval.utils import ActionType
from guieval.models.utils import *
from guieval.models.abcmodel import *
from guieval.models.resources.agentcpm_gui.prompt_builder import build as build_prompt


# section struct
logger = logging.getLogger(__name__)


# section main
@ModelRegistry.register()
class AgentCPM_GUI(ABCModel):
    NAMES = ("agentcpm-gui-8b", )
    MODEL_PATTERNS = ModelPatterns(answer_pattern=r'.*',
                                   answer_flags=[re.DOTALL, ],
                                   thinking_pattern=None,
                                   conclusion_pattern=None)
    DEFAULT_SAMPLING_PARAMS = {
        "agentcpm-gui-8b": SamplingParams(
        max_tokens=2048,
        temperature=0.1,
        top_p=0.3,
        n=1
    )}

    @first_level_parser.validate_patterns(MODEL_PATTERNS)
    def parse_response(self, parsed_matches):
        if parsed_matches['answer'] is not None:
            answer_str: str = parsed_matches['answer'].group(1)
            answer_str = answer_str.strip()

            arguments = ParserTools.parse_json_dict_block(answer_str)
            action = ActionType.action_map(arguments)
            thinking = arguments.get("thought")
            return dict(action=action,
                        arguments=arguments,
                        answer=answer_str,
                        thinking=thinking,
                        conclusion=None)
        else:
            raise parsed_matches['error']

    def model_2_minicpm(self, output_text, width, height):
        try:
            parsed_response: Dict[Literal['action', 'arguments', 'thinking', 'conclusion'],
                                  Union[Dict, Any]] = self.parse_response(output_text)
            parsed_response['action'] = parsed_response['action'].upper()
            return parsed_response
        except Exception as err:
            logger.debug(f"Error. No valid `ModelPatterns` Extraction:\n\t{err}")
            return {'action': None,
                    'arguments': dict()}

    @staticmethod
    def aitw_2_model_action(step_task: StepTaskModel, *args) -> dict[Literal['action', 'arguments'], str | dict]:
        return

    def model_2_contents(self, step_task: StepTaskModel, action, *,
                         online=False) -> tuple[str, CONTENT_SOURCE]:
        # no history content required
        return

    @staticmethod
    def _fetch_image(image_abspath: str) -> Image.Image:
        image = Image.open(image_abspath)
        resolution = image.size
        w, h = resolution
        max_line_res = 1120
        max_line = max_line_res
        if h > max_line_res:
            w = int(w * max_line_res / h)
            h = max_line_res
        if w > max_line:
            h = int(h * max_line_res / w)
            w = max_line_res
        img = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        return img

    def prepare_task_input(self, step_task: StepTaskModel, **kwargs) -> StepTaskModel:
        raw_input = super().prepare_task_input(step_task=step_task)

        formulated_messages = build_prompt(instruction=raw_input['instruction'],
                                           image_count=len(raw_input['step_images']))
        images = [self._fetch_image(_image_abspath)
                  for _image_abspath in step_task.image_abspaths]

        step_task.formulated_messages = formulated_messages
        step_task.images = images

        return step_task
