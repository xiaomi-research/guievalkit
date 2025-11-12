import os
import re
import threading
import json
import logging
import jinja2
import warnings

from PIL import Image
from pydantic import (BaseModel, ConfigDict, Field,
                      computed_field, model_validator, model_serializer)
from typing import (Any, ClassVar, Literal)
from typing_extensions import Self

# subsec internal
from guieval.eval import (parse_action,
                          EVALUATORS, EVALUATOR_NAMES)
from guieval.utils import (get_simplified_traceback,
                           ActionType, UNIFIED_ACTION, PREDICTION,
                           is_tap_action)
from guieval.main.utils import MAIN_RESOURCES
from vllm_serve.utils.message_types import Messages


# section struct
logger = logging.getLogger(__name__)

MODE = Literal['offline_rule', 'offline_model', 'semi_online', 'semi_online_model']
STEP_TASK_REPR_TEMPLATE = (MAIN_RESOURCES / 'step_task_repr.j2')


# section main
class EvaluateResult(BaseModel):
    type_match: bool = Field(default=False)
    exact_match: bool = Field(default=False)
    text_dist: float | None = Field(default=float('inf'))
    format_hit: bool | None = Field(default=False)
    pixel_distance: float | None = Field(default=float('inf'))


class StaticStepTaskModel(BaseModel):
    # task meta <save>
    episode_id: int | str = Field(frozen=True)  # protected <save>
    episode_start_index: int = Field(frozen=True)  # the original start step index of the episode <save>
    step_id: int = Field(frozen=True)  # 0-start aligned step id of the step in the episode sequence <save>
    episode_length: int = Field(frozen=True)
    image_width: int = Field(frozen=True)
    image_height: int = Field(frozen=True)
    image_path: str | list[str] = Field(frozen=True)
    instruction: str = Field(frozen=True)
    result_action_type: int = Field(frozen=True, default=None)
    result_touch_yx: str = Field(frozen=True, default=None)
    result_lift_yx: str = Field(frozen=True, default=None)
    duration: int | None = Field(frozen=True, default=None)
    result_action_text: str | None = Field(frozen=True, default=None)
    result_action_app_name: str | None = Field(frozen=True, default=None)
    ui_positions: str | None = Field(frozen=True, default=None)
    low_instruction: str | None = Field(frozen=True, default=None)

    # extra info <save>
    dataset: str = Field(frozen=True)
    split: str = Field(frozen=True)
    subset: str = Field(frozen=True)
    subset_dir: str = Field(frozen=True)  # abspath <save>
    episode_file_name: str = Field(frozen=True)  # relative path <save>

    # class variables
    image_suffices: ClassVar[set[str]] = {"jpeg", "png", "jpg"}

    @computed_field
    @property
    def target_action(self) -> UNIFIED_ACTION | None:
        warnings.warn(('The unified action space might be completely modified.'
                       'By then This method shall be deprecated. '
                       'Delay no more.'),
                      DeprecationWarning)

        if self.result_action_type == ActionType.TYPE:
            return "TYPE"

        elif self.result_action_type == ActionType.DUAL_POINT:
            normalized_start_yx = json.loads(self.result_touch_yx)
            normalized_end_yx = json.loads(self.result_lift_yx)
            if is_tap_action(normalized_start_yx, normalized_end_yx):
                return "CLICK"
            else:
                return "SCROLL"

        elif self.result_action_type == ActionType.LONG_POINT:
            return "LONG_POINT"

        elif self.result_action_type in [ActionType.PRESS_BACK,
                                         ActionType.PRESS_HOME,
                                         ActionType.PRESS_ENTER]:
            return "PRESS"

        elif (self.result_action_type == ActionType.STATUS_TASK_COMPLETE or
              self.result_action_type == ActionType.STATUS_TASK_IMPOSSIBLE):
            return "STOP"

        elif self.result_action_type == ActionType.NO_ACTION:
            return "WAIT"

        elif self.result_action_type == ActionType.OPEN_APP:
            return "OPEN"

        else:
            raise ValueError("Unknown action type.")


class RuntimeStepTaskModel(StaticStepTaskModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # task runtime content <save>
    model: str | None = None  # task executor <save>
    model_alias: str | None = None  # task model alias <save>
    episode_abspath: str | None = None  # loaded during model validation after instantiation if not provided <save>
    image_abspaths: list[str] | None = None  # ~ <save>
    images: list[Image.Image] | None = None  # ~ <save>
    formulated_messages: Messages | None = None
    history_content_srcs: list[Literal['online', 'offline_rule', 'offline_model']] = Field(default_factory=list)
    response: str | None = None
    answer: str | None = None
    thinking: str | None = None
    conclusion: str | None = None

    # evaluation content <save>
    prediction: PREDICTION | None = None
    pred_action: UNIFIED_ACTION | None = None


class StepTaskModel(RuntimeStepTaskModel):
    model_config = ConfigDict(extra="allow")

    # task config content <save>
    enable_think: bool = True
    enable_conclude: bool = True
    mode: MODE = 'offline_rule'

    # inference engine config <save>
    vllm_mode: Literal['online', 'offline', False] = 'online'

    _repr_template: ClassVar[jinja2.Template] = jinja2.Template(STEP_TASK_REPR_TEMPLATE.read_text(), trim_blocks=True)

    def __getstate__(self) -> dict:
        base: dict[Literal['__dict__'] | str, dict | Any] = super().__getstate__()
        base['__dict__']['_evaluation_lock'] = None
        return base

    def __setstate__(self, state: dict[Literal['__dict__'] | str, dict | Any]) -> None:
        state['__dict__']['_evaluation_lock'] = threading.RLock()
        return super().__setstate__(state)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __repr__(self) -> str:
        name = type(self).__name__
        evaluation_name = EvaluateResult.__name__
        type_match = (None
                      if self._cached_evaluation is None else
                      self._cached_evaluation.type_match)
        exact_match = (None
                       if self._cached_evaluation is None else
                       self._cached_evaluation.exact_match)
        evaluation = f"{evaluation_name}({type_match=}, {exact_match=}, ...)"
        return self._repr_template.render(name=name, step_task=self, evaluation=evaluation)

    def _reformulate_prediction(self) -> dict | None:
        try:
            actions, parameters, status = parse_action(self.prediction)
            transformed_entry = {
                "action_predict": {
                    "COA": {
                        "txt": {
                            "ACTION": actions,
                            "ARGS": parameters,
                            "STATUS": status
                        }
                    }
                }
            }
            return transformed_entry
        except Exception as err:
            raise Exception(f"Error processing step {self.step_id} in episode {self.episode_id}:\n\t{err}")

    @computed_field
    @property
    def evaluation(self) -> EvaluateResult:
        return self.evaluate()

    def get(self, key: str) -> Any:
        return getattr(self, key)

    def clear_evaluation_cache(self) -> None:
        """
        clear the cached evaluation result to force recomputation.
        """
        with self._evaluation_lock:
            self._cached_evaluation = None

    def evaluate(self) -> EvaluateResult:
        """
        evaluate and cache the evaluation result.
        """
        # thread-safe double-checked locking pattern <save>
        if self._cached_evaluation is not None:
            return self._cached_evaluation

        with self._evaluation_lock:
            # double-check after acquiring lock <save>
            if self._cached_evaluation is not None:
                return self._cached_evaluation

            if not self.prediction:
                logger.info(f'Sample {self.episode_id}-{self.step_id} Prediction is empty. '
                            'Evaluation default to False.')
                result = EvaluateResult()
                self._cached_evaluation = result  # cache the result <save>
                return result

            try:
                evaluator_name: EVALUATOR_NAMES = ('androidcontrol'
                                                   if 'androidcontrol' in self.dataset else
                                                   'common')
                result = EvaluateResult.model_validate(EVALUATORS[evaluator_name](self, self._reformulate_prediction()))
                self._cached_evaluation = result  # cache the result <save>
                return result
            except Exception as err:
                logger.error(f"ERROR evaluating sample {self.episode_id}-{self.step_id}:\n\t{err}\n"
                      f"Traceback:\n\t{get_simplified_traceback()}"
                      f"Default to False. ")
                result = EvaluateResult()
                self._cached_evaluation = result  # cache the error result as well to avoid repeated failures <save>
                return result

    def _locate_image_abspath(self, image_path: str) -> str:
        '''
        dependent on field `episode_abspath` loaded during model validation after instantiation if not provided,
        do not call this method directly.
        '''
        if os.path.exists(image_path):
            image_abspath = (image_path
                          if os.path.isabs(image_path) else
                          os.path.abspath(image_path))
        else:
            suffix_pattern = '|'.join(r'_{step}\.({suffix})$'.format(step=(self.step_id + self.episode_start_index),
                                                                     suffix=suffix)
                                      for suffix in self.image_suffices)
            for _file in os.listdir(self.episode_abspath):
                if re.search(suffix_pattern, _file):
                    break
            else:
                raise FileNotFoundError(f'No image found for step {self.step_id} of '
                                        f'episode {self.episode_id} in dataset {self.dataset}')
            image_abspath = os.path.join(self.episode_abspath, _file)
        if not os.path.exists(image_abspath):
            raise FileNotFoundError(f'Located image does not exist: {image_abspath}')
        return image_abspath

    @model_validator(mode='after')
    def _locate_abspaths(self) -> Self:
        if self.episode_abspath is None:
            self.episode_abspath = os.path.join(self.subset_dir, self.episode_file_name)

        if self.image_abspaths is None:
            self.image_abspaths = ([self._locate_image_abspath(self.image_path), ]
                                   if isinstance(self.image_path, str) else
                                   [self._locate_image_abspath(image_path) for image_path in self.image_path])

        return self

    def model_post_init(self, context):
        '''
        initialize the thread-safe evaluation cache.
        '''
        self._cached_evaluation = None
        self._evaluation_lock = threading.RLock()
        return self

    def concise_dump(self) -> dict:
        base_dict = dict(episode_id=self.episode_id,
                         step_id=self.step_id,
                         episode_length=self.episode_length,
                         instruction=self.instruction,
                         low_instruction=self.low_instruction,
                         target_action=self.target_action,
                         dataset=self.dataset,
                         image_abspaths=self.image_abspaths,
                         formulated_messages=self.formulated_messages,
                         history_content_srcs=self.history_content_srcs,
                         response=self.response,
                         answer=self.answer,
                         thinking=self.thinking,
                         conclusion=self.conclusion)

        # lazy serialization of evaluation result
        if hasattr(self, '_cached_evaluation') and self._cached_evaluation is not None:
            base_dict['evaluation'] = self._cached_evaluation.model_dump()

        return base_dict

    @model_serializer(mode='plain')
    def _serialize_model(self) -> dict:
        base_dict = dict(episode_id=self.episode_id,
                         step_id=self.step_id,
                         episode_length=self.episode_length,
                         image_width=self.image_width,
                         image_height=self.image_height,
                         image_path=self.image_path,
                         instruction=self.instruction,
                         result_action_type=self.result_action_type,
                         target_action=self.target_action,
                         result_touch_yx=self.result_touch_yx,
                         result_lift_yx=self.result_lift_yx,
                         duration=self.duration,
                         result_action_text=self.result_action_text,
                         result_action_app_name=self.result_action_app_name,
                         ui_positions=self.ui_positions,
                         low_instruction=self.low_instruction,
                         dataset=self.dataset,
                         split=self.split,
                         subset=self.subset,
                         subset_dir=self.subset_dir,
                         episode_file_name=self.episode_file_name,
                         enable_think=self.enable_think,
                         enable_conclude=self.enable_conclude,
                         mode=self.mode,
                         formulated_messages=self.formulated_messages,
                         history_content_srcs=self.history_content_srcs,
                         prediction=self.prediction,
                         pred_action=self.pred_action,
                         response=self.response,
                         answer=self.answer,
                         thinking=self.thinking,
                         conclusion=self.conclusion)

        # lazy serialization of evaluation result
        if hasattr(self, '_cached_evaluation') and self._cached_evaluation is not None:
            base_dict['evaluation'] = self._cached_evaluation.model_dump()

        return base_dict
