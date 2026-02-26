import os
import re
import threading
import json
import logging
import jinja2
import warnings
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator, model_serializer
from typing import (Any, ClassVar, Literal)
from typing_extensions import Self, TypedDict

# subsec internal
from guieval.eval import (parse_action,
                          EVALUATORS, EVALUATOR_NAMES)
from guieval.utils import (get_simplified_traceback,
                           ActionType, UNIFIED_ACTION, PREDICTION,
                           is_tap_action)
from guieval.main.utils import MAIN_RESOURCES, CONTENT_SOURCE
from guieval.main.history_sampler import ABCHistorySampler, HistorySamplerConfig
from vllm_serve.utils.message_types import Messages


# section struct
logger = logging.getLogger(__name__)

MODE = Literal['offline_rule', 'offline_model', 'semi_online', 'semi_online_model']
STEP_TASK_REPR_TEMPLATE = (MAIN_RESOURCES / 'step_task_repr.j2')


class EvaluateResult(BaseModel):
    type_match: bool = Field(default=False)
    exact_match: bool = Field(default=False)
    text_dist: float | None = Field(default=float('inf'))
    format_hit: bool | None = Field(default=False)
    pixel_distance: float | None = Field(default=float('inf'))


class EvaluateResultDict(TypedDict):
    type_match: bool
    exact_match: bool
    text_dist: float | None
    format_hit: bool | None
    pixel_distance: float | None


class StepTaskResultSample(TypedDict):
    response: str
    answer: str | None
    thought: str | None
    conclusion: str | None
    action: UNIFIED_ACTION | None
    prediction: PREDICTION
    evaluation: EvaluateResultDict


# section main
class StaticStepTaskModel(BaseModel):
    # task meta
    episode_id: int | str = Field(frozen=True)  # protected
    episode_step_id: int = Field(frozen=True)  # the original step index of the step task in the episode
    step_id: int = Field(frozen=True)  # 0-start aligned step id of the step in the episode sequence
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
    bbox: str | None = Field(frozen=True, default=None)
    low_instruction: str | None = Field(frozen=True, default=None)

    # extra info
    dataset: str = Field(frozen=True)
    split: str = Field(frozen=True)
    subset: str = Field(frozen=True)
    subset_dir: str = Field(frozen=True)  # abspath
    episode_file_name: str = Field(frozen=True)  # relative path

    # class variables
    image_suffices: ClassVar[set[str]] = {"jpeg", "png", "jpg"}

    @computed_field
    @property
    def target_action(self) -> UNIFIED_ACTION | None:
        warnings.warn(('The unified action space would be completely modified. '
                       'By then This method shall be deprecated. No more delay'),
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
            logger.error(f'Unknow action type {self.result_action_type}')
            return None


class RuntimeStepTaskModel(StaticStepTaskModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # task runtime content
    model: str | None = None  # task executor
    model_alias: str | None = None  # task model alias
    episode_abspath: str | None = None  # loaded during model validation after instantiation if not provided
    image_abspaths: list[str] | None = None  # ~
    images: list[Image.Image] | None = None  # ~
    formulated_messages: Messages | None = None
    history_content_srcs: list[CONTENT_SOURCE] = Field(default_factory=list)
    response: str | None = None
    answer: str | None = None
    thought: str | None = None
    conclusion: str | None = None

    # evaluation content
    prediction: PREDICTION | None = None
    pred_action: UNIFIED_ACTION | None = None

    # samples
    result_samples: list[StepTaskResultSample] = Field(default_factory=list)


class StepTaskModel(RuntimeStepTaskModel):
    model_config = ConfigDict(extra="allow")

    # task config content
    enable_think: bool = True
    enable_conclude: bool = True
    fixed_memory: bool = False
    fixed_thought: bool = False
    mode: MODE = 'offline_rule'

    history_sampler: HistorySamplerConfig = Field(default_factory=HistorySamplerConfig)

    # inference engine config
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

    def _reformulate_prediction(self, *,
                                prediction: PREDICTION | None = None) -> dict | None:
        if prediction is None:
            prediction = self.prediction
        try:
            actions, parameters, status = parse_action(prediction)
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
            raise Exception(f"Error processing prediction ${prediction} "
                            f"for step {self.step_id} in episode {self.episode_id}:\n\t{err}")

    def get(self, key: str) -> Any:
        return getattr(self, key)

    def assign_step_task_result(self, result: StepTaskResultSample) -> None:
        self.response = result['response']
        self.answer = result['answer']
        self.thought = result['thought']
        self.conclusion = result['conclusion']
        self.pred_action = result['action']
        self.prediction = result['prediction']

        with self._evaluation_lock:
            self._cached_evaluation = EvaluateResult.model_validate(result["evaluation"])

    def clear_evaluation_cache(self) -> None:
        """
        clear the cached evaluation result to force recomputation.
        """
        with self._evaluation_lock:
            self._cached_evaluation = None

    def evaluate(self, *,
                 prediction: PREDICTION | None = None,
                 use_cache: bool = True) -> EvaluateResult:
        """
        evaluate and cache the evaluation result.
        """
        if prediction is None:
            prediction = self.prediction

        # thread-safe double-checked locking pattern
        if self._cached_evaluation is not None and use_cache:
            return self._cached_evaluation

        with self._evaluation_lock:
            # double-check after acquiring lock
            if self._cached_evaluation is not None and use_cache:
                return self._cached_evaluation

            if not prediction:
                logger.info(f'Evaluating Prediction of Sample step {self.step_id} episode {self.episode_id} is empty. '
                            'Evaluation default to False.')
                result = EvaluateResult()
                if use_cache:
                    self._cached_evaluation = result  # cache the result
                return result

            try:
                evaluator_name: EVALUATOR_NAMES = ('androidcontrol'
                                                   if 'androidcontrol' in self.dataset else
                                                   'common')
                result = EvaluateResult.model_validate(EVALUATORS[evaluator_name](self,
                                                                                  self._reformulate_prediction(
                                                                                      prediction=prediction)))
                if use_cache:
                    self._cached_evaluation = result  # cache the result
                return result
            except Exception as err:
                logger.error(f"ERROR evaluating prediction `{prediction}` for "
                             f"sample {self.episode_id}-{self.step_id}:\n\t{err}\n"
                             f"Traceback:\n\t{get_simplified_traceback()}"
                             f"Default to False. ")
                result = EvaluateResult()
                if use_cache:
                    self._cached_evaluation = result  # cache the error result as well to avoid repeated failures
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
            suffix_pattern = '|'.join(r'_{step}\.({suffix})$'.format(step=self.episode_step_id,
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

        if self.history_sampler.sampler_name is None:
            if self.mode == "semi_online":
                self._history_sampler = ABCHistorySampler(source_choices=("online_pos", "low_instruction"),
                                                          first_choice_prob_lb=1.0)
            elif self.mode == "offline_rule":
                self._history_sampler = ABCHistorySampler(source_choices=("offline_rule", "low_instruction"),
                                                          first_choice_prob_lb=1.0)
            else:
                raise NotImplementedError("todo?")
        else:
            self._history_sampler = (ABCHistorySampler
                                     .get_sampler(name=self.history_sampler.sampler_name)
                                     .model_validate(self.history_sampler.model_dump()))
        return self

    @property
    def history_content_source_choices(self) -> list[CONTENT_SOURCE]:
        return self._history_sampler.content_sources

    def choose_history_sources(self) -> list[CONTENT_SOURCE]:
        return self._history_sampler.sample(history_window=self.step_id)

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
                         result_samples=self.result_samples,
                         response=self.response,
                         answer=self.answer,
                         thought=self.thought,
                         conclusion=self.conclusion)

        # lazy serialization of evaluation result
        if hasattr(self, '_cached_evaluation') and self._cached_evaluation is not None:
            base_dict['evaluation'] = self._cached_evaluation.model_dump()

        return base_dict

    @model_serializer(mode='plain')
    def _serialize_model(self) -> dict:
        base_dict = dict(episode_id=self.episode_id,
                         episode_step_id=self.episode_step_id,
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
                         bbox=self.bbox,
                         low_instruction=self.low_instruction,
                         dataset=self.dataset,
                         split=self.split,
                         subset=self.subset,
                         subset_dir=self.subset_dir,
                         episode_file_name=self.episode_file_name,
                         enable_think=self.enable_think,
                         enable_conclude=self.enable_conclude,
                         fixed_memory=self.fixed_memory,
                         history_sampler=self.history_sampler.model_dump(),
                         model_alias=self.model_alias,
                         mode=self.mode,
                         formulated_messages=self.formulated_messages,
                         history_content_srcs=self.history_content_srcs,
                         result_samples=self.result_samples,
                         prediction=self.prediction,
                         pred_action=self.pred_action,
                         response=self.response,
                         answer=self.answer,
                         thought=self.thought,
                         conclusion=self.conclusion)

        # lazy serialization of evaluation result
        if hasattr(self, '_cached_evaluation') and self._cached_evaluation is not None:
            base_dict['evaluation'] = self._cached_evaluation.model_dump()

        return base_dict

    def hash_encode(self, *,
                    meta: bool = False) -> str:
        '''
        This encoding method only stores the meta and config information of the step task,
        without any runtime content and evaluation result.

        Be as cautious as possible when using this encoding method to identify the step task, since it
        doesn't distinguish the different-runtime step tasks
        and step task results with the same meta and config information.
        '''
        if meta:
            return (f'<dataset>{self.dataset}</dataset>'
                    f'<split>{self.split}</split>'
                    f'<subset>{self.subset}</subset>'
                    f'<episode_id>{self.episode_id}</episode_id>'
                    f'<episode_step_id>{self.episode_step_id}</episode_step_id>'
                    f'<step_id>{self.step_id}</step_id>'
                    f'<instruction>{self.instruction}</instruction>')
        else:
            return (f'<dataset>{self.dataset}</dataset>'
                    f'<split>{self.split}</split>'
                    f'<subset>{self.subset}</subset>'
                    f'<episode_id>{self.episode_id}</episode_id>'
                    f'<episode_step_id>{self.episode_step_id}</episode_step_id>'
                    f'<step_id>{self.step_id}</step_id>'
                    f'<instruction>{self.instruction}</instruction>'
                    f'<model_alias>{self.model_alias}</model_alias>'
                    f'<enable_think>{self.enable_think}</enable_think>'
                    f'<enable_conclude>{self.enable_conclude}</enable_conclude>'
                    f'<mode>{self.mode}</mode>')
