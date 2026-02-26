from collections import defaultdict
import itertools
import logging
from abc import ABC, abstractmethod
from vllm import SamplingParams
from PIL import Image
from typing import Any, ClassVar, Literal, Sequence, TypeAlias, TypedDict, overload
from typing_extensions import NotRequired

# subsec internal
from guieval.main import StepTaskModel, DeployedModel
from guieval.models.utils import (ModelPatterns, first_level_parser,
                                  ThreadSafeMemory, PARSED_MATCHES)
from guieval.utils.action_space import UNIFIED_ACTION, PREDICTION
from guieval.utils import get_simplified_traceback, qwen_fetch_image
from vllm_serve.deployed import ExtraInferenceSetup


# section struct
logger = logging.getLogger(__name__)


EpisodeID: TypeAlias = int
StepID: TypeAlias = int
CONTENT_SOURCE = Literal['online', 'offline_rule', 'offline_model']


class MINICPM_ACTION(TypedDict):
    action: UNIFIED_ACTION
    arguments: PREDICTION | None


class MODEL_ACTION(TypedDict):
    action: str | None
    arguments: NotRequired[dict]


class HistoryContent(TypedDict):
    content: Any
    source: CONTENT_SOURCE


class RawInput(TypedDict):
    instruction: NotRequired[str]
    step_images: NotRequired[list[Image.Image]]
    history_images: NotRequired[list[Image.Image]]
    fetched_step_image_width: NotRequired[int]
    fetched_step_image_height: NotRequired[int]
    filled_history: NotRequired[list[StepTaskModel]]
    history_actions: NotRequired[list[MODEL_ACTION]]
    history_contents: NotRequired[list[Any]]
    history_content_srcs: NotRequired[list[CONTENT_SOURCE]]


# section model
class ABCModel(ABC):
    NAMES: ClassVar[tuple[str, ...]] = tuple()
    MODEL_PATTERNS: ModelPatterns = None
    DEFAULT_SAMPLING_PARAMS: dict[str, SamplingParams] = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # check if the subclass has overridden NAMES and it's not the default <save>
        if not (hasattr(cls, 'NAMES') and
                cls.NAMES and
                isinstance(cls.NAMES, tuple)):
            raise ValueError(f"{cls.__name__} must override NAMES class variable with "
                             "non empty tuple of model names string")
        if not (hasattr(cls, 'MODEL_PATTERNS') and
                cls.MODEL_PATTERNS and
                isinstance(cls.MODEL_PATTERNS, ModelPatterns)):
            raise ValueError(f"{cls.__name__} must override MODEL_PATTERNS class variable with "
                             "ModelPatterns instance")
        if not (hasattr(cls, 'DEFAULT_SAMPLING_PARAMS') and
                cls.DEFAULT_SAMPLING_PARAMS and
                isinstance(cls.DEFAULT_SAMPLING_PARAMS, dict)):
            logger.warning(f'{cls.__name__} didn\'t override DEFAULT_SAMPLING_PARAMS class variable with'
                           ' `model_name`-`SamplingParams_instance` map.\n'
                           '\tWhen there was no class instance `sampling_params` argument passed to the constructor, '
                           'and simultaneously no manually coded `sampling_params` field value in the class instance,\n'
                           '\tthen if using vLLM offline api to generate task batch, '
                           'the default vllm params will be applied. '
                           'See the `vllm.SamplingParams` field notation for more value setting details.')

    def __init__(self, *,
                 sampling_params: dict[str, SamplingParams] | None = None,
                 **kwargs):
        self._memory: ThreadSafeMemory[EpisodeID, dict[StepID, StepTaskModel]] = ThreadSafeMemory(default_factory=dict)
        self.sampling_params = (getattr(self, 'DEFAULT_SAMPLING_PARAMS', None)
                                if sampling_params is None else
                                sampling_params)
        self._kwargs = kwargs

    @abstractmethod
    @first_level_parser.validate_patterns(MODEL_PATTERNS)
    def parse_response(self, parsed_matches: PARSED_MATCHES) -> Any:
        '''
        If using `first_lelvel_parser` for preprocessing the response,
        the passed value to wrapped `self.parse_response` method would be a dictionary with the following fields:
        - 'answer': re.MatchObject of the MODEL_PATTERNS.answer_pattern | None.
        - 'thinking': re.MatchObject of the MODEL_PATTERNS.thinking_pattern | None.
        - 'conclusion': re.MatchObject of the MODEL_PATTERNS.conclusion_pattern | None.
        - 'error': `ModelPatternExtractionError` instance to claim parsing error for specific response.
        '''
        raise NotImplementedError()

    @abstractmethod
    def model_2_minicpm(self, output_text: str, width: int, height: int) -> MINICPM_ACTION:
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def aitw_2_model_action(step_task: StepTaskModel,
                            resized_height: int, resized_width: int) -> MODEL_ACTION:
        '''
        This tool only map the step_task_gold action to the model action space.
        Mapping result consists of field action and field arguments.
        Nothing more, nothing less.
        '''
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def model_2_contents(step_task: StepTaskModel, action: MODEL_ACTION, *, online: bool = False) -> HistoryContent:
        raise NotImplementedError()

    @abstractmethod
    def prepare_task_input(self, step_task: StepTaskModel,
                           min_pixels: int | None = None,
                           max_pixels: int | None = None,
                           **kwargs) -> RawInput:
        '''
        This original abstractmethod only do some preprocessing for the task input preparation.
        ```python
        return {
            'instruction': instruction,
            'step_images': [qwen_fetch_image(_image_abspath,
                                             min_pixels=min_pixels,
                                             max_pixels=max_pixels)
                            for _image_abspath in step_task.image_abspaths],
            'history_images': filled_history[-1].images | list(),
            'fetched_step_image_width': step_task.images[-1].size[0],
            'fetched_step_image_height': step_task.images[-1].size[1],
            'filled_history': [self._memory[step_task.episode_id][_history_step_id]
                               for _history_step_id in range(step_task.step_id)],
            'history_actions': self.aitw_2_model_action ->
                (filled_history, image_height, image_width): -> MODEL_ACTION,
            'history_contents': self.model_2_contents ->
                (step_task, action, online: bool = False): -> HistoryContent['content'],
            'history_content_srcs': self.model_2_contents ->
                (step_task, action, online: bool = False): -> HistoryContent['source'],
        }
        ```
        The completely implemented method should return `StepTaskModel` instance
        with loaded images, history content sources and formulated messages.
        i.e. `self.prepare_task_input(step_task) -> StepTaskModel`

        Use
        ```python
        step_task.images = loaded_images
        step_task.history_content_srcs = history_content_srcs
        step_task.formulated_messages = formulated_messages
        ```
        to assign the loaded images, history content sources and formulated messages to the `StepTaskModel` instance.
        '''
        instruction = (step_task.low_instruction
                      if step_task.dataset == 'androidcontrol_low' else
                      step_task.instruction)

        images = [qwen_fetch_image(_image_path,
                                   min_pixels=min_pixels,
                                   max_pixels=max_pixels)
                  for _image_path in step_task.image_abspaths]
        sizes = set(_image.size for _image in images)
        if len(sizes) > 1:
            logger.warning('The size of the images for '
                           f'step {step_task.step_id} episode {step_task.episode_id} is not the same.\n'
                           f'Received sizes: {sizes}\n'
                           'Currently take the size of the last image in the history image seq '
                           'as the size of the images for this step. '
                           'This may cause some errors in the model inference.')
        image_width, image_height = images[-1].size

        '''
        start_id for all the episode has been aligned to 0. See data.py ``load_episode`` for more details.
        Thus we can use step_task.step_id as the stop id for its history.
        Meanwhile, for every history step, the history sequence is defined to have the corresponding step task,
        which owes to the error tolerance of `generate_task_batch` function.
        Thus we can use the _history_step_id to index the history sequence.
        '''
        filled_history: list[StepTaskModel] = [self._memory[step_task.episode_id][_history_step_id]
                                               for _history_step_id in range(step_task.step_id)]

        if step_task.mode == 'offline_rule' or step_task.mode == 'semi_online':
            history_actions = [self.aitw_2_model_action(_history_step_task, image_height, image_width)
                               for _history_step_task in filled_history]
            if filled_history:
                _tagged_history_contents = [self.model_2_contents(step_task=_history_step_task,
                                                                  action=_action,
                                                                  online=(step_task.mode == 'semi_online'))
                                            for _history_step_task, _action in zip(filled_history, history_actions)]
                history_contents = [_tagged_content['content'] for _tagged_content in _tagged_history_contents]
                history_content_srcs = [_tagged_content['source'] for _tagged_content in _tagged_history_contents]
                history_images = filled_history[-1].images
            else:
                history_contents, history_content_srcs, history_images = list(), list(), list()
        elif step_task.mode == 'offline_model':
            raise NotImplementedError('Offline history model generation is not supported for GUI-OWL yet.')
        elif step_task.mode == 'semi_online_model':
            raise NotImplementedError('Semi-online history model generation is not supported for GUI-OWL yet.')
        else:
            raise ValueError(f'Invalid mode: {step_task.mode}')

        return {
            'instruction': instruction,
            'step_images': images,
            'history_images': history_images,
            'fetched_step_image_width': image_width,
            'fetched_step_image_height': image_height,
            'filled_history': filled_history,
            'history_actions': history_actions,
            'history_contents': history_contents,
            'history_content_srcs': history_content_srcs
        }

    @overload
    def generate_task(self, task: StepTaskModel, *,
                      sampling_params: SamplingParams) -> str:
        '''
        Generate task response for a single task under vLLM online inference mode.

        Args:
            task: task to be generated.
            sampling_params: sampling parameters for vllm online inference.
        Returns:
            str: generated task response.
        '''
        ...
    @overload  # noqa: E301
    def generate_task(self, task: Sequence[StepTaskModel], *,
                      sampling_params: SamplingParams) -> list[str]:
        '''
        Organize tasks in batch for vLLM offline inference.

        Args:
            task: task batch to be generated.
            sampling_params: sampling parameters for vllm offline inference.
        Returns:
            list[str]: generated task responses.
        '''
        ...
    def generate_task(self, task: StepTaskModel | Sequence[StepTaskModel], *,  # noqa: E301
                      sampling_params: SamplingParams) -> str | list[str] | None | list[None]:
        '''
        chat_template level thinking switch has been integrated into ExtraInferenceSetup
        '''
        if isinstance(task, StepTaskModel) and task.vllm_mode == 'online':
            try:
                extra_params = ExtraInferenceSetup(enable_thinking=task.enable_think)
                return DeployedModel[task.model_alias].wrapped_generate(messages=task.formulated_messages,
                                                                        images=task.images,
                                                                        sampling_params=sampling_params,
                                                                        extra_params=extra_params)
            except Exception as err:
                logger.error(f"vLLM online Inference Failed for:\n\t{repr(err)}\n"
                             f"Traceback:\n{get_simplified_traceback()}")
                return str()
        elif isinstance(task, Sequence) and all((_task.vllm_mode == 'offline') for _task in task):
            try:
                batch_inputs: dict[str, list[dict]] = defaultdict(list)
                for _task in task:
                    text_prompt: str = (DeployedModel[_task.model_alias]
                                        .tokenizer
                                        .apply_chat_template(_task.formulated_messages,
                                                             tokenize=False,
                                                             add_generation_prompt=True,
                                                             enable_thinking=_task.enable_think))
                    _task.sampling_params = repr(sampling_params)
                    batch_inputs[_task.model_alias].append({"prompt": text_prompt,
                                                            "multi_modal_data": {"image": _task.images}})

                return list(itertools.chain
                            .from_iterable(
                                DeployedModel[_model_alias]
                                .wrapped_generate(prompt=_inputs,
                                                  sampling_params=sampling_params)
                                                  for _model_alias, _inputs in batch_inputs.items()))
            except Exception as err:
                logger.error(f"vLLM offline Inference Failed for:\n\t{repr(err)}\n"
                             f"Traceback:\n{get_simplified_traceback()}")
                return [str() for _ in task]
        else:
            vllm_mode = {task.vllm_mode, } if isinstance(task, StepTaskModel) else set(_task.vllm_mode
                                                                                       for _task in task)
            if vllm_mode.difference({'offline', 'online'}):
                raise NotImplementedError('Currently only vllm online and offline inference supported')
            else:
                raise TypeError(f"Invalid vLLM mode and task group combination: ({vllm_mode}, {type(task)}). "
                                "`task` must be a `StepTaskModel` under vLLM online inference mode, "
                                "or a sequence of `StepTaskModel` under vLLM offline inference mode.")

    def run_task(self, task: StepTaskModel | Sequence[StepTaskModel], **kwargs) -> Sequence[StepTaskModel]:
        model = task[0].model if isinstance(task, Sequence) else task.model
        predict_str: str | list[str] = self.generate_task(task,
                                                          sampling_params=self.sampling_params.get(model,
                                                                                                   SamplingParams()))
        if isinstance(task, StepTaskModel):
            task_batch = [task, ]
            predict_str = [predict_str, ]
        elif isinstance(task, Sequence):
            task_batch = task
        else:
            raise TypeError('Some undiscovered logic error')

        for step_task, resp in zip(task_batch, predict_str):
            sizes = set(_image.size for _image in step_task.images)
            if len(sizes) > 1:
                logger.warning('The size of the loaded images for '
                               f'step {step_task.step_id} episode {step_task.episode_id} is not the same.\n'
                               f'Received sizes: {sizes}\n'
                               'Currently take the size of the last image in the history image seq '
                               'as the size of the images for this step. '
                               'This may cause some errors in the postprocessing for evaluation.')
            width, height = step_task.images[-1].size
            step_task.response = resp
            try:
                parsed_res = self.parse_response(resp=resp)
                step_task.answer = parsed_res['answer']
                step_task.thinking = parsed_res['thinking']
                step_task.conclusion = parsed_res['conclusion']
                minicpm = self.model_2_minicpm(resp, width, height)
                step_task.pred_action = minicpm.get('action')
                step_task.prediction = minicpm.get('arguments')
            except Exception as err:
                logger.error(f"Error occurred during processing sample: {repr(err)}.\n"
                             f"{get_simplified_traceback()}")
                step_task.prediction = {}

            step_task.evaluate()  # force evaluation

            self._memory[step_task.episode_id][step_task.step_id] = step_task

        return task_batch


# section opened exports
__all__ = ['ABCModel',
           'EpisodeID', 'StepID',
           'MINICPM_ACTION', 'MODEL_ACTION',
           'HistoryContent', 'CONTENT_SOURCE']
