import io
import logging
from typing import Sequence, Type
from typing_extensions import Self
from vllm import SamplingParams

# subsec internal
from guieval.main.step_task import StepTaskModel
from guieval.models.abcmodel import *

# section register
from guieval.models import *


logger = logging.getLogger(__name__)


# section main
class ModelProcessor(ModelRegistry):
    @classmethod
    def init(cls, model_name: str, *,
             sampling_params: SamplingParams | None = None,
             sample_size: int | None = None,
             sample_seed: int | None = None,
             temperature: float | None = None,
             top_p: float | None = None,
             top_k: int | None = None,
             repetition_penalty: float | None = None,
             presence_penalty: float | None = None,
             max_tokens: int | None = None) -> Self:
        try:
            _processor: Type[ABCModel] = cls[model_name]
        except KeyError as err:
            logger.error(f'No processor core registered for model_name `{model_name}`.\n'
                         f'Current registry:\n'
                         f'{cls.dump_registry()}\n'
                         'If you want to append model for one existing core, '
                         'add the corresponding model name to the `guieval.models.the_model.Core.NAMES` '
                         'and `guieval.models.the_model.Core.DEFAULT_SAMPLING_PARAMS` class attributes.\n'
                         '<split_line>')
            raise err
        return cls(_processor=_processor(sampling_params=sampling_params,
                                         sample_size=sample_size,
                                         sample_seed=sample_seed,
                                         temperature=temperature,
                                         top_p=top_p,
                                         top_k=top_k,
                                         repetition_penalty=repetition_penalty,
                                         presence_penalty=presence_penalty,
                                         max_tokens=max_tokens),
                    _explicit_init_lock=False)

    def __init__(self, *,
                 _processor: ABCModel,
                 _explicit_init_lock: bool = True):
        if _explicit_init_lock:
            raise ValueError('ModelProcessor must be initialized explicitly with `init` method')
        self._processor = _processor

    def prepare_task_input(self,
                           task: StepTaskModel,
                           **kwargs) -> Sequence[StepTaskModel]:
        return self._processor.prepare_task_input(task, **kwargs)

    def run_task(self,
                 task: StepTaskModel | Sequence[StepTaskModel], **kwargs) -> Sequence[StepTaskModel]:
        return self._processor.run_task(task, **kwargs)

    def fill_memory(self, task: StepTaskModel, **kwargs):
        with self._processor.suppress_logger():
            self._processor.prepare_task_input(step_task=task, **kwargs)
        self._processor._memory[task.episode_id][task.step_id] = task

    def dump_memory(self, io: dict[str, io.TextIOWrapper]):
        for _step_tasks in self._processor._memory.values():
            _step_tasks: dict[int, StepTaskModel]
            for _step_task in _step_tasks.values():
                io[_step_task.dataset].write(_step_task.model_dump_json() + '\n')

    def fill_fixed_thoughts(self, task: StepTaskModel):
        self._processor._fixed_thoughts[task.episode_id][task.step_id] = task

    def clear_memory(self):
        """Clear all memory and fixed thoughts from the processor."""
        self._processor._memory.clear()
        self._processor._fixed_thoughts.clear()

    def clear_episode_memory(self, episode_id: int | str):
        """Clear memory associated with a specific episode."""
        if episode_id in self._processor._memory:
            del self._processor._memory[episode_id]
        if episode_id in self._processor._fixed_thoughts:
            del self._processor._fixed_thoughts[episode_id]


__all__ = ['ModelProcessor', ]
