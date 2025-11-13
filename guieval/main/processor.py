from typing import Sequence, Type
from typing_extensions import Self
from vllm import SamplingParams

# subsec internal
from guieval.main import StepTaskModel
from guieval.models.utils.abcmodel import ABCModel


# section register
from guieval.models import *


# section main
class ModelProcessor(ModelRegistry):
    @classmethod
    def init(cls, model_name: str,
             sampling_params: SamplingParams | None = None) -> Self:
        _processor: Type[ABCModel] = cls[model_name]
        return cls(_processor=_processor(sampling_params=sampling_params),
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


__all__ = ['ModelProcessor', ]
