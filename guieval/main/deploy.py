import threading
import logging
from typing import Sequence, overload
from typing_extensions import Self
from vllm import LLM, SamplingParams, RequestOutput, PromptType
from PIL import Image

# subsec internal
from config import model_config_handler
from vllm_serve import DeployModel, ExtraInferenceSetup
from vllm_serve.utils import Messages, process_vision_content
from guieval.config import EvalTaskConfig


logger = logging.getLogger(__name__)


class DeployedModel:
    '''
    A registry of deployed models.

    Use method `init_worker` to initialize and register a `vllm.LLM`worker for a model.

    For accessing the worker, the following methods are provided:
    - `DeployedModel[alias]`: get a worker by alias
    - `first_worker`: get the first worker in the registry values.
        According to the behaviour of dict in python 3.10, it is also the temporarily first worker.
    - `deprecate_worker`: deprecate a worker by alias
    - `get_worker`: get a worker by alias
    '''
    workers: dict[str, Self] = dict()
    _worker_lock = threading.Lock()

    @classmethod
    def first_worker(cls) -> LLM | None:
        try:
            return list(cls.workers.values())[0]
        except Exception:
            return None

    @classmethod
    def deprecate_worker(cls, alias: str):
        if alias in cls.workers:
            del cls.workers[alias]
        else:
            raise ValueError(f"Worker {alias} not found")

    @classmethod
    def get_worker(cls, alias: str) -> Self | None:
        return cls.workers.get(alias, None)

    def __class_getitem__(cls, alias: str) -> 'DeployedModel':
        return cls.workers[alias]

    @classmethod
    def init_worker(cls, setup: EvalTaskConfig) -> Self:
        '''
        Initialize a worker for a model.
        If alias is not provided, it will be set to the model name.

        For first time initialization, the worker will be initialized, registered with the alias and returned.

        If the alias is already in use, the existing worker will be returned without reinitialization.

        If the model file does not exist, a FileNotFoundError will be raised.

        If the model is not found in the model_paths.json file, a ValueError will be raised.

        Args:
            model: The model name. It will be used to locate the model file.
            tp: The tensor parallel size.
            memory: The GPU memory utilization.
            alias: The alias for the model will be used as the key in the workers dictionary.

        Returns:
            The worker for the model.
        '''
        if setup.model.model_alias in cls.workers:
            return cls.workers[setup.model.model_alias]
        with cls._worker_lock:
            if setup.model.model_alias in cls.workers:
                return cls.workers[setup.model.model_alias]
            cls.workers[setup.model.model_alias] = cls(setup, _prevent_direct_init=False)
            return cls.workers[setup.model.model_alias]

    def __init__(self, setup: EvalTaskConfig,
                 _prevent_direct_init: bool = True):
        if _prevent_direct_init:
            raise RuntimeError("Direct initialization is not allowed. "
                               "Please use init_worker instead")

        self._setup = setup
        self._vllm_mode = setup.vllm_mode

        if self._vllm_mode == 'online':
            self._worker = DeployModel(setup=self._setup.model)
            self._worker.deploy()

        elif self._vllm_mode == 'offline':
            logger.info(f'Starting LLM: {self._setup.model.model}')
            self._model_config = model_config_handler(model_name=self._setup.model.model_name)
            self._tokenizer = (self._model_config
                               .tokenizer_class
                               .from_pretrained(self._setup.model.model,
                                                trust_remote_code=True))

            self._worker = LLM(model=self._setup.model.model,
                               tensor_parallel_size=self._setup.model.tensor_parallel_size,
                               gpu_memory_utilization=self._setup.model.gpu_memory_utilization,
                               limit_mm_per_prompt={"image": self._setup.model.image_limit},
                               trust_remote_code=True)
        else:
            raise NotImplementedError("Only VLLM is supported for now")

    @property
    def model_name(self) -> str:
        return self._setup.model.model_name

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def tensor_parallel_size(self) -> int:
        return self._setup.model.tensor_parallel_size

    @property
    def gpu_memory_utilization(self) -> float:
        return self._setup.model.gpu_memory_utilization

    @property
    def worker(self) -> LLM:
        return self._worker

    @overload
    def wrapped_generate(self, *,
                         messages: Messages | None = None,
                         images: list[Image.Image] | None = None,
                         sampling_params: SamplingParams | None = None,
                         extra_params: ExtraInferenceSetup | None = None) -> str:
        ...
    @overload  # noqa: E301
    def wrapped_generate(self, *,
                         prompt: PromptType | Sequence[PromptType] | None = None,
                         sampling_params: SamplingParams | None = None) -> RequestOutput | list[RequestOutput]:
        ...
    def wrapped_generate(self, *,  # noqa: E301
                         messages: Messages | None = None,
                         images: list[Image.Image] | None = None,
                         prompt: PromptType | Sequence[PromptType] = None,
                         sampling_params: SamplingParams | None = None,
                         extra_params: ExtraInferenceSetup | None = None) -> str | RequestOutput | list[RequestOutput]:
        # Determine which signature is being used based on non-None arguments
        has_messages = (messages is not None)
        has_prompt = (prompt is not None)

        if has_messages and has_prompt:
            raise ValueError("Cannot provide both messages and prompt")
        elif not (has_messages or has_prompt):
            raise ValueError("Must provide either messages or prompt")

        if has_messages:
            # Online mode
            if self._vllm_mode != 'online':
                raise ValueError("Messages provided but not in online mode")
            if sampling_params is None or extra_params is None:
                raise ValueError("sampling_params and extra_params required for online mode")
            return self._worker(messages=process_vision_content(messages, images),
                                sampling_params=sampling_params,
                                extra_params=extra_params)

        else:  # has_prompt
            # Offline mode
            if self._vllm_mode != 'offline':
                raise ValueError("Prompt provided but not in offline mode")
            if sampling_params is None:
                raise ValueError("sampling_params required for offline mode")
            outputs = self._worker.generate(prompt, sampling_params=sampling_params, use_tqdm=False)

            return [output.outputs[0].text for output in outputs]
