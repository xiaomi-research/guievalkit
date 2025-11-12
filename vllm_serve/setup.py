import os
import jinja2
from pydantic import BaseModel, computed_field, Field, model_validator
from typing import Any, Literal
from typing_extensions import Self

# subsec internal
from config import MODEL_PATHS, MODEL_PATH_FILE
from vllm_serve.utils import generate_timestamp, VLLM_SERVE_UTILS


# section main
SERVE_COMMAND_TEMPLATE: jinja2.Template = jinja2.Template((VLLM_SERVE_UTILS / 'online_serving.j2').read_text())


def render_serve_command(setup: 'ModelSetup') -> str:
    return SERVE_COMMAND_TEMPLATE.render(setup=setup)


class ModelSetup(BaseModel):
    model: str | None = Field(
        description='File path to the model(e.g., /models/llama-7b)'
    )  # auto-search model from MODEL_PATHS if not provided
    model_name: str = Field(
        frozen=True,
        description='The base model name'
    )
    model_alias: str | None = Field(
        frozen=True,
        description='Human-readable model identifier used in logs and UI',
        default=None
    )  # used in logs and UI

    tokenizer: str | None = Field(
        frozen=True,
        description='File path to the tokenizer(e.g., /models/tokenizer or huggingface/tokenizer)',
        default=None
    )
    max_model_len: int = Field(
        frozen=True,
        description='Maximum window size of the model(e.g., 1024)',
        default=8192
    )

    host: str = Field(
        frozen=True,
        description='Host address of the server(e.g., 0.0.0.0)',
        default="127.0.0.1"
    )
    port: int = Field(
        frozen=True,
        description='Port number of the server(e.g., 8000)',
        default="17721"
    )
    api_key: str = Field(
        frozen=True,
        description='API key for the server(e.g., 1234567890)',
        default='abc',
    )
    max_retries: int = Field(
        frozen=True,
        description='Retry limit for online inference api',
        default=1
    )

    dtype: Literal["auto", "half", "float16", "bfloat16", "float", "float32"] = Field(
        frozen=True,
        description='Data type of the model(e.g., float16)',
        default="auto"
    )
    task: Literal["auto", "generate", "embedding", "embed",
                  "classify", "score", "reward", "transcription"] = Field(
        frozen=True,
        description='Task of the model(e.g., generate)',
        default="generate"
    )

    tensor_parallel_size: int = Field(
        frozen=True,
        description='Number of tensor parallel GPUs(e.g., 1)',
        default=1
    )
    data_parallel_size: int = Field(
        frozen=True,
        description='Number of data parallel GPUs(e.g., 1)',
        default=1
    )
    pipeline_parallel_size: int = Field(
        frozen=True,
        description='Number of pipeline parallel GPUs(e.g., 1)',
        default=1
    )
    gpu_memory_utilization: float = Field(
        frozen=True,
        description='GPU memory utilization(e.g., 0.5)',
        default=0.9
    )
    max_num_batched_tokens: int = Field(
        frozen=True,
        description='Max batch size of processing tokens during one inference period',
        default=4096
    )
    max_num_seqs: int = Field(
        frozen=True,
        description='Max seq count during one inference period',
        default=32
    )

    image_limit: int = Field(
        frozen=True,
        description='Maximum number of images per prompt(e.g., 1)',
        default=3
    )
    video_limit: int = Field(
        frozen=True,
        description='Maximum number of videos per prompt(e.g., 1)',
        default=0
    )

    enable_log_stats: bool = Field(
        frozen=True,
        description='Enable logging of statistics(e.g., True)',
        default=True
    )
    enable_log_requests: bool = Field(
        frozen=True,
        description='Enable logging of requests(e.g., True)',
        default=True
    )

    log_base_dir: str = Field(
        frozen=True,
        description='Base directory for the log file(e.g., ./logs)',
        default="./logs/vllm_serve"
    )

    timestamp: str = Field(
        frozen=True,
        description="",
        default_factory=generate_timestamp
    )

    @staticmethod
    def _is_empty(value: str | Any):
        if not value or (isinstance(value, str) and
                         value.strip().lower() in {'', 'none', 'null', 'empty', 'blank', 'default', 'auto'}):
            return True
        else:
            return False

    @model_validator(mode='before')
    @classmethod
    def _fill_model_alias(cls, data: dict) -> dict:
        # If model_alias is not provided or is None/empty, default it to model_name
        if data is None:
            return data

        model_alias = data.get('model_alias')
        if cls._is_empty(value=model_alias):
            model_name = data.get('model_name')
            if model_name is not None:
                data['model_alias'] = model_name
        return data

    @model_validator(mode='after')
    def _validate_path(self) -> Self:
        try:
            if self._is_empty(value=self.model):
                self.model = MODEL_PATHS[self.model_name]['folder_name']
            if not os.path.exists(self.model):
                raise FileNotFoundError(f"Model {self.model_name} file {self.model} not found")
        except KeyError as err:
            raise ValueError(f'Model {self.model_name} not found in {MODEL_PATH_FILE} and '
                             'no valid specified `model` value for model path:\n'
                             f'\t{repr(err)}')

        return self

    @computed_field
    @property
    def log_dir(self) -> str:
        return os.path.join(self.log_base_dir, self.model_alias)

    @computed_field
    @property
    def log_path(self) -> str:
        return os.path.join(self.log_dir, f"{self.timestamp}.log")

    @computed_field
    @property
    def deployment_log_path(self) -> str:
        return os.path.join(self.log_dir, f"{self.timestamp}_deployment.log")

    @computed_field
    @property
    def serve_command(self) -> str:
        return render_serve_command(setup=self)

    @computed_field
    @property
    def openai_api_base(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def model_post_init(self, context):
        os.makedirs(self.log_dir, exist_ok=True)
        super().model_post_init(context)
        return
