import logging
from openai import OpenAI
from vllm import SamplingParams
from pydantic import BaseModel, Field, computed_field, model_serializer, SerializerFunctionWrapHandler, ConfigDict
from typing import Optional, Union, Any


# subsec internal
from vllm_serve.setup import ModelSetup
from vllm_serve.utils import StrictMessages
from utils import get_existing_logger


# section main
class ExtraInferenceSetup(BaseModel):
    '''
    see
    https://docs.vllm.ai/en/v0.8.3/serving/openai_compatible_server.html
    `Chat API-Extra parameters` for details
    '''
    model_config = ConfigDict(
        extra='allow'
    )

    use_beam_search: bool = False
    length_penalty: float = 1.0
    enable_thinking: bool = False  # private field

    echo: bool = Field(
        default=False,
        description=(
            "If true, the new message will be prepended with the last message "
            "if they belong to the same role."),
    )
    add_generation_prompt: bool = Field(
        default=True,
        description=(
            "If true, the generation prompt will be added to the chat template. "
            "This is a parameter used by chat template in tokenizer config of the "
            "model."),
    )
    continue_final_message: bool = Field(
        default=False,
        description=(
            "If this is set, the chat will be formatted so that the final "
            "message in the chat is open-ended, without any EOS tokens. The "
            "model will continue this message rather than starting a new one. "
            "This allows you to \"prefill\" part of the model's response for it. "
            "Cannot be used at the same time as `add_generation_prompt`."),
    )
    add_special_tokens: bool = Field(
        default=False,
        description=(
            "If true, special tokens (e.g. BOS) will be added to the prompt "
            "on top of what is added by the chat template. "
            "For most models, the chat template takes care of adding the "
            "special tokens so this should be set to false (as is the "
            "default)."),
    )
    documents: Optional[list[dict[str, str]]] = Field(
        default=None,
        description=(
            "A list of dicts representing documents that will be accessible to "
            "the model if it is performing RAG (retrieval-augmented generation)."
            " If the template does not support RAG, this argument will have no "
            "effect. We recommend that each document should be a dict containing "
            "\"title\" and \"text\" keys."),
    )
    chat_template: Optional[str] = Field(
        default=None,
        description=(
            "A Jinja template to use for this conversion. "
            "As of transformers v4.44, default chat template is no longer "
            "allowed, so you must provide a chat template if the tokenizer "
            "does not define one."),
    )
    mm_processor_kwargs: Optional[dict[str, Any]] = Field(
        default=None,
        description=("Additional kwargs to pass to the HF processor."),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description=("If specified, the output will follow the JSON schema."),
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[list[str]] = Field(
        default=None,
        description=(
            "If specified, the output will be exactly one of the choices."),
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description=(
            "If specified, the output will follow the context free grammar."),
    )
    guided_decoding_backend: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default guided decoding backend "
            "of the server for this specific request. If set, must be either "
            "'outlines' / 'lm-format-enforcer'"),
    )
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description=(
            "If specified, will override the default whitespace pattern "
            "for guided json decoding."),
    )
    priority: int = Field(
        default=0,
        description=(
            "The priority of the request (lower means earlier handling; "
            "default: 0). Any priority other than 0 will raise an error "
            "if the served model does not use priority scheduling."),
    )
    return_tokens_as_token_ids: Optional[bool] = Field(
        default=None,
        description=(
            "If specified with 'logprobs', tokens are represented "
            " as strings of the form 'token_id:{token_id}' so that tokens "
            "that are not JSON-encodable can be identified."))

    @computed_field
    @property
    def chat_template_kwargs(self) -> Optional[dict[str, Any]]:
        return dict(enable_thinking=self.enable_thinking)

    @model_serializer(mode='wrap')
    def _remove_private_field(self, handler: SerializerFunctionWrapHandler) -> dict[str, object]:
        serialized: dict = handler(self)
        serialized.pop('enable_thinking', None)
        return serialized

    def update_sampling_params(self, sampling_params: SamplingParams):
        self.best_of = sampling_params.best_of
        self.top_k = sampling_params.top_k
        self.min_p = sampling_params.min_p
        self.repetition_penalty = sampling_params.repetition_penalty
        self.stop_token_ids = sampling_params.stop_token_ids
        self.include_stop_str_in_output = sampling_params.include_stop_str_in_output
        self.ignore_eos = sampling_params.ignore_eos
        self.min_tokens = sampling_params.min_tokens
        self.skip_special_tokens = sampling_params.skip_special_tokens
        self.spaces_between_special_tokens = sampling_params.spaces_between_special_tokens
        self.truncate_prompt_tokens = sampling_params.truncate_prompt_tokens
        self.prompt_logprobs = sampling_params.prompt_logprobs


class DeployedModel:
    def __init__(self,
                 setup: ModelSetup,
                 *,
                 logger: logging.Logger | None = None):
        self._setup = setup
        self._client = OpenAI(api_key=setup.api_key,
                              base_url=setup.openai_api_base,
                              max_retries=setup.max_retries)

        self._logger = (get_existing_logger(setup.model_alias)
                        if logger is None else
                        logger)

    def chat(self,
             messages: StrictMessages,
             sampling_params: SamplingParams, *,
             extra_params: ExtraInferenceSetup = ExtraInferenceSetup()):
        extra_params.update_sampling_params(sampling_params=sampling_params)

        return self._client.chat.completions.create(
            messages=messages,
            model=self._setup.model_alias,
            max_completion_tokens=sampling_params.max_tokens,
            n=sampling_params.n,
            seed=sampling_params.seed,
            stop=sampling_params.stop,
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            extra_body=extra_params.model_dump())

    def _is_alive(self, *,
                  sampling_params: SamplingParams = SamplingParams(n=1, max_tokens=128),
                  extra_params: ExtraInferenceSetup = ExtraInferenceSetup(),
                  inform: bool = True):
        try:
            test_msgs = [{'role': 'user',
                          'content': 'Aloha! Introduce yourself. '}]
            test_request = self.chat(messages=test_msgs,
                                     sampling_params=sampling_params,
                                     extra_params=extra_params)

            self._logger.info(f'Service is alive:\n'
                              f'\t{test_request.choices[0].message.content}')

            return True
        except Exception as err:
            if inform:
                self._logger.info(f'Service is still offline: {err}')
            return False
