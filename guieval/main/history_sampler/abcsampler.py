import numpy as np
from pydantic import BaseModel, Field, model_validator, ConfigDict
from typing import Literal, ClassVar
from typing_extensions import Self

# subsec internal
from guieval.main.utils import CONTENT_SOURCE


# section struc
SAMPLER_TYPE = Literal['decay', 'increasing']


class HistorySamplerConfig(BaseModel):
    source_choices: tuple[CONTENT_SOURCE, CONTENT_SOURCE] | str = Field(
        default=("online_pos", "low_instruction"),
        validate_default=True,
        frozen=True,
        description=("Split the two choices with ','."))
    sampler_name: str | None = Field(default=None,
                              validate_default=True,
                              frozen=True,
                              description=("Note that explicitly assigned sampler name would override the behavior "
                                           "of assigned `eval_mode`s, including `offline_rule` and `semi_online`.\n"
                                           "The name of the history sampler. "
                                           "Current supported samplers:\n"
                                           "- `constant`: constant probability `self.lb` for all choices.\n"
                                           "- `linear`: linearly increasing or decreasing"
                                           " the probability of the first choice."))
    sampler_type: SAMPLER_TYPE = Field(default="decay",
                                       validate_default=True,
                                       frozen=True,
                                       description=('Type of the sampling probability distribution.\n'
                                                    '- decay: the sampling probability decays from lb to ub.\n'
                                                    '- increasing: the sampling probability increases from lb to ub.'))
    first_choice_prob_lb: float = Field(default=1.0,
                                        validate_default=True,
                                        ge=0.0, le=1.0,
                                        description=('Lower boundary of the step-i-decay sampling '
                                                     'probability of the first content source'
                                                     'It is the end point of the decay process, '
                                                     'or the starting point of the increasing process.'))
    first_choice_prob_ub: float = Field(default=1.0,
                                        validate_default=True,
                                        ge=0.0, le=1.0,
                                        description=('Upper boundary of the step-i-decay sampling probability '
                                                     'of the first content source. '
                                                     'It is the starting point of the decay process, '
                                                     'or the end point of the increasing process.'))
    logistic_k: float | int = Field(default=7,
                                    validate_default=True,
                                    gt=0.0,
                                    description="todo")
    mean: float | None = Field(default=None,
                               validate_default=True,
                               ge=0.0, le=1.0,
                               description='Mean of the given first-content-source-sampling distribution.')

    @model_validator(mode='before')
    @classmethod
    def _parse_source_choices(cls, data: dict) -> dict:
        source_choices = data.get('source_choices')
        if isinstance(source_choices, str):
            data['source_choices'] = [choice.strip() for choice in source_choices.split(',') if choice.strip()]

        return data


# section main
class ABCHistorySampler(BaseModel):
    model_config = ConfigDict(
        extra='allow'
    )

    # implicit registry on the global inheritence tree of all `ABCHistorySampler` subclasses
    _name_root_registry: ClassVar[dict[str, type['ABCHistorySampler']]] = dict()

    NAME: ClassVar[str] = 'constant'

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        if cls.NAME is __class__.NAME:
            raise ValueError('You are supposed to reassign the `NAME` class '
                             f'attribute of `{cls.__name__}` to a unique name.')
        elif cls.NAME in cls._name_root_registry:
            raise ValueError(f'For Sampler `{cls.__name__}`: the name `{cls.NAME}` '
                             f'is already registered for Sample  `{cls._name_root_registry[cls.NAME].__name__}`.')

        if cls.sampling_distribution_factory is __class__.sampling_distribution_factory:
            raise NotImplementedError('Your are supposed to implement the decaying method of '
                                      f'the first-content-source sampling probability for `{cls.__name__}`.')

        cls._name_root_registry[cls.NAME] = cls

    content_sources: tuple[CONTENT_SOURCE, CONTENT_SOURCE] = Field(default=("online_pos", "low_instruction"),
                                                                   validate_default=True,
                                                                   min_length=2, max_length=2,
                                                                   alias='source_choices')
    lb: float = Field(default=1.0,
                      validate_default=True,
                      ge=0.0, le=1.0,
                      alias='first_choice_prob_lb',
                      description=('Lower boundary of the step-i-decay sampling probability of the first content source'
                                   'It is the end point of the decay process, '
                                   'or the starting point of the increasing process.'))
    ub: float = Field(default=1.0,
                      validate_default=True,
                      ge=0.0, le=1.0,
                      alias='first_choice_prob_ub',
                      description=('Upper boundary of the step-i-decay '
                                   'sampling probability of the first content source. '
                                   'It is the starting point of the decay process, '
                                   'or the end point of the increasing process.'))
    mean: float | None = Field(default=None,
                               validate_default=True,
                               ge=0.0, le=1.0,
                               description='Mean of the given first-content-source-sampling distribution.')

    type: SAMPLER_TYPE = Field(default='decay',
                               validate_default=True,
                               alias='sampler_type',
                               description=('Type of the sampling probability distribution.\n'
                                            '- decay: the sampling probability decays from lb to ub.\n'
                                            '- increasing: the sampling probability increases from lb to ub.'))

    @classmethod
    def get_sampler(cls, name: str, *,
                    default=None) -> 'ABCHistorySampler':
        if name == __class__.NAME:
            return __class__
        try:
            return cls._name_root_registry[name]
        except KeyError:
            if default is None:
                raise KeyError(f'No sampler named `{name}` is registered.')
            else:
                return default

    @staticmethod
    def ensure_binary_probs(p, tol=1e-12):
        if p <= tol:  # cut the prob under tolerance
            return np.array([0.0, 1.0])
        elif p >= 1.0 - tol:
            return np.array([1.0, 0.0])

        probs = np.array([p, 1.0 - p], dtype=float)
        probs = probs / probs.sum()

        return probs

    @model_validator(mode='after')
    def _validate_lb_ub(self) -> Self:
        if self.lb > self.ub:
            raise ValueError('The lower boundary of the sampling probability '
                             'should be less than or equal to the upper boundary.')
        return self

    def sampling_distribution_factory(self, history_window: int) -> list[float]:
        '''
        You are supposed to construct a proper decay process based on the given task.
        This decay process can be taken as the core definition of this sampler.

        For step task i (zero-indexed):
            This method should return a list of sampling probabilities with length of i: [p_0, p_1, ..., p_i-1]
            where p_j is the sampling probability of the first content
            source for the j-th content source inferred from the j-th step task,
            and then, the sampling probability of the second content source
            for the j-th content source is 1 - p_j.
        '''
        return [self.lb for _ in range(history_window)]

    def sample(self, history_window: int) -> list[CONTENT_SOURCE]:
        return [str(np.random.choice(self.content_sources, p=self.ensure_binary_probs(p=first_content_source_prob)))
                for first_content_source_prob in self.sampling_distribution_factory(history_window=history_window)]

    # todo register all the fields of the subclasses into SamplerConfig
