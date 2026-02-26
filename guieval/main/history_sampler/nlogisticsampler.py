import random
import numpy as np
from pydantic import Field, model_validator
from typing import ClassVar, Self
from scipy.optimize import brentq

# subsec internal
from guieval.main.history_sampler.abcsampler import ABCHistorySampler


class NormalizedLogistic:
    MU_EPS = 1e-6

    @staticmethod
    def sigma(z) -> float:
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def nlogistic(cls,
                  x: float, kappa: float, mu: float, *,
                  reversed: bool = False) -> float:
        '''
        Linearly normalize `sigma` to [0, 0] ~ [1, 1]
        '''
        if x < 0 or x > 1:
            raise ValueError(f"x must be in [0, 1], but got {x}")

        y = cls.sigma(kappa * (x - mu))
        y0 = cls.sigma(kappa * (0 - mu))
        y1 = cls.sigma(kappa * (1 - mu))

        return ((y - y0) / (y1 - y0)) if not reversed else ((y - y1) / (y0 - y1))

    @classmethod
    def integral_area(cls, kappa: float, mu: float, *,
                      reversed: bool = False) -> tuple[float, float]:
        '''
        integral_0^1 nlogistic(x) dx in (0, 1)
        '''
        num = (
            (np.log(1 + np.exp(kappa * (1 - mu)))
            - np.log(1 + np.exp(-kappa * mu))) / kappa
            - cls.sigma(-kappa * mu)
        )
        den = cls.sigma(kappa * (1 - mu)) - cls.sigma(-kappa * mu)
        return num / den if not reversed else (1 - num / den)

    @classmethod
    def area_interval(cls, kappa, mu_eps: float = MU_EPS, *,
                      reversed: bool = False) -> tuple[float, float]:
        left_inflection = cls.integral_area(kappa=kappa, mu=mu_eps, reversed=reversed)
        right_infection = cls.integral_area(kappa=kappa, mu=(1 - mu_eps), reversed=reversed)
        return (right_infection, left_inflection) if not reversed else (left_inflection, right_infection)

    @classmethod
    def area_mapped_mu(cls, kappa: float, area: float, tol=1e-10, *,
                       reversed: bool = False):
        def mu_mapped_area(mu: float) -> float:
            return cls.integral_area(kappa=kappa, mu=mu, reversed=reversed) - area

        return brentq(mu_mapped_area, cls.MU_EPS, (1 - cls.MU_EPS), xtol=tol)

    def __init__(self,
                 kappa: float = 7.0,
                 mean: float = 0.5,
                 mu_eps: float = MU_EPS, *,
                 reversed: bool = False):
        self.kappa = kappa

        mean_interval = self.area_interval(kappa=kappa, reversed=reversed)
        if mean < mean_interval[0] or mean > mean_interval[1]:
            raise ValueError('Invalid value for `mean`. '
                             f'Proper mean interval for assigned kappa `{kappa}` is `{mean_interval}`')

        self.reversed = reversed
        self.mean = mean
        self.mu_eps = mu_eps

    def __call__(self,
                 x: float, *,
                 kappa: float | None = None,
                 mean: float | None = None) -> float:
        if kappa is None:
            kappa = self.kappa
        if mean is None:
            mean = self.mean

        mu = self.area_mapped_mu(kappa, area=mean, reversed=self.reversed)

        return self.nlogistic(x, kappa, mu, reversed=self.reversed)

    @classmethod
    def random(cls, kappa: float, *,
               reversed: bool = False) -> Self:
        area_interval = cls.area_interval(kappa=kappa, reversed=reversed)
        rd_area = random.uniform(*area_interval)

        return cls(kappa=kappa, mean=rd_area, reversed=reversed)


class NormalizedLogisticHistorySampler(ABCHistorySampler):
    NAME: ClassVar[str] = 'nlogistic'

    k: float | int = Field(default=7,
                            validate_default=True,
                            gt=0.0,
                            alias='logistic_k',
                            description="todo")

    @model_validator(mode='after')
    def _validate_mean(self) -> Self:
        if self.mean <= self.lb or self.mean >= self.ub:
            raise ValueError()
        return self

    def model_post_init(self, context):
        super().model_post_init(context)
        if self.mean is None:
            raise ValueError('First sampling prob mean must be assigned for `NormalizedLogisticHistorySampler`')

        _mean_residue = self.mean - self.lb
        self._distribution_width = self.ub - self.lb
        _width_scaled_mean_residue = _mean_residue / self._distribution_width

        self._step_ratio_distribution = NormalizedLogistic(kappa=self.k,
                                                           mean=_width_scaled_mean_residue,
                                                           reversed=(self.type == 'decay'))

    def _truncate_step_ratio(self, step_ratio: float) -> float:
        if step_ratio <= self._step_ratio_distribution.mu_eps:
            return self._step_ratio_distribution.mu_eps
        elif step_ratio < (1 - self._step_ratio_distribution.mu_eps):
            return step_ratio
        else:
            return (1 - self._step_ratio_distribution.mu_eps)

    def compute_prob_residue(self, step_ratio: float) -> float:
        return (self._distribution_width *
                self._step_ratio_distribution(
                    self._truncate_step_ratio(
                        step_ratio=step_ratio
                    )
                )
        )

    def estimate_prob(self, step_id: float, history_window: int) -> float:
        step_ratio = step_id / history_window
        step_ratio_stride = 1 / history_window
        step_neighbourhood_delta = step_ratio_stride / max(2, history_window)

        left_neighbour = self.lb + self.compute_prob_residue(step_ratio=(step_ratio - step_neighbourhood_delta))
        right_neighbour = self.lb + self.compute_prob_residue(step_ratio=(step_ratio + step_neighbourhood_delta))

        return float((left_neighbour + right_neighbour) / 2)

    def sampling_distribution_factory(self, history_window: int):
        return [self.estimate_prob(step_id=_step, history_window=history_window)
                for _step in range(1, history_window + 1)]
