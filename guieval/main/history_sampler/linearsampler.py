import numpy as np
from typing import ClassVar

# subsec internal
from guieval.main.history_sampler.abcsampler import ABCHistorySampler


class LinearHistorySampler(ABCHistorySampler):
    NAME: ClassVar[str] = 'linear'

    def sampling_distribution_factory(self, history_window):
        if self.type == "increasing":
            return np.linspace(self.lb, self.ub, num=history_window).tolist()
        elif self.type == "decay":
            return np.linspace(self.ub, self.lb, num=history_window).tolist()
