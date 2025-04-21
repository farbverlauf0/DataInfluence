from abc import ABC, abstractmethod
from random import sample, seed


class AbstractSampler(ABC):

    @abstractmethod
    def __call__(self, x, y, weight, *args, **kwargs):
        pass


class BaseSampler(AbstractSampler):
    def __init__(self):
        pass

    def __call__(self, x, y, weight, *args, **kwargs):
        return x, y, weight


class IndexSampler(AbstractSampler):
    def __init__(self):
        pass

    def __call__(self, x, y, weight, *args, **kwargs):
        index = kwargs["index"]
        return x[index], y[index], weight[index]


class RandomSampler(AbstractSampler):
    def __init__(self, num_samples: int):
        self.index_sampler = IndexSampler()
        self.num_samples = num_samples

    def __call__(self, x, y, weight, *args, **kwargs):
        if x.shape[0] != y.shape[0] or x.shape[0] != weight.shape[0]:
            raise ValueError

        if "seed" in kwargs and kwargs["seed"] is not None:
            seed(kwargs["seed"])

        index = sample(range(x.shape[0]), k=self.num_samples)

        return self.index_sampler(x, y, weight, index=index)
