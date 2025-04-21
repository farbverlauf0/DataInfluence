from ..base import AbstractSampler, IndexSampler


class InfluenceSampler(AbstractSampler):
    def __init__(self, num_samples: int):
        self.index_sampler = IndexSampler()
        self.num_samples = num_samples

    def __call__(self, x, y, weight, *args, **kwargs):
        if x.shape[0] != y.shape[0] or x.shape[0] != weight.shape[0]:
            raise ValueError

        influences = kwargs["influences"]
        index = sorted(range(x.shape[0]), key=lambda j: influences[j])[:self.num_samples]

        return self.index_sampler(x, y, weight, index=index)
