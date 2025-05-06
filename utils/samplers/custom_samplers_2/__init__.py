import numpy as np
from pydvl.value import compute_shapley_values
from pydvl.utils import Dataset, Utility
from pydvl.value.shapley import ShapleyMode, MaxUpdates
from sklearn.ensemble import HistGradientBoostingRegressor
from ..base import AbstractSampler, IndexSampler


class ShapleySampler(AbstractSampler):
    def __init__(self, num_samples):
        super().__init__()
        self.num_samples = num_samples
        self.index_sampler = IndexSampler()

    def __call__(self, x, y, weight, *args, **kwargs):
        x_eval, y_eval = kwargs['x_eval'], kwargs['y_eval']
        dataset = Dataset(x_train=x, y_train=y, x_test=x_eval, y_test=y_eval)
        model = HistGradientBoostingRegressor(max_iter=50, max_depth=3)
        utility = Utility(model, dataset)

        shapley_values_tmc = compute_shapley_values(
            utility,
            mode=ShapleyMode.TruncatedMontecarlo,
            done=MaxUpdates(3),
            n_jobs=1,
            progress=True
        )

        indices = np.argsort(shapley_values_tmc.values)[-self.num_samples:]

        return self.index_sampler(x, y, weight, index=indices)
