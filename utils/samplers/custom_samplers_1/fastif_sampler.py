from ..base import AbstractSampler, IndexSampler
from .fast_influence.nn_utils import compute_influences
from .fast_influence.faiss_utils import FAISSIndex
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
import time


def get_dataset(x, y):
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)

    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset


def get_dataloader(x, y, batch_size=1, random=False):
    dataset = get_dataset(x, y)
    if random:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


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


class Net(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=64),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=512),
            nn.Sigmoid(),
            nn.Linear(in_features=512, out_features=2048),
            nn.Sigmoid(),
            nn.Linear(in_features=2048, out_features=512),
            nn.Sigmoid(),
            nn.Linear(in_features=512, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=128),
            nn.Sigmoid(),
            nn.Linear(in_features=128, out_features=1),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits


class FastIFSampler(AbstractSampler):
    def __init__(self, num_samples: int):
        self.model = Net(in_features=13)  # HERE WILL BE THE TRAINED MODEL!!!
        self.sampler = InfluenceSampler(num_samples)
        self.num_samples = num_samples

    def __call__(self, x, y, weight, *args, **kwargs):
        if (any(it not in kwargs for it in ["x_eval", "y_eval", "batch_size"])
                or not isinstance(kwargs["batch_size"], int)):
            raise ValueError

        x_eval = kwargs["x_eval"]
        y_eval = kwargs["y_eval"]

        # dataloader to compute estimations of hvp
        batch_train_data_loader = get_dataloader(x, y, batch_size=kwargs["batch_size"], random=True)
        # two dataloaders to compute influences
        instance_train_data_loader = get_dataloader(x, y, batch_size=1, random=False)
        eval_instance_data_loader = get_dataloader(x_eval, y_eval, batch_size=1, random=False)
        # number of test (validation) points for which Influence Function will be calculated
        num_examples_to_test = kwargs["num_examples_to_test"] if "num_examples_to_test" in kwargs else x_eval.shape[0]  #  // 4
        # number of batches on which hvp is calculated
        s_test_num_samples = min(x.shape[0] // kwargs["batch_size"] - 1, 1000)

        influences = np.zeros(shape=[num_examples_to_test, x.shape[0]])
        num_examples_tested = 0

        if "use_knn" in kwargs and kwargs["use_knn"]:
            index = FAISSIndex(x.shape[-1])
            index.add(x)
            nearest_neighbors = index.search(
                k=kwargs["knn_k_value"] if "knn_k_value" in kwargs else x.shape[0] // 10,
                queries=x_eval[:num_examples_to_test]
            )[1]
        else:
            nearest_neighbors = ["all"] * num_examples_to_test

        verbose = kwargs["verbose"] if "verbose" in kwargs else False
        verbose_frequency = kwargs["verbose_frequency"] if "verbose_frequency" in kwargs else 10

        for test_index, test_inputs in enumerate(eval_instance_data_loader):
            begin = time.time()
            if num_examples_tested >= num_examples_to_test:
                break

            influences[test_index] = compute_influences(
                batch_train_data_loader=batch_train_data_loader,
                instance_train_data_loader=instance_train_data_loader,
                model=self.model,
                test_inputs=test_inputs,
                s_test_num_samples=s_test_num_samples,
                s_test_iterations=1,
                train_indices_to_include=nearest_neighbors[test_index],
                fill_value=-np.inf,
            )

            num_examples_tested += 1

            if verbose and num_examples_tested % verbose_frequency:
                print(
                    "Influences calculated for %d points of %d. Estimated time left: %.2f" %
                      (num_examples_tested, num_examples_to_test,
                       (time.time() - begin)*(num_examples_to_test-num_examples_tested))
                )


        # IFs are calculated just for k nearest neighbors for each test instance so we have a lot of empty cells in
        # influence array. They are filled with -infinity so max function would ignore them. Sampler chooses
        # train points with the least influence values (i.e. the least harmful) so -infinity changed to +infinity
        # (just in case IF wasn't calculated for some train points).
        inf_test = np.nan_to_num(influences.max(axis=0), nan=np.inf, neginf=np.inf)
        # TODO: add possibility to change function 'max' by user's parameters

        return self.sampler(x, y, weight, influences=inf_test)
