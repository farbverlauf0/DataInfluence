from ..base import AbstractSampler
from .fastif_sampler import InfluenceSampler, get_dataloader, train_nn, Net

import torch
from torch import nn
import numpy as np
from tqdm import tqdm


class RepresenterPointSampler(AbstractSampler):
    model: nn.Module

    def __init__(self, num_samples: int):
        super().__init__()
        self.sampler = InfluenceSampler(num_samples)
    
    def __call__(self, x, y, weight=None, *args, **kwargs):
        if weight is None:
            weight = torch.ones_like(y)
        if any(it not in kwargs for it in ["x_eval", "y_eval"]):
            raise ValueError
        x_eval = kwargs["x_eval"]
        y_eval = kwargs["y_eval"]

        batch_size = kwargs["batch_size"] if "batch_size" in kwargs else 4096

        if "weight_decay" not in kwargs or kwargs["weight_decay"] <= 0:
            weight_decay = 2e-5
        else:
            weight_decay = kwargs["weight_decay"]

        loss_function = kwargs["loss_function"] if "loss_function" in kwargs else nn.MSELoss()

        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = Net(in_features=x.shape[-1]).to(device)
        train_nn(
            self.model,
            get_dataloader(x, y, batch_size=batch_size, random=True),
            x_eval, y_eval,
            kwargs["num_epochs"] if "num_epochs" in kwargs else 50, .0001,
        )
        # to get the stable results we should freeze all layers except the last and train model's last layer until it converges
        self.model.freeze_all_but_last_layers()
        train_nn(
            self.model,
            get_dataloader(x, y, batch_size=batch_size, random=True),
            x_eval, y_eval,
            10, .0001,
            weight_decay=weight_decay
        )

        instance_train_dataloader = get_dataloader(x, y, batch_size=1, random=False)
        test_dataloader = get_dataloader(x_eval, y_eval, batch_size=batch_size, random=False)
        influences = np.zeros(shape=[y_eval.shape[0], x.shape[0]])

        self.model.eval()
        with tqdm(total=len(instance_train_dataloader)) as pbar:
            for i, (train_input, train_output) in enumerate(instance_train_dataloader):
                train_input, train_output = train_input.to(device), train_output.to(device)
                f = self.model.get_last_layer_input(train_input)  # not sure may be needed method .to(device)
                loss = loss_function(self.model.run_through_last_layer(f), train_output)
                loss.backward()
                # we represent the model as F(x, Theta) = sigma(Phi(x, Theta)) = sigma(Theta_1 @ f) where f is the last layer input
                # The influence of training point z_i on test point z_te can be estimated as I(x_i, z_te) = alpha_i * f_i @ f_te * y_te
                # where alpha_i = dL(Theta_1 @ f, y) / d(Theta_1 @ f)
                # torch returns grad by parameters:
                # dL(Theta_1 @ f, y) / d(Theta_1) = dL(Theta_1 @ f, y) / d(Theta_1 @ f) * d(Theta_1 @ f) / d(Theta_1)
                # The last multiplier is f so to calculate dL(Theta_1 @ f, y) / d(Theta_1 @ f) we divide (by element)
                # two same shape tensors and receive the tensor (the same shape as the previous two) each component of which
                # is alpha
                alpha = (self.model.net[-1].weight.grad / f).mean() / torch.tensor([weight_decay * x.shape[0]]).to(device)
                for j, (test_input, test_output) in enumerate(test_dataloader):
                    test_input, test_output = test_input.to(device), test_output.to(device)
                    influences[j * batch_size:(j + 1) * batch_size, i] = (
                            alpha * (f @ self.model.get_last_layer_input(test_input).transpose(1, 0)) * test_output
                    ).to("cpu").numpy()
                pbar.update()

        return self.sampler(x, y, weight, influences=influences.sum(axis=0))


if __name__ == "__main__":
    net = Net(in_features=1)
    print(list(net.parameters()))