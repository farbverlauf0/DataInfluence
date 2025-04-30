import os
import matplotlib.pyplot as plt
from typing import Iterable

def plot(all_losses: Iterable[str], base_path_to_graphics: str):
    all_samplers = list(all_losses['train'].keys())
    assert all_samplers == list(all_losses['test'].keys())
    for mode in ('train', 'test'):
        for sampler_type in all_samplers:
            losses = all_losses[mode][sampler_type]
            steps = range(1, len(losses) + 1)
            plt.plot(steps, losses, label=sampler_type, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("MSE")
        plt.title(f"Changing the loss function depending on the iterations ({mode})")
        plt.legend()
        plt.grid()
        graphics_path = os.path.join(base_path_to_graphics, f'{mode}.png')
        plt.savefig(graphics_path)
        plt.cla()
