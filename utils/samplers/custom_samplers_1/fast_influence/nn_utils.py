import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Union, Optional, Tuple


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_loss_with_weight_decay(
        model: torch.nn.Module,
        inputs: Tuple[torch.tensor, torch.tensor],
) -> float:

    if torch.cuda.is_available():
        inputs = inputs[0].cuda(), inputs[1].cuda()
    outputs = model(inputs[0])
    loss = ((inputs[1] - outputs)**2).mean()

    return loss


def compute_gradients(
        model: torch.nn.Module,
        inputs: Tuple[torch.tensor, torch.tensor],
) -> List[torch.FloatTensor]:
    model.zero_grad()
    loss = get_loss_with_weight_decay(model=model, inputs=inputs,)

    return torch.autograd.grad(
        outputs=loss,
        inputs=model.parameters(),
        create_graph=True
    )


def compute_hessian_vector_products(
        model: torch.nn.Module,
        inputs: Tuple[torch.tensor, torch.tensor],
        vectors: torch.FloatTensor,
) -> List[torch.Tensor]:

    model.zero_grad()
    loss = get_loss_with_weight_decay(
        model=model,
        inputs=inputs
    )

    grad_tuple = torch.autograd.grad(
        outputs=loss,
        inputs=model.parameters(),
        create_graph=True
    )

    model.zero_grad()
    grad_grad_tuple = torch.autograd.grad(
        outputs=grad_tuple,
        inputs=model.parameters(),
        grad_outputs=vectors,
        only_inputs=True
    )

    return grad_grad_tuple


def compute_s_test(
        model: torch.nn.Module,
        test_inputs: Tuple[torch.tensor, torch.tensor],
        train_data_loaders: List[torch.utils.data.DataLoader],
        damp: float,
        scale: float,
        num_samples: Optional[int] = None,
) -> List[torch.Tensor]:

    v = compute_gradients(model=model, inputs=test_inputs)
    last_estimate = list(v).copy()
    cumulative_num_samples = 0
    for data_loader in train_data_loaders:
        for i, inputs in enumerate(data_loader):
            this_estimate = compute_hessian_vector_products(
                model=model,
                vectors=last_estimate,
                inputs=inputs,
            )
            with torch.no_grad():
                new_estimate = [
                    a + (1 - damp) * b - c / scale
                    for a, b, c in zip(v, last_estimate, this_estimate)
                ]

            cumulative_num_samples += 1
            last_estimate = new_estimate
            if num_samples is not None and i > num_samples:
                break

    inverse_hvp = [X / scale for X in last_estimate]

    if cumulative_num_samples not in [num_samples, num_samples + 2]:
        raise ValueError(f"cumulative_num_samples={cumulative_num_samples} f"
                         f"but num_samples={num_samples}: Untested Territory")

    return inverse_hvp


def compute_grad_zs(
        model: torch.nn.Module,
        data_loader: torch.utils.data.DataLoader,
) -> List[List[torch.Tensor]]:

    grad_zs = []
    for inputs in data_loader:
        grad_z = compute_gradients(model=model, inputs=inputs)
        with torch.no_grad():
            grad_zs.append([X.cpu() for X in grad_z])

    return grad_zs


def compute_influences(
        model: torch.nn.Module,
        test_inputs: Tuple[torch.tensor, torch.tensor],
        batch_train_data_loader: torch.utils.data.DataLoader,
        instance_train_data_loader: torch.utils.data.DataLoader,
        s_test_damp: float = 3e-5,
        s_test_scale: float = 1e4,
        s_test_num_samples: Optional[int] = None,
        s_test_iterations: int = 1,
        precomputed_s_test: Optional[List[torch.FloatTensor]] = None,
        train_indices_to_include: Optional[Union[np.ndarray, List[int], str]] = None,
        fill_value: float = np.nan
) -> Tuple[Dict[int, float], Dict[int, Dict], List[torch.FloatTensor]]:

    if s_test_iterations < 1:
        raise ValueError("`s_test_iterations` must >= 1")

    if precomputed_s_test is not None:
        s_test = precomputed_s_test
    else:
        s_test = None
        for _ in range(s_test_iterations):
            _s_test = compute_s_test(
                model=model,
                test_inputs=test_inputs,
                train_data_loaders=[batch_train_data_loader],
                damp=s_test_damp,
                scale=s_test_scale,
                num_samples=s_test_num_samples,
            )

            if s_test is None:
                s_test = _s_test
            else:
                s_test = [
                    a + b for a, b in zip(s_test, _s_test)
                ]
        s_test = [a / s_test_iterations for a in s_test]

    influences = np.full(len(instance_train_data_loader), fill_value=fill_value)
    for index, train_inputs in enumerate(instance_train_data_loader):

        if (isinstance(train_indices_to_include, np.ndarray) and index not in train_indices_to_include) \
            and (not isinstance(train_indices_to_include, str) or train_indices_to_include != "all"):
            continue

        grad_z = compute_gradients(model=model, inputs=train_inputs)

        with torch.no_grad():
            influence = [
                - torch.sum(x * y)
                for x, y in zip(grad_z, s_test)
            ]

        influences[index] = sum(influence).item()

    return influences


def run_full_influence_functions(
        model,
        batch_train_data_loader,
        instance_train_data_loader,
        eval_instance_data_loader,
        num_examples_to_test: int,
        s_test_num_samples: int = 1000
):
    num_examples_tested = 0
    outputs_collections = {}
    with tqdm(total=num_examples_to_test) as pbar:
        for test_index, test_inputs in enumerate(eval_instance_data_loader):
            if num_examples_tested >= num_examples_to_test:
                break

            influences, _, s_test = compute_influences(
                batch_train_data_loader=batch_train_data_loader,
                instance_train_data_loader=instance_train_data_loader,
                model=model,
                test_inputs=test_inputs,
                s_test_num_samples=s_test_num_samples,
                s_test_iterations=1,
            )

            outputs = {
                "test_index": test_index,
                "influences": influences,
                "s_test": s_test,
            }
            num_examples_tested += 1
            outputs_collections[test_index] = outputs
            pbar.update(1)

    return outputs_collections
