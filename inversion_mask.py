import torch
import inversefed
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from torchvision.utils import save_image
import fire
import random


def flat_tensor_list(tensor_list):
    tensor_sizes = [x.size() for x in tensor_list]
    flatten_tensors = [x.flatten() for x in tensor_list]
    flatten_sizes = [len(x) for x in flatten_tensors]
    flat_weights = torch.cat(flatten_tensors, dim=0)
    return flat_weights, tensor_sizes, flatten_sizes


def plot(
    tensor,
    ds,
    dm,
    save_name=None,
):
    tensor = tensor.clone().detach()
    tensor.mul_(ds).add_(dm).clamp_(0, 1)
    plt.axis("off")
    if tensor.shape[0] == 1:
        plt.imshow(tensor[0].permute(1, 2, 0).cpu())
        if save_name:
            plt.savefig(save_name)


def mask_encrypt(input_parameters, gradients, policy, ratio):
    # Prepare the updates and gradients
    flat_updates, tensor_sizes, flatten_sizes = flat_tensor_list(input_parameters)
    flat_grads, _, _ = flat_tensor_list(gradients)
    total_number = len(flat_updates)
    subset_size = int(total_number * ratio)

    # Select indices
    if policy == "gradient":
        element_mul = 0 - flat_grads * flat_updates
        subset_indices = element_mul.topk(subset_size).indices.tolist()
    elif policy == "random":
        subset_indices = random.sample(range(0, total_number), int(subset_size))

    # The model updates on the selected position are unknown to attacker, set to 0
    # We can also set them to the average value which doesn't change the attack result.
    flat_updates[subset_indices] = 0

    # Reconstruct the updates
    reconstructed_tensors = []
    cursor = 0
    for ts, fs in zip(tensor_sizes, flatten_sizes):
        reconstructed_tensors.append(
            torch.reshape(flat_updates[cursor : cursor + fs], ts)
        )
        cursor += fs

    return reconstructed_tensors


def attack(model_arch, dataset, mask_policy, mask_ratio):
    assert model_arch in ["ResNet18", "LeNet5"]
    assert dataset in ["CIFAR10", "MNIST"]
    assert mask_policy in ["gradient", "random"]
    assert 0.0 <= mask_ratio <= 1.0

    arch = model_arch

    setup = inversefed.utils.system_startup()

    # Set up attack system
    defs = inversefed.training_strategy("conservative")
    defs.augmentations = False

    loss_fn, trainloader, validloader = inversefed.construct_dataloaders(dataset, defs)

    num_channels = 3 if dataset == "CIFAR10" else 1

    model, _ = inversefed.construct_model(
        arch, num_classes=10, num_channels=num_channels
    )
    model.to(**setup)
    model.eval()

    if dataset == "CIFAR10":
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif dataset == "MNIST":
        dm = torch.as_tensor(inversefed.consts.mnist_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.mnist_std, **setup)[:, None, None]

    # Set up attack ground truth
    input_id = 7
    ground_truth = trainloader.dataset[input_id][0].to("cuda").unsqueeze(0).contiguous()
    labels = torch.as_tensor(
        (trainloader.dataset[input_id][1],), device=setup["device"]
    )
    plot(ground_truth, ds, dm, save_name="ground_truth_%s.png" % dataset)

    # Simulate local training
    local_lr = 1e-4
    local_steps = 5
    use_updates = True

    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    input_parameters, gradients = inversefed.reconstruction_algorithms.loss_steps_grads(
        model,
        ground_truth,
        labels,
        lr=local_lr,
        local_steps=local_steps,
        use_updates=use_updates,
    )
    input_parameters = [p.detach() for p in input_parameters]

    # Apply encryption mask and get the exposed updates
    exposed_updates = mask_encrypt(input_parameters, gradients, mask_policy, mask_ratio)

    # Perform data reconstruction Attack
    config = dict(
        signed=True,
        boxed=True,
        cost_fn="sim",
        indices="def",
        weights="equal",
        lr=0.1,
        optim="adam",
        restarts=1,
        max_iterations=8_000,
        total_variation=1e-6,
        init="randn",
        filter="none",
        lr_decay=True,
        scoring_choice="loss",
    )

    rec_machine = inversefed.FedAvgReconstructor(
        model, (dm, ds), local_steps, local_lr, config, use_updates=use_updates
    )
    output, stats = rec_machine.reconstruct(
        exposed_updates,
        labels,
        img_shape=(3, 32, 32) if dataset == "CIFAR10" else (1, 28, 28),
    )

    plt.title(f"Rec. loss: {stats['opt']:2.4f}", fontsize=25)
    plot(output, ds, dm, "%s_%s_%s.png" % (dataset, mask_policy, mask_ratio))


if __name__ == "__main__":
    fire.Fire(attack)
