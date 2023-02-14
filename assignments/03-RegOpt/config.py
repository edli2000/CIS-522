from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomRotation,
    RandomHorizontalFlip,
)


class CONFIG:
    batch_size = 32
    num_epochs = 8
    initial_learning_rate = 0.001
    initial_weight_decay = initial_learning_rate / num_epochs

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "lr": initial_learning_rate,
        "wd": initial_weight_decay,
        "num_epochs": num_epochs,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            RandomHorizontalFlip(),
            RandomRotation(10),
            ToTensor(),
            Normalize((0.5, 0.5, 0.5), (0.25, 0.25, 0.25)),
        ]
    )
