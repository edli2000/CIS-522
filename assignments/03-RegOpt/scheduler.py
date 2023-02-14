from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Class for custom learning rate scheduling
    """

    def __init__(self, optimizer, gamma=0.1, last_epoch=-1):
        """
        Create a new scheduler.

        Arguments:
            optimizer (torch.optim.Optimizer): model optimizer
            gamma (float): Learning Rate decay factor
            last_epoch (int): Last epoch index, defaulted to -1

        Returns:
            None

        """
        self.gamma = gamma
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Retrieves the list of learning rates

        Arguments:
            None

        Returns:
            None
        """
        return [i * self.gamma**self.last_epoch for i in self.base_lrs]
