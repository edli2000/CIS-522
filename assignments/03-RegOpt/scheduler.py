from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Class for custom learning rate scheduling
    """

    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        """
        Create a new scheduler.

        Arguments:
            optimizer (torch.optim.Optimizer): model optimizer
            gamma (float): Learning Rate decay factor
            last_epoch (int): Last epoch index, defaulted to -1

        Returns:
            None

        """
        self.lr = kwargs["lr"]
        self.wd = kwargs["wd"]
        self.num_epochs = kwargs["num_epochs"]

        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Retrieves the list of learning rates

        Arguments:
            None

        Returns:
            None
        """
        return [self.lr / (1 + i * self.wd) for i in range(self.num_epochs)]
        # return [i * self.gamma**(self.last_epoch // self.step_size) for i in self.base_lrs]
