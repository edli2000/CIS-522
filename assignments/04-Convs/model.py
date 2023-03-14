import torch


class Model(torch.nn.Module):
    """
    Class for CNN model.
    """

    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Init method for the CNN.
        Arguments:
            num_channels: Number of channels to use
            num_classes: Number of classes to use
        Returns:
            None
        """
        super(Model, self).__init__()
        self.fc1 = torch.nn.Linear(288, 1000)
        self.fc2 = torch.nn.Linear(1000, num_classes)
        self.conv1 = torch.nn.Conv2d(num_channels, 8, 3)
        self.pool = torch.nn.MaxPool2d(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the CNN.
        Arguments:
            x: Torch Tensor
        Returns:
            x: Torch Tensor after forward
        """
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 288)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
