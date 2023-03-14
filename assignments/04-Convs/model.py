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
        self.conv1 = torch.nn.Conv2d(num_channels, 32, 2, 2, activation="relu")
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(800, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward method for the CNN.
        Arguments:
            x: Torch Tensor
        Returns:
            x: Torch Tensor after forward
        """
        x = self.conv1(x)
        # x = torch.nn.functional.relu(x)
        x = self.batch_norm1(x)
        x = torch.nn.functional.avg_pool2d(x, 3)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x
