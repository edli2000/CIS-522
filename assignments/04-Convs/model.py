import torch


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 32, 2, 2, activation='relu')
        self.dropout = torch.nn.Dropout(0.25)
        self.fc1 = torch.nn.Linear(800, 256)
        self.fc2 = torch.nn.Linear(256, num_classes)
        self.batch_norm1 = torch.nn.BatchNorm2d(num_features=32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        #x = torch.nn.functional.relu(x)
        x = self.batch_norm1(x)
        x = torch.nn.functional.avg_pool2d(x, 3)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x