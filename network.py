import torch
import torch.nn.functional as F
from torch.nn import Conv2d, MaxPool2d, Linear, BatchNorm2d, BatchNorm1d, ReLU, Flatten

class Net(torch.nn.Module):
    def __init__(self):
        """
        This function initializes the Net class and defines the network architecture:

        Args:

        Returns:
        """

        super(Net,self).__init__()

        self.conv_net = torch.nn.Sequential(
            Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=2),
            ReLU(),
            Flatten(start_dim=1),
            Linear(in_features=6144, out_features=2000),
            ReLU(),
            Linear(in_features=2000, out_features=10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function receives the input x and pass it over the network, returning the model outputs:

        Args:
            - x (tensor): input data

        Returns:
            - out (tensor): network output given the input x
        """
        out = self.conv_net(x)
        
        return out