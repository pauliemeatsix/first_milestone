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
          
                Conv2d(3, 32, kernel_size=3, padding=1),
                ReLU(),
                Conv2d(32, 64,kernel_size=3, padding=1),
                BatchNorm2d(64),
                ReLU(),
                MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                
                Conv2d(64, 128, kernel_size=3, padding=1),
                # ReLU(),
                # Conv2d(128, 128, kernel_size=3, padding=1),
                BatchNorm2d(128),
                ReLU(),
                MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
                
                Conv2d(128, 256, kernel_size=3, padding=1),
                # ReLU(),
                # Conv2d(256, 256,kernel_size=3, padding=1),
                BatchNorm2d(256),
                ReLU(),
                MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

                Flatten(start_dim=1, end_dim=-1),
                Linear(in_features=4096, out_features=512, bias=True),
                # ReLU(),
                # Linear(in_features=1024, out_features=512, bias=True),
                ReLU(),
                Linear(in_features=512, out_features=10, bias=True)
  )
        #       Conv2d(3, 6, 5,1),
        #       BatchNorm2d(6),
        #       ReLU(),
        #       MaxPool2d(2, 2),
        #       Conv2d(6, 16, 5,1),
        #       BatchNorm2d(16),
        #       ReLU(),

        #       Flatten(start_dim=1),
        #       Linear(16 * 5 * 5, 120),
        #       Linear(120, 84),
        #       Linear(84, 10),
        # )

            # Conv2d(3, 32, kernel_size=3, padding=1),
            # #BatchNorm2d(32),
            # ReLU(),
            # Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            
            # ReLU(),
            # MaxPool2d(2, 2), # output: 64 x 16 x 16

            # Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # ReLU(),
            # Conv2d(128, 128, kernel_size=3, stride=1, padding=1), 
            # ReLU(),
            # MaxPool2d(2, 2), # output: 128 x 8 x 8
            
            # Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            # ReLU(),
            # Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # #BatchNorm2d(256),
            # ReLU(),
            # MaxPool2d(2, 2), # output: 256 x 4 x 4

                        # Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            # ReLU(),

            # Conv2d(3,64, kernel_size=5), 
            # BatchNorm2d(64),
            # MaxPool2d(2, 2), 
            # ReLU(),

            # Conv2d(64,50, kernel_size=5), 
            # BatchNorm2d(50),
            # #Dropout2d(),
            # MaxPool2d(2, 2),
            # ReLU(),

            # Flatten(), 
            # Linear(50*5*5, 100),
            # BatchNorm1d(100),
            # ReLU(),
            # Linear(100, 500),
            # ReLU(),
            # Linear(500, 10)
            

        #     Conv2d(in_channels=3, out_channels=16, kernel_size=5, padding=2),
        #     BatchNorm2d(16),
        #     ReLU(),
        #     MaxPool2d(kernel_size=2, stride=2),
        #     Conv2d(in_channels=3, out_channels=32, kernel_size=5, padding=2),
        #     BatchNorm2d(32),
        #     ReLU(),
        #     Flatten(start_dim=1),
        #     Linear(in_features=1024, out_features=512),
        #     ReLU(),
        #     Linear(in_features=512, out_features=64),
        #     ReLU(),
        #     Linear(in_features=64, out_features=10)
        # )


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