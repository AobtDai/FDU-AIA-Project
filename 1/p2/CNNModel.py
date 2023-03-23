import torch.nn as nn

class CNNModel(nn.Module):
    
    def __init__(self):
        super(CNNModel, self).__init__()
        # # try 3:
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 6, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(6, 16, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(16, 32, kernel_size=3),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Flatten(),
        #     nn.Linear(32*2*2, 120),
        #     nn.Sigmoid(),
        #     nn.Linear(120, 84),
        #     nn.Sigmoid(),
        #     nn.Linear(84, 12),
        # )

        # try 2:
        self.layers = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(6, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
            nn.Linear(16*6*6, 120),
            nn.Sigmoid(),
            nn.Linear(120, 84),
            nn.Sigmoid(),
            nn.Linear(84, 12),
        )

        # # try 1:
        # self.layers = nn.Sequential(
        #     nn.Conv2d(1, 6, kernel_size=5, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Conv2d(6, 16, kernel_size=5),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),

        #     nn.Flatten(),
        #     nn.Linear(16*5*5, 120),
        #     nn.Sigmoid(),
        #     nn.Linear(120, 84),
        #     nn.Sigmoid(),
        #     nn.Linear(84, 12),
        # )
    
    def forward(self, x):
        x = self.layers(x)
        return x