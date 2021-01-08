import torch.nn as nn


class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3,16,3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,64,5),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64,128,3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128,256,5),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256,64,3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64,32,5),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fcblock = nn.Sequential(
            nn.Linear(32*22*22,1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512,2)
        )
    def forward(self,x):
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = out.view(-1, 32*22*22)
        out = self.fcblock(out)
        out = out.view(-1,2)
        return out


