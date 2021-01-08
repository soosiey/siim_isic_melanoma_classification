import torch.nn as nn
from efficientnet_pytorch import EfficientNet
class CustomENet(nn.Module):
    def __init__(self):
        super(CustomENet, self).__init__()

        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.addon = nn.Linear(1000,2)

    def forward(self,x):
        out = self.backbone(x)
        out = self.addon(out)
        out = out.view(-1,2)
        return out

