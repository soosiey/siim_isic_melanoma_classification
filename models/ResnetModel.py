import torch.nn as nn
import torchvision.models as models
class CustomResnet(nn.Module):
    def __init__(self):
        super(CustomResnet, self).__init__()

        self.backbone = models.resnet50(pretrained=True)
        self.addon = nn.Linear(1000,2)

    def forward(self,x):
        out = self.backbone(x)
        out = self.addon(out)
        out = out.view(-1,2)
        return out

