from torch import nn
import torchvision.models as models


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class deeplabv3(nn.Module):
    def __init__(self, outputs_nc, pretrained=False):
        super().__init__()

        print("Network pretrained: ", pretrained)
        self.model = models.segmentation.deeplabv3_resnet101(pretrained=pretrained)
        self.model.classifier[4] = Identity()
        print(outputs_nc)
        n_features = 256
        self.task_head = nn.ModuleList()
        self.task_head.append(
            nn.Conv2d(n_features, outputs_nc, kernel_size=(1, 1), stride=(1, 1))
        )

    def forward(self, x):
        x = self.model(x)["out"]
        out = self.task_head[0](x)
        return [out]
