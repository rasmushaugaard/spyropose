# Initially from https://github.com/usuyama/pytorch-unet (MIT License)
# Architecture slightly changed (removed some expensive high-res convolutions)
import torch
import torchvision
from torch import nn


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )


class ResNetUNet(nn.Module):
    def __init__(self, n_class, feat_preultimate=64, norm_layer=None):
        super().__init__()

        self.base_model = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1,
            norm_layer=norm_layer,
        )
        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer1 = nn.Sequential(
            *self.base_layers[3:5]
        )  # size=(N, 64, x.H/4, x.W/4)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False
        )

        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)
        self.conv_original_size = convrelu(128, feat_preultimate, 3, 1)
        self.conv_last = nn.Conv2d(feat_preultimate, n_class, 1)

    def forward(self, input):
        # encoder
        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        layers = [layer0, layer1, layer2, layer3, layer4]

        # decoder
        x = self.layer4_1x1(layer4)
        x = self.upsample(x)
        for layer_idx in 3, 2, 1, 0:
            projection = getattr(self, f"layer{layer_idx}_1x1")(layers[layer_idx])
            x = torch.cat([x, projection], dim=1)
            x = getattr(self, f"conv_up{layer_idx}")(x)
            x = self.upsample(x)
        x = self.conv_original_size(x)
        x = self.conv_last(x)
        return x
