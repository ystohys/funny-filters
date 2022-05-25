import torch
from torch import nn


def build_conv_block(in_feat, out_feat, activate='relu', **kwargs):
    activations = nn.ModuleDict({
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU()
    })
    conv_block = nn.Sequential(
        nn.Conv2d(in_feat, out_feat, **kwargs),
        nn.BatchNorm2d(out_feat),
        activations[activate]
    )

    return conv_block


def build_linear_block(in_feat, out_feat, activate='relu', **kwargs):
    activations = nn.ModuleDict({
        'relu': nn.ReLU(),
        'lrelu': nn.LeakyReLU()
    })
    linear_block = nn.Sequential(
        nn.Linear(in_feat, out_feat, **kwargs),
        activations[activate],
        nn.Dropout(p=0.1)
    )

    return linear_block


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = build_conv_block(1, 64, kernel_size=5, stride=1)
        self.conv2 = build_conv_block(64, 128, kernel_size=3, stride=1)
        self.conv3 = build_conv_block(128, 128, kernel_size=3, stride=1)
        self.conv4 = build_conv_block(128, 64, kernel_size=1, stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.flat = nn.Flatten()

        self.dense1 = build_linear_block(64*5*5, 128)
        self.dense2 = build_linear_block(128, 256)
        self.dense3 = build_linear_block(256, 128)

        self.pre_final_layer = nn.Linear(in_features=128, out_features=30)
        self.final_layer = nn.Linear(in_features=30, out_features=30)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))

        x = self.flat(x)

        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)

        x = self.pre_final_layer(x)
        x = self.final_layer(x)
        x = torch.reshape(x, (-1, 2, 15))
        # [[y1,y2,y3,...],
        #  [x1,x2,x3,...]]  -> output shape
        return x
