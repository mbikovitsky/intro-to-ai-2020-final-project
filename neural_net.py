#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader


class Net(nn.Module):
    """
    Neural network for RNA degradation prediction.
    """

    def __init__(self):
        super().__init__()

        self.stage1 = create_conv_layer(4, 96, 12)
        self.stage2 = ResidualLayer(96, 5)
        self.stage3 = nn.AvgPool1d(10)
        self.stage4 = create_conv_layer(96, 196, 9)
        self.stage5 = nn.Linear(196, 1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = x.view(-1, 196)
        x = self.stage5(x)

        return x


class ResidualLayer(nn.Module):
    """
    A residual layer, as defined in the paper.
    """

    def __init__(self, in_channels: int, kernel_size: int):
        super().__init__()

        # The paper's code uses TensorFlow's SAME padding algorithm, which to the best
        # of my knowledge (https://stackoverflow.com/a/42195267/851560)
        # can be replicated using the 'padding' parameter.
        self.conv1 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=conv1d_same_padding(kernel_size),
        )
        self.norm1 = nn.BatchNorm1d(in_channels)

        self.conv2 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=conv1d_same_padding(kernel_size),
        )
        self.norm2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        temp = self.conv1(x)
        temp = self.norm1(temp)
        temp = F.relu(temp)

        temp = self.conv2(temp)
        temp = self.norm2(temp)

        x = x + temp
        x = F.relu(x)

        return x


def create_conv_layer(
    in_channels: int, out_channels: int, kernel_size: int
) -> nn.Module:
    """
    Creates a convolution layer, as described in the paper.
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )


def conv1d_same_padding(kernel_size: int) -> int:
    """
    Calculates necessary padding for a 1D convolution layer, so that the output shape
    becomes equal to the input shape. It is assumed the stride is 1.

    :param kernel_size: Size of the convolution kernel. Must be an odd integer.

    :return: Required padding.
    """
    # https://arxiv.org/abs/1603.07285
    assert kernel_size % 2 == 1
    return (kernel_size - 1) // 2


def train_network(network: Net, data_loader: DataLoader, epochs: int):
    network.train()

    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(network.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        for sequences, rates in data_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(sequences)
            loss = criterion(outputs, rates)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}")


def test_network(network: Net, data_loader: DataLoader) -> float:
    network.eval()

    with torch.no_grad():
        errors = []
        for (sequences, rates) in data_loader:
            outputs = network(sequences)
            errors.extend((outputs - rates) ** 2)

    mse = np.mean(errors)

    return mse
