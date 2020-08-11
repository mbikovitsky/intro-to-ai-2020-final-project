#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Implementations of various neural networks for RNA degradation rate prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from util import conv_1d_same_padding


class ResidualDegrade(nn.Module):
    """
    Neural network for RNA degradation prediction, based on ResidualBind
    (https://doi.org/10.1101/418459).
    """

    def __init__(
        self,
        stage1_conv_channels: int = 96,
        stage1_conv_kernel_size: int = 12,
        stage2_conv_kernel_size: int = 5,
        stage3_pool_kernel_size: int = 10,
        stage4_conv_channels: int = 196,
    ):
        super().__init__()

        # The input to stage 3 has the same length as the output of stage 1 (see
        # illustration in the paper). That length is a result of the convolution,
        # which is performed without padding.
        stage3_pool_in_length = 110 - stage1_conv_kernel_size + 1

        # Calculate stage 3's output length according to:
        # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html
        stage3_pool_out_length = int(
            (stage3_pool_in_length - stage3_pool_kernel_size) / stage3_pool_kernel_size
            + 1
        )

        self.stage1 = create_conv_layer(
            4, stage1_conv_channels, stage1_conv_kernel_size
        )
        self.stage2 = ResidualLayer(stage1_conv_channels, stage2_conv_kernel_size)
        self.stage3 = nn.AvgPool1d(stage3_pool_kernel_size)
        self.stage4 = create_conv_layer(
            stage1_conv_channels, stage4_conv_channels, stage3_pool_out_length
        )
        self.stage5 = nn.Linear(stage4_conv_channels, 1)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(torch.flatten(x, start_dim=1))

        return x


class ConvDegrade(nn.Module):
    """
    A convolutional neural net using only the RNA sequence.
    """

    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(3, 128, 8),
            nn.ReLU(),
            nn.MaxPool1d(4, stride=2),
            nn.Conv1d(128, 128, 8),
            nn.ReLU(),
            nn.AvgPool1d(4, stride=2),
            nn.Flatten(),
            nn.Linear(2560, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.layers(x)


class SSDegrade(nn.Module):
    """
    A convolutional neural net using both the RNA sequence and the secondary structure.
    """

    def __init__(self):
        super().__init__()

        self.sequence_layers = nn.Sequential(
            nn.Conv1d(4, 96, 7), nn.ReLU(), nn.Flatten(),
        )

        self.ss_layers = nn.Sequential(
            nn.Conv2d(1, 8, (20, 110)), nn.ReLU(), nn.AvgPool2d((91, 1)), nn.Flatten(),
        )

        self.combined_layers = nn.Sequential(
            nn.Conv1d(1, 96, 7), nn.ReLU(), nn.Flatten(), nn.Linear(958656, 1),
        )

    def forward(self, x: torch.Tensor):
        sequences = x[:, : 110 * 4].view(x.shape[0], -1, 4).transpose(1, 2)

        secondary_structures = x[:, 110 * 4 :].view(-1, 1, 110, 110)

        sequence_features = self.sequence_layers(sequences)
        ss_features = self.ss_layers(secondary_structures)

        combined_features = torch.cat((sequence_features, ss_features), dim=1,)
        combined_features = combined_features.view(combined_features.shape[0], 1, -1)

        result = self.combined_layers(combined_features)

        return result


class ResidualLayer(nn.Module):
    """
    A residual layer, as defined in the ResidualBind paper.
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
            padding=conv_1d_same_padding(kernel_size),
        )
        self.norm1 = nn.BatchNorm1d(in_channels)

        self.conv2 = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            padding=conv_1d_same_padding(kernel_size),
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
    Creates a convolution layer, as described in the ResidualBind paper.
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(),
    )
