#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import re
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def occurrences(string: Union[str, pd.Series, pd.Index], sub: str) -> np.ndarray:
    """
    Count the overlapping occurrences of a string in another string.
    """
    if isinstance(string, str):
        string = pd.Series([string])
    # https://stackoverflow.com/a/11706065/851560
    return string.str.count(f"(?={re.escape(sub)})").to_numpy()


def match_parens(string: str, dtype=np.uint8) -> np.ndarray:
    """
    Returns a matrix of matching parentheses. For each pair of indices i, j
    in the input string, the cell (i, j) in the matrix will have a value of 1
    iff i and j contain a matching pair of parens.
    """

    pairs_matrix = np.zeros((len(string), len(string)), dtype=dtype)

    stack = []
    for index, char in enumerate(string):
        if char == "(":
            stack.append(index)
        elif char == ")":
            open_index = stack.pop()
            pairs_matrix[open_index, index] = 1
            pairs_matrix[index, open_index] = 1
    assert not stack

    return pairs_matrix


def conv_1d_output_length(
    length_in: int,
    kernel_size: int,
    padding: int = 0,
    dilation: int = 1,
    stride: int = 1,
) -> int:
    """
    Calculates the length of the output tensor for a 1D convolutional layer.

    See https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html for more
    information about the arguments to this function.

    :param length_in:   Length of the input tensor.
    :param kernel_size: Size of the convolution kernel.
    :param padding:     Padding applied to the convolution.
    :param dilation:    Dilation applied to the convolution.
    :param stride:      Convolution stride.
    """
    return conv_2d_output_length(
        length_in, length_in, kernel_size, padding, dilation, stride
    )[0]


def conv_1d_same_padding(kernel_size: int) -> int:
    """
    Calculates necessary padding for a 1D convolution layer, so that the output shape
    becomes equal to the input shape. It is assumed the stride is 1.

    :param kernel_size: Size of the convolution kernel. Must be an odd integer.

    :return: Required padding.
    """
    # https://arxiv.org/abs/1603.07285
    assert kernel_size % 2 == 1
    return (kernel_size - 1) // 2


def conv_2d_output_length(
    height_in: int,
    width_in: int,
    kernel_size: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    stride: Union[int, Tuple[int, int]] = 1,
) -> Tuple[int, int]:
    """
    Calculates the width and height of the output tensor for a 2D convolutional layer.

    See https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html for more
    information about the arguments to this function.

    :param height_in:   Height of the input tensor.
    :param width_in:    Width of the input tensor.
    :param kernel_size: Size of the convolution kernel.
    :param padding:     Padding applied to the convolution.
    :param dilation:    Dilation applied to the convolution.
    :param stride:      Convolution stride.

    :return: A tuple of the output height and width.
    """
    if not isinstance(kernel_size, Tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(padding, Tuple):
        padding = (padding, padding)
    if not isinstance(dilation, Tuple):
        dilation = (dilation, dilation)
    if not isinstance(stride, Tuple):
        stride = (stride, stride)

    def formula(height_or_width: int, index: int) -> int:
        return int(
            (
                height_or_width
                + 2 * padding[index]
                - dilation[index] * (kernel_size[index] - 1)
                - 1
            )
            / stride[index]
            + 1
        )

    height_out = formula(height_in, 0)
    width_out = formula(width_in, 1)

    return height_out, width_out


def max_pool_2d_output_length(
    height_in: int,
    width_in: int,
    kernel_size: Union[int, Tuple[int, int]],
    padding: Union[int, Tuple[int, int]] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    stride: Optional[Union[int, Tuple[int, int]]] = None,
) -> Tuple[int, int]:
    """
    Calculates the width and height of the output tensor for a 2D max pooling layer.

    See https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html for more
    information about the arguments to this function.

    :param height_in:   Height of the input tensor.
    :param width_in:    Width of the input tensor.
    :param kernel_size: Size of the pooling kernel.
    :param padding:     Padding applied to the pooling.
    :param dilation:    Dilation applied to the pooling.
    :param stride:      Pooling stride.

    :return: A tuple of the output height and width.
    """
    if not isinstance(kernel_size, Tuple):
        kernel_size = (kernel_size, kernel_size)
    if not isinstance(padding, Tuple):
        padding = (padding, padding)
    if not isinstance(dilation, Tuple):
        dilation = (dilation, dilation)

    if stride is None:
        stride = kernel_size
    if not isinstance(stride, Tuple):
        stride = (stride, stride)

    def formula(height_or_width: int, index: int) -> int:
        return int(
            (
                height_or_width
                + 2 * padding[index]
                - dilation[index] * (kernel_size[index] - 1)
                - 1
            )
            / stride[index]
            + 1
        )

    height_out = formula(height_in, 0)
    width_out = formula(width_in, 1)

    return height_out, width_out


def train_network(
    network: nn.Module,
    data_loader: DataLoader,
    epochs: int,
    criterion: nn.Module,
    optimizer,
    verbose: bool = True,
):
    """
    Trains a PyTorch neural network.

    :param network:     Instance of the network to train.
    :param data_loader: PyTorch DataLoader that returns batches of network inputs and
                        the corresponding correct outputs.
    :param epochs:      Number of training epochs.
    :param criterion:   Loss function to use. For instance, nn.MSELoss.
    :param optimizer:   Optimizer to use for training. For instance, torch.optim.Adam.
    :param verbose:     If True, prints a message to stdout each time a training epoch
                        is completed.
    """
    network.train()

    for epoch in range(epochs):
        for sequences, rates in data_loader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(sequences)
            loss = criterion(outputs, rates)
            loss.backward()
            optimizer.step()

        if verbose:
            print(f"Epoch {epoch + 1}")


def test_network(network: nn.Module, data_loader: DataLoader) -> float:
    network.eval()

    with torch.no_grad():
        errors = []
        for sequences, rates in data_loader:
            outputs = network(sequences)
            errors.extend((outputs - rates) ** 2)

    return np.mean(errors)
