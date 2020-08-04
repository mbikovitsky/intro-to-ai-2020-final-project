#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod

import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_X_y, check_array
from sklearn.utils.validation import check_is_fitted
from torch import from_numpy, nn
from torch.utils.data import DataLoader, TensorDataset

from neural_net import Net
from util import train_network


class NeuralNetEstimator(ABC, BaseEstimator, RegressorMixin):
    @abstractmethod
    def __init__(self):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    def fit(self, X, y, training_epochs: int = 1) -> "NeuralNetEstimator":
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        X = from_numpy(X).to(self.device)
        y = from_numpy(y.reshape(-1, 1)).to(self.device)

        network = self._create_network().to(self.device)

        dataset = TensorDataset(X, y)
        data_loader = DataLoader(dataset, batch_size=4, shuffle=True)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(network.parameters(), lr=0.001)
        train_network(
            network,
            data_loader,
            training_epochs,
            criterion=criterion,
            optimizer=optimizer,
            verbose=False,
        )
        network.eval()

        # noinspection PyAttributeOutsideInit
        self._network = network

        return self

    def predict(self, X):
        check_is_fitted(self, attributes="_network")

        X = check_array(X, ensure_2d=False, allow_nd=True)
        X = from_numpy(X).to(self.device)

        with torch.no_grad():
            return self._network(X).cpu().numpy().reshape(-1)

    @abstractmethod
    def _create_network(self) -> nn.Module:
        pass


class PaperEstimator(NeuralNetEstimator):
    def __init__(
        self,
        stage1_conv_channels: int = 96,
        stage1_conv_kernel_size: int = 12,
        stage2_conv_kernel_size: int = 5,
        stage3_pool_kernel_size: int = 10,
        stage4_conv_channels: int = 196,
    ):
        super().__init__()

        self.stage1_conv_channels = stage1_conv_channels
        self.stage1_conv_kernel_size = stage1_conv_kernel_size
        self.stage2_conv_kernel_size = stage2_conv_kernel_size
        self.stage3_pool_kernel_size = stage3_pool_kernel_size
        self.stage4_conv_channels = stage4_conv_channels

    def _create_network(self) -> nn.Module:
        return Net(
            stage1_conv_channels=self.stage1_conv_channels,
            stage1_conv_kernel_size=self.stage1_conv_kernel_size,
            stage2_conv_kernel_size=self.stage2_conv_kernel_size,
            stage3_pool_kernel_size=self.stage3_pool_kernel_size,
            stage4_conv_channels=self.stage4_conv_channels,
        )
