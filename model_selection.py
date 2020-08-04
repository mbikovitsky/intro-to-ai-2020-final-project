#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import argparse
import sys
from typing import Dict

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from torch import from_numpy, nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from neural_net import Net
from preprocessing import one_hot_encode_sequences, read_all_data
from util import train_network


class NNEstimator(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        stage1_conv_channels: int = 96,
        stage1_conv_kernel_size: int = 12,
        stage2_conv_kernel_size: int = 5,
        stage3_pool_kernel_size: int = 10,
        stage4_conv_channels: int = 196,
    ):
        self.stage1_conv_channels = stage1_conv_channels
        self.stage1_conv_kernel_size = stage1_conv_kernel_size
        self.stage2_conv_kernel_size = stage2_conv_kernel_size
        self.stage3_pool_kernel_size = stage3_pool_kernel_size
        self.stage4_conv_channels = stage4_conv_channels

    def fit(self, X, y, training_epochs: int = 1) -> "NNEstimator":
        X, y = check_X_y(X, y, ensure_2d=False, allow_nd=True, y_numeric=True)
        X = from_numpy(X)
        y = from_numpy(y.reshape(-1, 1))

        network = Net(
            self.stage1_conv_channels,
            self.stage1_conv_kernel_size,
            self.stage2_conv_kernel_size,
            self.stage3_pool_kernel_size,
            self.stage4_conv_channels,
        )

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
        X = from_numpy(X)

        with torch.no_grad():
            return self._network(X).numpy().reshape(-1)


def randomized_search(
    X, y, training_epochs: int, iterations: int, jobs: int
) -> Dict[str, int]:
    param_distributions = {
        "stage1_conv_channels": range(96, 101),
        "stage1_conv_kernel_size": range(7, 18),
        "stage2_conv_kernel_size": range(3, 10, 2),
        "stage3_pool_kernel_size": range(8, 13),
        "stage4_conv_channels": range(196, 201),
    }
    search = RandomizedSearchCV(
        estimator=NNEstimator(),
        param_distributions=param_distributions,
        n_iter=iterations,
        scoring="neg_mean_squared_error",
        n_jobs=jobs,
        cv=3,
        refit=False,
        verbose=10,
    )

    search.fit(X, y, training_epochs=training_epochs)

    return search.best_params_


def main():
    args = parse_command_line()

    df = read_all_data(
        args.ss_filename,
        args.sequence_ids_filename,
        args.a_plus_deg_rates_filename,
        args.a_minus_deg_rates_filename,
    )
    df.dropna(inplace=True)

    X = one_hot_encode_sequences(df["sequence"])
    if args.model == "a+":
        y = df["log2_deg_rate_a_plus"].to_numpy().astype(np.float32)
    else:
        y = df["log2_deg_rate_a_minus"].to_numpy().astype(np.float32)

    X_dev, X_eval, y_dev, y_eval = train_test_split(X, y, train_size=0.9)

    best_params = randomized_search(
        X_dev,
        y_dev,
        training_epochs=args.epochs,
        iterations=args.iterations,
        jobs=args.jobs,
    )
    print(f"Best params: {best_params}")

    model = NNEstimator(**best_params).fit(X_dev, y_dev, training_epochs=5)
    mse = mean_squared_error(y_true=y_eval, y_pred=model.predict(X_eval))
    print(f"Eval MSE: {mse}")


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "ss_filename", help="File containing secondary structure information."
    )
    parser.add_argument("sequence_ids_filename", help="File containing sequence IDs.")
    parser.add_argument(
        "a_plus_deg_rates_filename", help="File containing A+ degradation rates."
    )
    parser.add_argument(
        "a_minus_deg_rates_filename", help="File containing A- degradation rates."
    )
    parser.add_argument(
        "model", choices=("a+", "a-"), help="Degradation model to evaluate."
    )
    parser.add_argument("--epochs", type=int, default=1, help="NN training epochs.")
    parser.add_argument(
        "--iterations", type=int, default=30, help="NN parameter combinations to test."
    )
    parser.add_argument(
        "--jobs", type=int, default=1, help="Number of jobs to run in parallel."
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
