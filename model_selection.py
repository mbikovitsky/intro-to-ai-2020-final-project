#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import argparse
import sys
from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split

from neural_net_estimators import ResidualDegradeEstimator
from preprocessing import one_hot_encode_sequences, read_all_data


def randomized_search(
    X, y, training_epochs: int, iterations: int, jobs: int
) -> Dict[str, int]:
    param_distributions = {
        "stage1_conv_channels": range(96, 101),
        "stage1_conv_kernel_size": range(7, 10),
        "stage2_conv_kernel_size": range(3, 10, 2),
        "stage3_pool_kernel_size": range(8, 13),
        "stage4_conv_channels": range(196, 201),
    }
    search = RandomizedSearchCV(
        estimator=ResidualDegradeEstimator(),
        param_distributions=param_distributions,
        n_iter=iterations,
        scoring="neg_mean_squared_error",
        n_jobs=jobs,
        pre_dispatch=jobs,
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

    model = ResidualDegradeEstimator(**best_params).fit(X_dev, y_dev, training_epochs=5)
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
