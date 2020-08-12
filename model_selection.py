#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Script for CV-based model selection and evaluation.
"""

# See https://scikit-learn.org/stable/developers/develop.html
# for more information on how this all fits together.

import argparse
import sys
from importlib import import_module
from types import ModuleType

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, cross_validate, train_test_split

from preprocessing import read_all_data


def main():
    args = parse_command_line()

    evaluation_module = import_module(f"evaluation.{args.model}")

    df = read_all_data(
        args.ss_filename,
        args.sequence_ids_filename,
        args.a_plus_deg_rates_filename,
        args.a_minus_deg_rates_filename,
    )
    df.dropna(inplace=True)

    X = evaluation_module.preprocess_data(df)
    if args.deg_model == "a+":
        y = df["log2_deg_rate_a_plus"].to_numpy(np.float32)
    else:
        y = df["log2_deg_rate_a_minus"].to_numpy(np.float32)

    args.func(args, X, y, evaluation_module)


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--deg-model",
        choices=("a+", "a-"),
        required=True,
        help="Degradation model to evaluate.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Prediction model to evaluate. See files in the 'evaluation' "
        "directory for the model definitions.",
    )
    parser.add_argument(
        "--ss-filename",
        default="data/ss_out.txt",
        help="File containing secondary structure information.",
    )
    parser.add_argument(
        "--sequence-ids-filename",
        default="data/3U_sequences_final.txt",
        help="File containing sequence IDs.",
    )
    parser.add_argument(
        "--a-plus-deg-rates-filename",
        default="data/3U.models.3U.40A.seq1022_param.txt",
        help="File containing A+ degradation rates.",
    )
    parser.add_argument(
        "--a-minus-deg-rates-filename",
        default="data/3U.models.3U.00A.seq1022_param.txt",
        help="File containing A- degradation rates.",
    )
    parser.add_argument("--epochs", type=int, default=1, help="NN training epochs.")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of jobs to run in parallel. "
        "Specify -1 to use all available CPUs.",
    )
    parser.add_argument(
        "--folds", type=int, default=3, help="Number of folds for cross-validation"
    )
    parser.add_argument(
        "--dev-ratio",
        type=float,
        default=0.9,
        help="Ratio of the total data to use for cross-validation. "
        "A float in the range (0, 1). "
        "For model evaluation, the range is (0, 1].",
    )
    parser.add_argument(
        "--dev-seed",
        type=int,
        default=None,
        help="Seed to use when randomly selecting data for use with CV",
    )

    subparsers = parser.add_subparsers(title="Sub-commands", required=True)

    select_parser = subparsers.add_parser(
        "select",
        help="Model selection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    select_parser.add_argument(
        "--iterations", type=int, default=30, help="NN parameter combinations to test."
    )

    eval_parser = subparsers.add_parser(
        "eval",
        help="Model evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    select_parser.set_defaults(func=handle_model_selection)
    eval_parser.set_defaults(func=handle_model_evaluation)

    return parser.parse_args()


def handle_model_selection(
    args: argparse.Namespace, X, y, evaluation_module: ModuleType
):
    X_dev, X_eval, y_dev, y_eval = train_test_split(
        X, y, train_size=args.dev_ratio, random_state=args.dev_seed
    )

    search = RandomizedSearchCV(
        estimator=evaluation_module.get_estimator(),
        param_distributions=evaluation_module.get_param_distributions(),
        n_iter=args.iterations,
        scoring="neg_mean_squared_error",
        n_jobs=args.jobs,
        pre_dispatch="n_jobs",
        cv=args.folds,
        refit=False,
        verbose=10,
    )
    search.fit(X_dev, y_dev, training_epochs=args.epochs)

    print(f"Best params: {search.best_params_}")

    model = (
        evaluation_module.get_estimator()
        .set_params(**search.best_params_)
        .fit(X_dev, y_dev, training_epochs=5)
    )
    mse = mean_squared_error(y_true=y_eval, y_pred=model.predict(X_eval))
    print(f"Eval MSE: {mse}")


def handle_model_evaluation(
    args: argparse.Namespace, X, y, evaluation_module: ModuleType
):
    if args.dev_ratio != 1:
        X, _, y, _ = train_test_split(
            X, y, train_size=args.dev_ratio, random_state=args.dev_seed
        )

    estimator: BaseEstimator = evaluation_module.get_estimator().set_params(
        **evaluation_module.get_eval_params()
    )

    cv_results = cross_validate(
        estimator=estimator,
        X=X,
        y=y,
        scoring=("neg_mean_squared_error", "r2"),
        cv=args.folds,
        n_jobs=args.jobs,
        pre_dispatch="n_jobs",
        verbose=10,
        fit_params={"training_epochs": args.epochs},
    )

    print(f"Neg MSE: {cv_results['test_neg_mean_squared_error']}")
    print(f"R^2: {cv_results['test_r2']}")

    print(f"Mean neg MSE: {np.mean(cv_results['test_neg_mean_squared_error'])}")
    print(f"Mean R^2: {np.mean(cv_results['test_r2'])}")


if __name__ == "__main__":
    sys.exit(main())
