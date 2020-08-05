#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import argparse
import sys
from importlib import import_module

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score

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

    estimator: BaseEstimator = evaluation_module.get_estimator()

    scores = cross_val_score(
        estimator=estimator.set_params(**evaluation_module.get_eval_params()),
        X=X,
        y=y,
        scoring="neg_mean_squared_error",
        cv=args.folds,
        n_jobs=args.jobs,
        pre_dispatch=args.jobs,
        verbose=10,
        fit_params={"training_epochs": args.epochs},
    )

    print(f"Scores: {scores}")
    print(f"Mean score: {np.mean(scores)}")


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
        "deg_model", choices=("a+", "a-"), help="Degradation model to evaluate."
    )
    parser.add_argument("model", help="Prediction model to evaluate")
    parser.add_argument("--epochs", type=int, default=1, help="NN training epochs.")
    parser.add_argument(
        "--jobs", type=int, default=1, help="Number of jobs to run in parallel."
    )
    parser.add_argument(
        "--folds", type=int, default=3, help="Number of folds for cross-validation"
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
