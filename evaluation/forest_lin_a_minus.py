#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Model selection/evaluation parameters for the random forest,
using the A- linear model features.
"""

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor

from linear_regression import Model
from preprocessing import read_sequence_ids


def get_estimator() -> BaseEstimator:
    """
    Returns an instance of an sklearn estimator.
    """
    return RandomForestRegressor()


def get_param_distributions() -> Dict[str, Sequence[Any]]:
    """
    Returns a dict of parameter distributions for use with RandomizedSearchCV.
    """
    return {
        "n_estimators": range(40, 120),
        "max_depth": range(50, 100),
        "min_samples_leaf": range(5, 20),
        "bootstrap": (True,),
    }


def get_eval_params() -> Dict[str, Any]:
    """
    Returns a dict of estimator parameters for model evaluation.
    """
    return {
        "n_estimators": 80,
        "max_depth": 60,
        "min_samples_leaf": 15,
        "bootstrap": True,
    }


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame returned by preprocessing.read_all_data, returns a tensor for use
    as input to the sklearn estimator.
    """
    # The DataFrame we were given contains truncated sequences,
    # but we need the full ones
    complete_sequences = read_sequence_ids("data/3U_sequences_final.txt")
    complete_sequences.set_index("id", inplace=True)

    assert frozenset(df.index).issubset(frozenset(complete_sequences.index))

    # Drop sequences that are not present in the given DataFrame
    complete_sequences.drop(
        index=complete_sequences.index.difference(df.index), inplace=True
    )

    # Make sure both DFs are in the same order
    complete_sequences = complete_sequences.reindex(index=df.index)

    assert (complete_sequences.index == df.index).all()

    linear_model = Model.load("data/run_linear_3U_00Am1_dg_BEST.out.mat")
    return linear_model.kmer_count_matrix(complete_sequences["sequence"])
