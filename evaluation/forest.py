#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Model selection/evaluation parameters for the random forest.
"""

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from sklearn.ensemble import RandomForestRegressor
from preprocessing import one_hot_encode_sequences
from linear_regression import Model


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
        # n_estimators=100, random_state = None, bootstrap = True, max_depth = 70, min_samples_leaf = 10
        "n_estimators": range(40, 120),
        "max_depth": range(50, 100),
        "min_samples_leaf": range(20, 5),
    }


def get_eval_params() -> Dict[str, Any]:
    """
    Returns a dict of estimator parameters for model evaluation.
    """
    return {
        "n_estimators": 80,
        "max_depth": 60,
        "min_samples_leaf": 15,
        "random_state": None,
        "bootstrap": True,
    }


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame returned by preprocessing.read_all_data, returns a tensor for use
    as input to the sklearn estimator.
    """
    model_a_plus = Model.load("data/run_linear_3U_40A_dg_BEST.out.mat")
    model_a_minus = Model.load("data/run_linear_3U_00Am1_dg_BEST.out.mat")

    kmer_cnt_matrix_a_plus = model_a_plus.kmer_cnt_matrix(df["sequence"])
    kmer_cnt_matrix_a_minus = model_a_minus.kmer_cnt_matrix(df["sequence"])

    # Add chosing between a_plus or a_minus to send out.
    return kmer_cnt_matrix_a_plus
