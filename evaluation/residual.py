#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-
"""
Model selection/evaluation parameters for the ResidualDegrade neural network.
"""

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from neural_net_estimators import ResidualDegradeEstimator
from preprocessing import one_hot_encode_sequences


def get_estimator() -> BaseEstimator:
    """
    Returns an instance of an sklearn estimator.
    """
    return ResidualDegradeEstimator()


def get_param_distributions() -> Dict[str, Sequence[Any]]:
    """
    Returns a dict of parameter distributions for use with RandomizedSearchCV.
    """
    return {
        "stage1_conv_channels": range(96, 101),
        "stage1_conv_kernel_size": range(7, 10),
        "stage2_conv_kernel_size": range(3, 10, 2),
        "stage3_pool_kernel_size": range(8, 13),
        "stage4_conv_channels": range(196, 201),
    }


def get_eval_params() -> Dict[str, Any]:
    """
    Returns a dict of estimator parameters for model evaluation.
    """
    return {
        "stage1_conv_channels": 97,
        "stage1_conv_kernel_size": 7,
        "stage2_conv_kernel_size": 3,
        "stage3_pool_kernel_size": 8,
        "stage4_conv_channels": 198,
    }


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame returned by preprocessing.read_all_data, returns a tensor for use
    as input to the sklearn estimator.
    """
    return one_hot_encode_sequences(df["sequence"])
