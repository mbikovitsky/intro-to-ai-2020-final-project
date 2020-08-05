#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from neural_net_estimators import ConvDegradeEstimator
from preprocessing import one_hot_encode_sequences


def get_estimator() -> BaseEstimator:
    return ConvDegradeEstimator()


def get_param_distributions() -> Dict[str, Sequence[Any]]:
    return {}


def get_eval_params() -> Dict[str, Any]:
    return {}


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    return one_hot_encode_sequences(df["sequence"], drop_first=True)
