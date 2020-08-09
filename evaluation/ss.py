#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from neural_net_estimators import SSDegradeEstimator
from preprocessing import one_hot_encode_sequences
from util import match_parens


def get_estimator() -> BaseEstimator:
    return SSDegradeEstimator()


def get_param_distributions() -> Dict[str, Sequence[Any]]:
    return {}


def get_eval_params() -> Dict[str, Any]:
    return {}


def preprocess_data(df: pd.DataFrame) -> np.ndarray:
    all_pairs_matrices = np.vstack(
        df["secondary_structure"].map(
            lambda struct: match_parens(struct, np.float32).flatten()
        )
    )

    sequences = one_hot_encode_sequences(df["sequence"], flat=True)

    everything = np.hstack((sequences, all_pairs_matrices))

    return everything
