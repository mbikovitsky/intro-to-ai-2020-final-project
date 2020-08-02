#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

from typing import NamedTuple, List

import numpy as np
import pandas as pd
from scipy.io import loadmat

from util import occurrences


class Model(NamedTuple):
    coefficients: np.ndarray
    intercept: np.float
    kmers: List[str]

    def predict(self, sequences: pd.Series) -> np.ndarray:
        X = np.zeros((len(sequences), len(self.coefficients)), dtype=int)

        # Count the occurrences of each feature k-mer in each of the sequences
        # we are predicting.
        for column, kmer in enumerate(self.kmers):
            X[:, column] = occurrences(sequences, kmer)

        prediction = X @ self.coefficients + self.intercept

        return prediction

    @staticmethod
    def load(filename: str) -> "Model":
        raw_model = loadmat(filename)
        kmers = [element[0][0] for element in raw_model["Krows"]]

        return Model(raw_model["B0"], raw_model["b0"][0][0], kmers)
