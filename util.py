#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import re
from typing import Union, Tuple, Optional

import numpy as np
import pandas as pd


def occurrences(string: Union[str, pd.Series, pd.Index], sub: str) -> np.ndarray:
    """
    Count the overlapping occurrences of a string in another string.
    """
    if isinstance(string, str):
        string = pd.Series([string])
    # https://stackoverflow.com/a/11706065/851560
    return string.str.count(f"(?={re.escape(sub)})").to_numpy()


def match_parens(string: str) -> np.ndarray:
    """
    Returns a matrix of matching parentheses. For each pair of indices i, j
    in the input string, the cell (i, j) in the matrix will have a value of 1
    iff i and j contain a matching pair of parens.
    """

    pairs_matrix = np.zeros((len(string), len(string)), dtype=np.uint8)

    stack = []
    for index, char in enumerate(string):
        if char == "(":
            stack.append(index)
        elif char == ")":
            open_index = stack.pop()
            pairs_matrix[open_index, index] = 1
            pairs_matrix[index, open_index] = 1
    assert not stack

    return pairs_matrix


def mse(
    a: np.ndarray, b: np.ndarray, axis: Optional[Union[int, Tuple[int, ...]]] = None
) -> Union[np.ndarray, float]:
    """
    Calculates the MSE from the given arrays.

    :param a:    First array.
    :param b:    Second array.
    :param axis: Axis the calculate the mean against. See numpy.mean for the description
                 of this parameter.

    :return: The mean squared error.
    """
    return ((a - b) ** 2).mean(axis=axis)
