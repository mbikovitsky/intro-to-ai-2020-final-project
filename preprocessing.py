#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import re
from typing import Tuple

import numpy as np
import pandas as pd


SS_PATTERN = re.compile(
    r"""^([UCAG]{110})
([.()]{110}) \( {0,2}(-?[0-9]{1,2}\.[0-9]{2})\)
([.,|(){}]{110})""",
    re.IGNORECASE | re.MULTILINE,
)


def read_secondary_structures(filename: str) -> pd.DataFrame:
    """
    Read secondary structure data into a DataFrame.

    :param filename: File to read from.

    :return: DataFrame with the following columns:
             - sequence: RNA sequence string
             - secondary_structure: Dot-bracket encoding of the secondary structure
             - free_energy
             - secondary_structure_prob: Extended dot-bracket encoding of the secondary
                                         structure
    """
    # Read file into a DataFrame

    with open(filename, mode="r") as input_file:
        raw_data = input_file.read()

    data = (match.groups() for match in SS_PATTERN.finditer(raw_data))

    data = (
        (sequence, secondary_structure, float(free_energy), secondary_structure_prob)
        for (
            sequence,
            secondary_structure,
            free_energy,
            secondary_structure_prob,
        ) in data
    )

    df = pd.DataFrame.from_records(
        data,
        columns=[
            "sequence",
            "secondary_structure",
            "free_energy",
            "secondary_structure_prob",
        ],
    )

    # Drop duplicates and transform sequences by replacing U with T

    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["sequence"] = df["sequence"].str.replace("U", "T")

    return df


def read_sequence_ids(
    filename: str, sequence_slice: slice = slice(None)
) -> pd.DataFrame:
    """
    Read a mapping from sequence string to sequence ID into a DataFrame.

    :param filename:       File to read from.
    :param sequence_slice: Slice of the sequence string to retain in the result.

    :return: DataFrame with an index of RNA sequences, and a single column with a
             sequence ID.
    """

    sequence_ids = pd.read_table(
        filename,
        header=None,
        names=["id", "sequence"],
        dtype={"id": str},
        converters={"sequence": lambda string: string[sequence_slice].upper()},
    )

    assert sequence_ids["id"].is_unique
    assert sequence_ids["sequence"].is_unique

    return sequence_ids


def read_degradation_rates(filename: str) -> pd.DataFrame:
    """
    Read RNA degradation rates.

    :param filename: File to read from.

    :return: DataFrame with the following columns:
             - id: RNA sequence ID
             - log2_deg_rate: log2 of the degradation rate
             - log2_x0
             - onset_time
    """
    return pd.read_table(
        filename,
        header=None,
        names=["id", "log2_deg_rate", "log2_x0", "onset_time"],
        dtype={"id": str},
        index_col="id",
    )


def read_all_data(
    ss_filename: str,
    sequence_ids_filename: str,
    a_plus_deg_rates_filename: str,
    a_minus_deg_rates_filename: str,
) -> pd.DataFrame:
    """
    Read all sequence data into a single DataFrame.

    :param ss_filename:                File containing secondary structure information.
    :param sequence_ids_filename:      File containing sequence IDs.
    :param a_plus_deg_rates_filename:  File containing A+ degradation rates.
    :param a_minus_deg_rates_filename: File containing A- degradation rates.
    :return:
    """
    df = read_secondary_structures(ss_filename)

    # Add ID to the main DataFrame

    sequence_ids = read_sequence_ids(sequence_ids_filename, slice(20, -20))
    sequence_ids.set_index("sequence", inplace=True)

    df = df.join(sequence_ids, on="sequence")

    assert df["id"].is_unique
    assert (
        (
            df.sort_values(by="id", axis="index").reset_index()[["sequence", "id"]]
            == sequence_ids.sort_values(by="id", axis="index").reset_index()
        )
        .all()
        .all()
    )

    df.set_index("id", inplace=True)

    # Add degradation rates to the main DataFrame

    deg_rate_a_minus = read_degradation_rates(a_minus_deg_rates_filename)
    deg_rate_a_plus = read_degradation_rates(a_plus_deg_rates_filename)
    assert (deg_rate_a_plus.index == deg_rate_a_minus.index).all()

    deg_rate_a_minus = deg_rate_a_minus.add_suffix("_a_minus")
    deg_rate_a_plus = deg_rate_a_plus.add_suffix("_a_plus")

    df = df.join([deg_rate_a_plus, deg_rate_a_minus])

    return df


def read_original_predictions(filename: str) -> Tuple[pd.DataFrame, float, float]:
    """
    Read the original degradation rate predictions.

    :param filename: File to read.

    :return: A tuple of:
             - A DataFrame with the predictions, with columns a_plus and a_minus,
               and indexed by sequence ID.
             -
    """
    df = pd.read_table(
        filename,
        header=None,
        names=["id", "a_plus", "a_minus"],
        dtype={"id": str},
        index_col="id",
    )

    a_minus_clip = df.loc["EMPTY"]["a_minus"]
    a_plus_clip = df.loc["EMPTY"]["a_plus"]
    df.drop("EMPTY", inplace=True)

    df.sort_index(inplace=True)

    return df, a_minus_clip, a_plus_clip


def one_hot_encode_sequences(
    sequences: pd.Series, drop_first: bool = False
) -> np.ndarray:
    """
    One-hot encodes a series of genetic sequences.

    :param sequences:  Sequences to encode.
    :param drop_first: Whether to encode each nucleotide as 3 dummies, instead of 4.

    :return: Array of shape Nx4xL, where N is the number of sequences and L
             is the length of each sequence. If drop_first is True, the shape is Nx3xL.
    """
    # One-hot encode the sequences
    sequences = sequences.str.split("", expand=True)
    sequences.drop(columns=[sequences.columns[0], sequences.columns[-1]], inplace=True)
    sequences = pd.get_dummies(sequences, sparse=True, drop_first=drop_first)

    columns = 3 if drop_first else 4

    # Convert to a tensor
    sequences_tensor = sequences.to_numpy()
    sequences_tensor = sequences_tensor.reshape(
        -1, sequences_tensor.shape[1] // columns, columns
    )
    sequences_tensor = sequences_tensor.transpose(0, 2, 1)
    sequences_tensor = sequences_tensor.astype(np.float32)

    return sequences_tensor
