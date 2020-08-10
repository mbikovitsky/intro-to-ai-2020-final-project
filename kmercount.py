#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import argparse
import sys
from collections import Counter
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from util import find_unsigned_integer_dtype


def main():
    args = parse_command_line()

    sequences = pd.read_table(
        args.sequences,
        header=None,
        names=["id", "sequence"],
        dtype={"id": str},
        converters={"sequence": lambda string: string.upper()},
    )

    df = aggregate_kmercount(sequences, args.k, args.jobs, verbose=True)

    print("*** Saving data... ***", file=sys.stderr)
    df.to_csv(args.output, na_rep="0")


def aggregate_kmercount(
    sequences: pd.DataFrame, k: int, jobs: int = -1, verbose: bool = False
) -> pd.DataFrame:
    """
    Counts the occurrences of each possible k-mer in a list of genetic sequence.

    :param sequences: DataFrame with an 'id' column and a 'sequence' column.
    :param k:         Length of k-mers to count.
    :param jobs:      Number of parallel jobs
    :param verbose:   Whether to output progress messages to stderr.

    :return: DataFrame, indexed by sequence id, and with columns corresponding to
             k-mers encountered. Each cell is the k-mer count.
    """
    results: Sequence[Dict[str, int]] = Parallel(
        n_jobs=jobs, verbose=10 if verbose else 0
    )(delayed(kmercount)(sequence, k) for sequence in sequences["sequence"])

    if verbose:
        print("*** Merging results... ***", file=sys.stderr)

    total_kmers = 0  # Total number of unique kmers in the given sequences
    kmer_columns = {}  # Column corresponding to each kmer
    max_count = 0  # Maximum count of a some kmer in any of the sequences
    for result in results:
        max_count = max(max_count, max(result.values()))
        for kmer in result.keys():
            if kmer not in kmer_columns:
                kmer_columns[kmer] = total_kmers
                total_kmers += 1
    assert total_kmers == len(kmer_columns)

    dtype = find_unsigned_integer_dtype(max_count)

    matrix = np.zeros((sequences.shape[0], total_kmers), dtype=dtype)

    for index, result in enumerate(results):
        for kmer, count in result.items():
            matrix[index, kmer_columns[kmer]] = count

    df = pd.DataFrame(
        data=matrix,
        index=sequences["id"],
        columns=sorted(kmer_columns.keys(), key=lambda kmer: kmer_columns[kmer]),
    )

    return df


def kmercount(sequence: str, k: int) -> Dict[str, int]:
    """
    Counts the occurrences of each possible k-mer in a genetic sequence.

    Note that the function matches k-mers *case-sensitively*.

    :param sequence: Sequence to count k-mers in.
    :param k:        Length of k-mers to count.

    :return: Dict mapping k-mer to its number of occurrences in the sequence.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    return Counter(
        sequence[index : index + k] for index in range(len(sequence) - k + 1)
    )


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("sequences", help="File containing the sequences to process")
    parser.add_argument("output", help="Output filename (CSV)")
    parser.add_argument("k", type=int, help="k-mer length")
    parser.add_argument(
        "-j", "--jobs", type=int, default=-1, help="Number of parallel jobs"
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
