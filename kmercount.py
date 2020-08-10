#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import argparse
import sys
from collections import Counter
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def main():
    args = parse_command_line()

    sequences = pd.read_table(
        args.sequences,
        header=None,
        names=["id", "sequence"],
        dtype={"id": str},
        converters={"sequence": lambda string: string.upper()},
    )

    results: Sequence[Dict[str, int]] = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(kmercount)(row["sequence"], args.k) for _, row in sequences.iterrows()
    )

    print("*** Merging results... ***", file=sys.stderr)

    key_count = 0
    key_columns = {}
    max_count = 0
    for result in results:
        max_count = max(max_count, max(result.values()))
        for key in result.keys():
            if key not in key_columns:
                key_columns[key] = key_count
                key_count += 1

    dtype = find_unsigned_integer_dtype(max_count)

    matrix = np.zeros((sequences.shape[0], key_count), dtype=dtype)

    for index, result in enumerate(results):
        for kmer, count in result.items():
            matrix[index, key_columns[kmer]] = count

    df = pd.DataFrame(
        data=matrix,
        index=sequences["id"],
        columns=sorted(key_columns.keys(), key=lambda key: key_columns[key]),
    )

    print("*** Saving data... ***", file=sys.stderr)
    df.to_csv(args.output, na_rep="0")


def kmercount(sequence: str, k: int) -> Dict[str, int]:
    if k < 1:
        raise ValueError("k must be at least 1")
    return Counter(
        sequence[index : index + k] for index in range(len(sequence) - k + 1)
    )


def find_unsigned_integer_dtype(maximum_value: int) -> np.dtype:
    if maximum_value > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("Maximum value cannot be represented")

    if maximum_value < 0:
        raise ValueError("Maximum value must be unsigned")

    if maximum_value == 0:
        return np.uint8

    required_bits = int(np.ceil(np.log2(maximum_value)))

    dtypes = (
        (8, np.uint8),
        (16, np.uint16),
        (32, np.uint32),
        (64, np.uint64),
    )

    for dtype_bits, dtype in dtypes:
        if required_bits <= dtype_bits:
            return dtype


def parse_command_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("sequences", help="File containing the sequences to process")
    parser.add_argument("output", help="Output filename (CSV)")
    parser.add_argument("k", type=int, help="k-mer length")
    parser.add_argument(
        "-j", "--jobs", type=int, default=-1, help="Number of parallel jobs"
    )

    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
