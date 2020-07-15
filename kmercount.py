#!/usr/bin/env python3.7
# -*- coding: utf-8 -*-

import argparse
import sys
from collections import Counter
from typing import Dict

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

    results = Parallel(n_jobs=args.jobs, verbose=10)(
        delayed(kmercount)(row["sequence"], args.k) for _, row in sequences.iterrows()
    )

    print("*** Merging results... ***", file=sys.stderr)
    df = pd.DataFrame(
        data=results,
        index=sequences["id"],
        dtype=pd.SparseDtype(dtype="uint64", fill_value=pd.NA),
    )

    print("*** Saving data... ***", file=sys.stderr)
    df.to_csv(args.output, na_rep="0")


def kmercount(sequence: str, k: int) -> Dict[str, int]:
    if k < 1:
        raise ValueError("k must be at least 1")
    return Counter(sequence[index:index + k] for index in range(len(sequence) - k + 1))


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
