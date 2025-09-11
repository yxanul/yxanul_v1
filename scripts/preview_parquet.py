#!/usr/bin/env python3
import argparse
import pyarrow.parquet as pq
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', type=str, default='mixed_dataset_large.parquet')
    ap.add_argument('--rows', type=int, default=3)
    args = ap.parse_args()

    pf = pq.ParquetFile(args.path)
    print('Row groups:', pf.num_row_groups)
    print('Schema:')
    print(pf.schema)

    # Read first row group (or at least some rows)
    b = pf.read_row_group(0, columns=None)
    df = b.to_pandas().head(args.rows)
    print(f"\nFirst {args.rows} rows (truncated):")
    with pd.option_context('display.max_colwidth', 160, 'display.width', 240):
        print(df)

    # Heuristic: which columns look like text
    text_cols = [n for n in df.columns if df[n].dtype == 'object']
    print('\nCandidate text columns:', text_cols)


if __name__ == '__main__':
    main()

