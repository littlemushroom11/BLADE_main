import argparse
import glob
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', default=None)
    args = p.parse_args()
    files = glob.glob(os.path.join(args.root, '**', 'metrics.csv'), recursive=True)
    rows = []
    for f in files:
        df = pd.read_csv(f)
        last = df[df['stage'] == 'attack'].tail(1)
        if len(last) == 0:
            last = df.tail(1)
        row = last.iloc[0].to_dict()
        row['run_dir'] = os.path.dirname(f)
        rows.append(row)
    out = pd.DataFrame(rows)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
