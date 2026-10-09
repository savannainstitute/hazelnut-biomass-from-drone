#!/usr/bin/env python
"""
Compare per-plant trait tables from two campaigns of the same planting.

Joins the two CSVs on an identifier, reports for every shared numeric trait
the paired count, the Pearson and Spearman correlations, the median and
inter-quartile change, and the fraction of plants that increased, and writes
a per-plant table of year-two minus year-one differences. Optional scatter
and difference figures go to --figdir.

Usage
    compare_years.py A.csv B.csv --id plant_number --label-a 2025
        --label-b 2026 --traits p99_1m vol_cellmax_ff --out deltas.csv
        --figdir figs
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


def numeric_shared(a, b, traits):
    if traits:
        return [t for t in traits if t in a.columns and t in b.columns]
    shared = [c for c in a.columns if c in b.columns]
    return [
        c
        for c in shared
        if pd.api.types.is_numeric_dtype(a[c])
        and pd.api.types.is_numeric_dtype(b[c])
    ]


def summarize(a, b, traits, la, lb):
    rows = []
    for t in traits:
        x, y = a[t].astype(float), b[t].astype(float)
        ok = x.notna() & y.notna()
        if ok.sum() < 3:
            continue
        d = y[ok] - x[ok]
        rows.append(
            {
                "trait": t,
                "n": int(ok.sum()),
                f"median_{la}": float(x[ok].median()),
                f"median_{lb}": float(y[ok].median()),
                "pearson": float(x[ok].corr(y[ok])),
                "spearman": float(x[ok].corr(y[ok], method="spearman")),
                "median_change": float(d.median()),
                "q25_change": float(d.quantile(0.25)),
                "q75_change": float(d.quantile(0.75)),
                "fraction_increased": float((d > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def figures(a, b, traits, la, lb, figdir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(figdir, exist_ok=True)
    for t in traits:
        x, y = a[t].astype(float), b[t].astype(float)
        ok = x.notna() & y.notna()
        if ok.sum() < 3:
            continue
        fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
        lim = [
            float(min(x[ok].min(), y[ok].min())),
            float(max(x[ok].max(), y[ok].max())),
        ]
        ax[0].scatter(x[ok], y[ok], s=6, alpha=0.4)
        ax[0].plot(lim, lim, color="gray", lw=1)
        ax[0].set_xlabel(f"{t} {la}")
        ax[0].set_ylabel(f"{t} {lb}")
        r = x[ok].corr(y[ok])
        ax[0].set_title(f"n = {int(ok.sum())}, r = {r:.2f}")
        d = (y[ok] - x[ok]).values
        ax[1].hist(d, bins=60, color="tab:green")
        ax[1].axvline(0, color="gray", lw=1)
        ax[1].set_xlabel(f"{t}: {lb} minus {la}")
        ax[1].set_title(f"median {np.median(d):+.3f}")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, f"{t}_{la}_vs_{lb}.png"), dpi=130)
        plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--id", required=True, help="identifier column in both")
    ap.add_argument("--label-a", default="a")
    ap.add_argument("--label-b", default="b")
    ap.add_argument("--traits", nargs="*", default=None)
    ap.add_argument(
        "--filter",
        default=None,
        help="pandas query applied to both tables first, e.g. "
        "\"lidar_status == 'alive'\"",
    )
    ap.add_argument("--out", required=True, help="per-plant differences CSV")
    ap.add_argument("--summary", default=None, help="per-trait summary CSV")
    ap.add_argument("--figdir", default=None)
    args = ap.parse_args(argv)

    a = pd.read_csv(args.a, dtype={args.id: str})
    b = pd.read_csv(args.b, dtype={args.id: str})
    if args.filter:
        a = a.query(args.filter)
        b = b.query(args.filter)
    for name, t in ((args.label_a, a), (args.label_b, b)):
        dup = t[args.id].duplicated().sum()
        if dup:
            raise SystemExit(f"{dup} duplicate {args.id} values in {name}")
    a = a.set_index(args.id)
    b = b.set_index(args.id)
    common = a.index.intersection(b.index)
    print(
        f"{len(a)} rows in {args.label_a}, {len(b)} in {args.label_b}, "
        f"{len(common)} shared on {args.id}"
    )
    a, b = a.loc[common], b.loc[common]
    traits = numeric_shared(a, b, args.traits)
    summary = summarize(a, b, traits, args.label_a, args.label_b)
    pd.set_option("display.width", 200)
    print(summary.round(3).to_string(index=False))
    if args.summary:
        summary.to_csv(args.summary, index=False)
    deltas = pd.DataFrame(index=common)
    for t in traits:
        deltas[f"{t}_{args.label_a}"] = a[t]
        deltas[f"{t}_{args.label_b}"] = b[t]
        deltas[f"{t}_change"] = b[t].astype(float) - a[t].astype(float)
    deltas.index.name = args.id
    deltas.to_csv(args.out)
    if args.figdir:
        figures(a, b, traits, args.label_a, args.label_b, args.figdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
