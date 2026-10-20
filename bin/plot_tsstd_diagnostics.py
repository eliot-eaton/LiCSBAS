#!/usr/bin/env python3
"""Temporal, distributional, and spatial diagnostics for LiCSBAS tsstd."""
import argparse
import glob
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='TS_new_GEOCml1/cum.h5', help='Input LiCSBAS cum.h5')
    parser.add_argument('--output-dir', default='.', help='Directory for output PNGs')
    parser.add_argument('--matrix-glob', default='tsstd_matrix_diagnostics_*.csv', help='Optional inversion matrix diagnostics CSV glob')
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, 'r') as f:
        tsstd = f['tsstd'][:]
        raw_dates = f['imdates'][:]

    dates = [datetime.strptime(d.decode() if isinstance(d, bytes) else str(d), '%Y%m%d') for d in raw_dates[1:]]
    if tsstd.shape[0] != len(dates):
        raise ValueError(f'tsstd has {tsstd.shape[0]} epochs, but imdates[1:] has {len(dates)}')

    median, p05, p95, pct_high = [], [], [], []
    for layer in tsstd:
        valid = layer[np.isfinite(layer)]
        if valid.size:
            median.append(float(np.median(valid)))
            p05.append(float(np.percentile(valid, 5)))
            p95.append(float(np.percentile(valid, 95)))
            pct_high.append(float(100 * np.mean(valid > 100)))
        else:
            median.append(np.nan)
            p05.append(np.nan)
            p95.append(np.nan)
            pct_high.append(np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(11, 11))
    ax = axes[0]
    ax.plot(dates, median, 'o-', label='Median')
    ax.plot(dates, p05, 'o--', label='5th percentile')
    ax.plot(dates, p95, 'o-', label='95th percentile')
    ax.set_yscale('log')
    ax.set_ylabel('1σ uncertainty (mm)')
    ax.set_title('LiCSBAS WLS cumulative displacement uncertainty')
    ax.grid(alpha=0.3)
    ax.legend()

    valid_all = tsstd[np.isfinite(tsstd) & (tsstd > 0)]
    if valid_all.size:
        axes[1].hist(np.log10(valid_all), bins=100)
    axes[1].set_xlabel('log10(1σ uncertainty / mm)')
    axes[1].set_ylabel('Number of values')
    axes[1].set_title('Distribution of uncertainty')
    axes[1].grid(alpha=0.3)

    axes[2].plot(dates, pct_high, 'o-')
    axes[2].set_ylabel('Pixels with σ > 100 mm (%)')
    axes[2].set_xlabel('Acquisition date')
    axes[2].set_title('Spatial prevalence of extreme uncertainty')
    axes[2].grid(alpha=0.3)
    for ax in (axes[0], axes[2]):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.tick_params(axis='x', rotation=45)
    fig.tight_layout()
    path1 = out / 'tsstd_diagnostics.png'
    fig.savefig(path1, dpi=200)
    plt.close(fig)

    # Spatial maps: earliest, highest-median, and latest epoch.
    medians = np.asarray(median)
    if not np.isfinite(medians).any():
        raise ValueError('No finite tsstd values available for spatial maps')
    indices = [0, int(np.nanargmax(medians)), tsstd.shape[0] - 1]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    for ax, k in zip(axes, indices):
        data = np.ma.masked_invalid(tsstd[k])
        data = np.ma.masked_less_equal(data, 0)
        im = ax.imshow(data, cmap='magma', norm=LogNorm(vmin=0.1, vmax=1e5), interpolation='nearest')
        ax.set_title(f'{dates[k]:%Y%m%d}\nMedian = {medians[k]:.1f} mm')
        ax.set_xlabel('X pixel')
        ax.set_ylabel('Y pixel')
    fig.colorbar(im, ax=axes, shrink=0.8, label='1σ displacement uncertainty (mm)')
    path2 = out / 'tsstd_spatial_comparison.png'
    fig.savefig(path2, dpi=200)
    plt.close(fig)
    # Optional: compare singular values and residual variance for sampled pixels.
    files = sorted(glob.glob(args.matrix_glob))
    if files:
        arrays = []
        for file in files:
            data = np.genfromtxt(file, delimiter=',', names=True, ndmin=1)
            if data.size:
                arrays.append(data)
        if arrays:
            diag = np.concatenate(arrays)
            good = np.isfinite(diag['max_tsstd_mm']) & (diag['max_tsstd_mm'] > 0)
            diag = diag[good]
            if len(diag):
                fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
                for ax, field, title in zip(
                    axes,
                    ['sigma0_sq', 'condition', 'max_cx_diag'],
                    ['Weighted residual variance factor', 'Design-matrix condition number', 'Largest parameter variance']
                ):
                    x = diag['max_tsstd_mm']
                    y = diag[field]
                    ok = (np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0))
                    ax.scatter(x[ok], y[ok], s=15, alpha=0.55)
                    ax.set_xscale('log')
                    ax.set_yscale('log')
                    ax.set_xlabel('Maximum pixel tsstd (mm)')
                    ax.set_title(title)
                    ax.grid(alpha=0.3)
                path3 = out / 'tsstd_matrix_diagnostics.png'
                fig.savefig(path3, dpi=200)
                plt.close(fig)
                print(f'Saved {path3} from {len(diag)} sampled pixels')
                for label, selector in [
                    ('Ordinary uncertainty (2-10 mm)', (diag['max_tsstd_mm'] >= 2) & (diag['max_tsstd_mm'] <= 10)),
                    ('Extreme uncertainty (>1000 mm)', diag['max_tsstd_mm'] > 1000),
                ]:
                    group = diag[selector]
                    if len(group):
                        print(f"{label}: n={len(group)}, median sigma0_sq={np.median(group['sigma0_sq']):.4g}, median condition={np.median(group['condition']):.4g}, median rank={np.median(group['rank']):.0f}/{np.median(group['n_param']):.0f}")
    else:
        print('No matrix diagnostic CSVs found; matrix comparison skipped.')
        print('To produce them, instrument wls_nsbas_with_error() and rerun inversion.')

    print(f'Saved {path1}')
    print(f'Saved {path2}')
    print('Spatial map dates:', ', '.join(dates[k].strftime('%Y%m%d') for k in indices))


if __name__ == '__main__':
    main()
