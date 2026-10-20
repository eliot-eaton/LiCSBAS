#!/usr/bin/env python3
"""Remove small disconnected valid-unwrapped components from a COPY of GEOCml1.

Input/output raster convention: LiCSBAS native float32 .unw, zeros invalid.
Only .unw pixels are altered. All other files are copied unchanged.
"""
import argparse
import csv
import os
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

IFG_RE = re.compile(r'^\d{8}_\d{8}$')


def read_dims(stack):
    params = (stack / 'slc.mli.par').read_text()
    def value(key):
        match = re.search(r'^\s*' + re.escape(key) + r'\s*:\s*([^\s]+)', params, re.M)
        if not match:
            raise ValueError(f'Missing {key} in slc.mli.par')
        return int(match.group(1))
    return value('azimuth_lines'), value('range_samples')


def copy_stack(src, dst):
    if dst.exists():
        raise FileExistsError(f'Output exists: {dst}. Remove/rename it explicitly before rerunning.')
    if dst.resolve().is_relative_to(src.resolve()):
        raise ValueError('Output directory must not be inside input directory')
    shutil.copytree(src, dst, symlinks=True)


def plot_comparison(original, cleaned, removed, path, name):
    valid = np.isfinite(original) & (original != 0)
    values = original[valid]
    if values.size:
        lo, hi = np.percentile(values, [2, 98])
        if lo == hi:
            lo -= 1
            hi += 1
    else:
        lo, hi = -1, 1
    before = np.where(valid, original, np.nan)
    after = np.where(np.isfinite(cleaned) & (cleaned != 0), cleaned, np.nan)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    cmap = plt.get_cmap('RdBu_r').copy()
    cmap.set_bad('#eeeeee')
    axes[0].imshow(before, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
    axes[1].imshow(after, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
    axes[2].imshow(removed, cmap='gray_r', vmin=0, vmax=1, interpolation='nearest')
    for ax, title in zip(axes, ['Original .unw', 'Cleaned .unw', 'Removed pixels (dark)']):
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(name)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-d', '--input', required=True, type=Path, help='Original GEOCml1')
    p.add_argument('-o', '--output', required=True, type=Path, help='New GEOCml1_noislands')
    p.add_argument('--min-island-pixels', type=int, default=100, help='Remove components smaller than this (default 100)')
    p.add_argument('--connectivity', type=int, choices=(4, 8), default=8)
    p.add_argument('--png-dir', type=Path, help='PNG directory (default: sibling of output)')
    args = p.parse_args()
    src, dst = args.input.resolve(), args.output.resolve()
    if args.min_island_pixels < 1:
        p.error('--min-island-pixels must be >= 1')
    if not src.is_dir():
        p.error(f'Input directory missing: {src}')
    if src == dst:
        p.error('Input and output must be different')
    length, width = read_dims(src)
    ifgs = sorted(x.name for x in src.iterdir() if x.is_dir() and IFG_RE.fullmatch(x.name))
    if not ifgs:
        p.error('No interferogram directories found')
    # Preflight all rasters before copying any data.
    expected = length * width * 4
    for name in ifgs:
        f = src / name / f'{name}.unw'
        if not f.is_file() or f.stat().st_size != expected:
            p.error(f'Missing/incorrect float32 .unw: {f}')
    png_dir = (args.png_dir or dst.parent / f'{dst.name}_island_comparisons').resolve()
    if png_dir == src or png_dir == dst or png_dir.is_relative_to(src) or png_dir.is_relative_to(dst):
        p.error('PNG directory must be outside both input and output stacks')
    if png_dir.exists():
        p.error(f'PNG directory already exists: {png_dir}')
    copy_stack(src, dst)
    png_dir.mkdir(parents=True)
    structure = ndimage.generate_binary_structure(2, 2 if args.connectivity == 8 else 1)
    rows = []
    for j, name in enumerate(ifgs, 1):
        source_file = src / name / f'{name}.unw'
        target_file = dst / name / f'{name}.unw'
        original = np.fromfile(source_file, dtype=np.float32).reshape(length, width)
        valid = np.isfinite(original) & (original != 0)
        labels, ncomp = ndimage.label(valid, structure=structure)
        sizes = np.bincount(labels.ravel())
        drop = valid & (sizes[labels] < args.min_island_pixels)
        cleaned = original.copy()
        cleaned[drop] = 0.0
        # Replace any symlink in copied stack, never follow it to write elsewhere.
        if target_file.is_symlink():
            target_file.unlink()
        cleaned.tofile(target_file)
        plot_comparison(original, cleaned, drop, png_dir / f'{name}_before_after.png', name)
        rows.append(dict(ifg=name, valid_before=int(valid.sum()), removed_pixels=int(drop.sum()),
                         valid_after=int(valid.sum() - drop.sum()), components_before=ncomp,
                         removed_fraction_of_valid=f'{drop.sum() / max(valid.sum(), 1):.6f}'))
        print(f'[{j}/{len(ifgs)}] {name}: removed {drop.sum()} pixels ({ncomp} components)', flush=True)
    with (png_dir / 'island_removal_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nCleaned stack: {dst}\nBefore/after PNGs and CSV: {png_dir}')


if __name__ == '__main__':
    main()
