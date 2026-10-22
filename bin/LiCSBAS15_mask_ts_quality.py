#!/usr/bin/env python3
"""LiCSBAS Step 15 replacement with relative-displacement quality mask.

Example:
 python LiCSBAS15_mask_ts_quality.py -t TS_GEOCml1 \
   --quality-ref-pixel 450,320 --quality-threshold 0.3

Q = coherence * (abs(relative net displacement) + D0) /
                (abs(relative net displacement) + D0 + residual RMS).
All displacement/RMS quantities are in mm. D0 defaults to 2 mm.
This is a screening heuristic, not an estimated probability or uncertainty.
"""
import argparse
import math
import os
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def read_parameters(path):
    out = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 2 and not fields[0].startswith('#'):
            out[fields[0].rstrip(':')] = fields[1]
    return out


def read_raster(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    if data.size != int(np.prod(shape)):
        raise ValueError(f'{path}: expected {np.prod(shape)} float32 values, got {data.size}')
    return data.reshape(shape)


def parse_pixel(value):
    try:
        x, y = (int(v) for v in value.split(','))
        return x, y
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError('Expected X,Y, e.g. 450,320')


def save_image(array, filename, title, cmap='viridis', lo=None, hi=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(array, cmap=cmap, vmin=lo, vmax=hi, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('X pixel'); ax.set_ylabel('Y pixel')
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('-t', required=True, help='TS_GEOCml* directory')
    p.add_argument('--quality-ref-pixel', required=True, type=parse_pixel,
                   metavar='X,Y', help='Reference column,row (zero based)')
    p.add_argument('--quality-threshold', type=float, default=0.3)
    p.add_argument('--quality-d0', type=float, default=2.0, help='Baseline displacement in mm')
    p.add_argument('--quality-only', action='store_true', help='Apply only composite quality criterion (plus valid-data mask)')
    p.add_argument('-c', type=float, help='Minimum average coherence')
    p.add_argument('-u', type=float, help='Minimum n_unw / number of acquisitions')
    p.add_argument('-v', type=float, help='Maximum velocity standard deviation (mm/year)')
    p.add_argument('-T', type=float, help='Minimum maxTlen (years)')
    p.add_argument('-g', type=float, help='Maximum network gaps')
    p.add_argument('-s', type=float, help='Maximum STC (mm)')
    p.add_argument('-i', type=float, help='Maximum no-loop IFG count')
    p.add_argument('-l', type=float, help='Maximum loop-error count')
    p.add_argument('-L', type=float, help='Maximum loop-error ratio')
    p.add_argument('-r', type=float, help='Maximum residual RMS (mm)')
    p.add_argument('--avg_phase_bias', type=float, help='Maximum average loop phase bias (rad)')
    p.add_argument('--n_gap_use_merged', action='store_true')
    p.add_argument('--keep_isolated', action='store_true')
    p.add_argument('--noautoadjust', action='store_true', help='Accepted for compatibility; thresholds are never changed silently')
    p.add_argument('--vmin', type=float); p.add_argument('--vmax', type=float)
    p.add_argument('--sbovl', action='store_true', help='Use SBOI/SBOVL default thresholds')
    p.add_argument('--sbovl_abs', action='store_true', help='Use available bootvel_abs velocity if present')
    args = p.parse_args()

    if not (0 <= args.quality_threshold <= 1):
        p.error('--quality-threshold must be between 0 and 1')
    if args.quality_d0 <= 0:
        p.error('--quality-d0 must be > 0')
    tsa = Path(args.t).resolve()
    results = tsa / 'results'
    info = tsa / 'info'
    parms = info / '13parameters.txt'
    if not parms.is_file():
        parms = info / 'parameters.txt'
    if not parms.is_file() or not results.is_dir():
        p.error('Missing results/ or info/13parameters.txt (or info/parameters.txt)')
    pp = read_parameters(parms)
    shape = (int(pp['azimuth_lines']), int(pp['range_samples']))
    length, width = shape
    n_im = int(pp['n_im'])
    wavelength = float(pp['wavelength'])
    x, y = args.quality_ref_pixel
    if not (0 <= x < width and 0 <= y < length):
        p.error(f'Reference pixel {x},{y} outside raster {width}x{length}')

    vel_path = results / 'vel'
    if args.sbovl_abs:
        trial = results / 'bootvel_abs'
        if trial.exists():
            vel_path = trial
    vel = read_raster(vel_path, shape)
    coh = read_raster(results / 'coh_avg', shape)
    rms = read_raster(results / 'resid_rms', shape)

    cum_path = tsa / 'cum.h5'
    if not cum_path.exists():
        p.error(f'{cum_path} not found; Step 13 cumulative displacement is needed')
    with h5py.File(cum_path, 'r') as h5:
        if 'cum' not in h5 or len(h5['cum'].shape) != 3:
            raise ValueError('cum.h5 needs a 3D dataset named cum')
        cube = h5['cum']
        if cube.shape[0] < 2 or cube.shape[1:] != shape:
            raise ValueError(f'cum shape {cube.shape} inconsistent with {shape}')
        first = np.asarray(cube[0], dtype=np.float32)
        last = np.asarray(cube[-1], dtype=np.float32)
        if 'imdates' in h5:
            imdates = h5['imdates'][:]
            print(f'Cumulative displacement interval: {imdates[0]} to {imdates[-1]}')

    # Taking the difference before spatial referencing preserves the common time origin.
    net = last - first
    ref_net = float(net[y, x])
    if not np.isfinite(ref_net):
        p.error('Reference pixel lacks a valid displacement at first or last epoch')
    drel = net - ref_net
    dabs = np.abs(drel)
    with np.errstate(invalid='ignore', divide='ignore'):
        quality = coh * (dabs + args.quality_d0) / (dabs + args.quality_d0 + rms)
    valid_quality = (np.isfinite(coh) & (coh >= 0) & (coh <= 1) &
                     np.isfinite(rms) & (rms >= 0) & np.isfinite(drel))
    quality[~valid_quality] = np.nan

    # The original Step 15 uses finite nonzero velocity pixels as candidates,
    # then explicitly restores the velocity-zero reference pixel.
    valid_vel = np.isfinite(vel)
    candidates = valid_vel & (vel != 0)
    good = candidates & np.isfinite(quality) & (quality <= 1) & (quality >= args.quality_threshold)
    diagnostics = [('quality_score', quality, args.quality_threshold, '>=')]
    applied = [('quality_score', args.quality_threshold, '>=')]

    # Match the original C-band/L-band defaults, but never auto-relax a cut.
    cband = wavelength < 0.2
    sb = args.sbovl or args.sbovl_abs
    gaps_name = 'n_gap_merged' if args.n_gap_use_merged else 'n_gap'
    default_noloop = 500
    no_loop_dir = tsa / '12no_loop_ifg_ras'
    if no_loop_dir.is_dir():
        default_noloop = len(list(no_loop_dir.iterdir())) + 1
    defaults = {
        'coh_avg': (args.c if args.c is not None else (0.05 if sb else 0.15 if cband else 0.01), '>='),
        'n_unw': (int(n_im * (args.u if args.u is not None else 0.5 if sb else 1.5 if cband else 1.0)), '>='),
        'vstd': (args.v if args.v is not None else (100 if cband else 200), '<='),
        'maxTlen': (args.T if args.T is not None else 1, '>='),
        gaps_name: (args.g if args.g is not None else (50 if sb else 10 if cband else 1), '<='),
        'stc': (args.s if args.s is not None else (10 if sb else 10 if cband else 10), '<='),
        'n_ifg_noloop': (args.i if args.i is not None else (1000 if sb else default_noloop), '<='),
        'resid_rms': (args.r if args.r is not None else (15 if sb else 15 if cband else 10), '<='),
    }
    if not sb:
        if (results / 'n_loop_err_rat').exists():
            defaults['n_loop_err_rat'] = (args.L if args.L is not None else 0.7, '<=')
        elif (results / 'n_loop_err').exists():
            defaults['n_loop_err'] = (args.l if args.l is not None else (5 if cband else 1), '<=')
        else:
            p.error('No n_loop_err or n_loop_err_rat raster; rerun Step 12')
    if args.l is not None and 'n_loop_err' not in defaults:
        defaults['n_loop_err'] = (args.l, '<=')
    if args.L is not None and 'n_loop_err_rat' not in defaults and not sb:
        defaults['n_loop_err_rat'] = (args.L, '<=')
    if args.avg_phase_bias is not None:
        defaults['loop_ph_avg_abs'] = (args.avg_phase_bias, '<=')

    if not args.quality_only:
        for name, (threshold, op) in defaults.items():
            array = read_raster(results / name, shape)
            if name == 'stc' and args.keep_isolated:
                array = np.where(np.isnan(array), 0, array)
            passed = np.isfinite(array) & ((array >= threshold) if op == '>=' else (array <= threshold))
            good &= passed
            diagnostics.append((name, array, threshold, op))
            applied.append((name, threshold, op))

    # Preserve the original zero-velocity reference locations (standard Step 15).
    mask = np.full(shape, np.nan, dtype=np.float32)
    mask[candidates] = good[candidates].astype(np.float32)
    mask[valid_vel & (vel == 0)] = 1.0
    masked_vel = vel.copy()
    masked_vel[mask != 1] = np.nan
    mask.tofile(results / 'mask')
    masked_vel.astype(np.float32).tofile(str(vel_path) + '.mskd')
    quality.astype(np.float32).tofile(results / 'quality_score')
    drel.astype(np.float32).tofile(results / 'displacement_relative')

    save_image(mask, results / 'mask.png', 'Step 15 mask (1=kept)', lo=0, hi=1)
    save_image(masked_vel, Path(str(vel_path) + '.mskd.png'), 'Masked velocity (mm/year)', cmap='RdBu_r', lo=args.vmin, hi=args.vmax)
    save_image(quality, results / 'quality_score.png', 'Quality score (higher is better)', lo=0, hi=0.5)
    save_image(drel, results / 'displacement_relative.png', 'Relative net cumulative displacement (mm)', cmap='RdBu_r')

    # Diagnostics: original, masked velocity, mask, quality + all existing indices.
    panels = [('vel.mskd', masked_vel), ('vel', vel), ('mask', mask)] + [(n, a) for n, a, _, _ in diagnostics]
    ncols = 5
    nrows = math.ceil(len(panels) / ncols)
    for filename, masked in [('mask_ts.png', False), ('mask_ts_masked.png', True)]:
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.9), squeeze=False)
        for ax, (name, arr) in zip(axes.flat, panels):
            display = arr
            if masked and name not in ('vel.mskd', 'vel', 'mask'):
                display = np.where(mask == 1, arr, np.nan)
            cmap = 'RdBu_r' if name.startswith('vel') else 'viridis'
            im = ax.imshow(display, cmap=cmap, interpolation='nearest',
                           vmin=0 if name in ('mask','quality_score') else None,
                           vmax=0.5 if name in ('mask','quality_score') else None)
            ax.set_title(name, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        for ax in axes.flat[len(panels):]:
            ax.axis('off')
        fig.tight_layout()
        fig.savefig(tsa / filename, dpi=120)
        plt.close(fig)

    eligible = int(np.count_nonzero(candidates))
    kept = int(np.count_nonzero(good))
    with (info / '15parameters.txt').open('w') as out:
        out.write('LiCSBAS15_mask_ts_quality.py\n')
        out.write(f'Reference pixel (x,y): {x},{y}\nReference net displacement: {ref_net:.5f} mm\n')
        out.write(f'Quality D0: {args.quality_d0} mm\n')
        for name, threshold, op in applied:
            out.write(f'{name}: {op} {threshold}\n')
        out.write(f'Kept candidate pixels: {kept}/{eligible}\n')
    print(f'Reference pixel (x,y): {x},{y}; reference net displacement: {ref_net:.3f} mm')
    print(f'Candidate pixels kept: {kept}/{eligible} ({100 * kept / eligible if eligible else 0:.1f}%)')
    print(f'Wrote {results / "mask"}, {vel_path}.mskd, quality_score, displacement_relative')
    print(f'Diagnostics: {tsa / "mask_ts.png"}')


if __name__ == '__main__':
    main()
