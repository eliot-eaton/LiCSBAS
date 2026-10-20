#!/usr/bin/env python3
"""Recover Step-11 rejected IFGs based on target-area coherence and valid .unw coverage.

Step 12 reads info/11bad_ifg.txt. This script updates that list after backing
it up, and regenerates info/11removed_image.txt for consistency.
"""
import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

IFG_RE = re.compile(r'^\d{8}_\d{8}$')


def param(path, key, cast=float):
    txt = path.read_text()
    m = re.search(r'^\s*' + re.escape(key) + r'\s*:\s*([^\s]+)', txt, re.M)
    if not m:
        raise ValueError(f'Missing {key} in {path}')
    return cast(m.group(1))


def read_ifgs(path):
    return [x.strip().split()[0] for x in path.read_text().splitlines()
            if x.strip() and not x.lstrip().startswith('#')]


def target_slice(dempar, length, width, coords):
    lat1, lon1, lat2, lon2 = coords
    lat0 = param(dempar, 'corner_lat')
    lon0 = param(dempar, 'corner_lon')
    dlat = param(dempar, 'post_lat')
    dlon = param(dempar, 'post_lon')
    if dlat == 0 or dlon == 0:
        raise ValueError('Invalid DEM pixel spacing')
    # Pixel-center coordinates. Include all pixel centers within bounding box.
    ys = np.arange(length)
    xs = np.arange(width)
    ysel = ys[(lat0 + ys*dlat >= min(lat1, lat2)) & (lat0 + ys*dlat <= max(lat1, lat2))]
    xsel = xs[(lon0 + xs*dlon >= min(lon1, lon2)) & (lon0 + xs*dlon <= max(lon1, lon2))]
    if not len(ysel) or not len(xsel):
        raise ValueError('Target box contains no GEOC grid pixel centers; check bounds')
    return slice(int(ysel[0]), int(ysel[-1])+1), slice(int(xsel[0]), int(xsel[-1])+1)


def read_coherence(path, length, width):
    n = length*width
    size = path.stat().st_size
    if size == n:
        arr = np.fromfile(path, dtype=np.uint8).reshape(length, width).astype(np.float32)/255.0
    elif size == 4*n:
        arr = np.fromfile(path, dtype=np.float32).reshape(length, width)
    else:
        raise ValueError(f'Unsupported .cc size {size} bytes (expected {n} or {4*n})')
    return arr


def atomic_lines(path, lines):
    tmp = path.with_name(path.name + '.11b.tmp')
    with tmp.open('w') as f:
        for line in lines:
            f.write(line + '\n')
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-d', '--input', required=True, type=Path, help='GEOCml1_noislands stack')
    p.add_argument('-t', '--tsdir', required=True, type=Path, help='Step 11 time-series directory')
    p.add_argument('--target', nargs=4, type=float, metavar=('LAT1', 'LON1', 'LAT2', 'LON2'), required=True)
    p.add_argument('--coh-threshold', type=float, default=0.3)
    p.add_argument('--min-coverage', type=float, default=0.2)
    p.add_argument('--dry-run', action='store_true', help='Create report and map without editing Step 11 files')
    args = p.parse_args()
    if not 0 <= args.coh_threshold <= 1 or not 0 <= args.min_coverage <= 1:
        p.error('Threshold and coverage must lie between 0 and 1')
    stack, tsdir = args.input.resolve(), args.tsdir.resolve()
    info = tsdir / 'info'
    bad_path = info / '11bad_ifg.txt'
    if not bad_path.is_file():
        p.error(f'Run Step 11 first; missing {bad_path}')
    length = param(stack/'slc.mli.par', 'azimuth_lines', int)
    width = param(stack/'slc.mli.par', 'range_samples', int)
    ys, xs = target_slice(stack/'EQA.dem_par', length, width, args.target)
    area = (ys.stop - ys.start) * (xs.stop - xs.start)
    all_ifgs = sorted(p.name for p in stack.iterdir() if p.is_dir() and IFG_RE.fullmatch(p.name))
    rejected = sorted(set(read_ifgs(bad_path)))
    report = []
    recovered = set()
    for ifg in rejected:
        row = dict(ifg=ifg, mean_coherence='', target_coverage='', valid_pixels=0,
                   target_pixels=area, decision='reject', reason='')
        folder = stack/ifg
        unwfile = folder/(ifg+'.unw')
        ccfile = folder/(ifg+'.cc')
        try:
            if ifg not in all_ifgs:
                raise ValueError('IFG directory missing')
            if not unwfile.is_file() or not ccfile.is_file():
                raise ValueError('Missing .unw or .cc')
            if unwfile.stat().st_size != length*width*4:
                raise ValueError('Invalid .unw file size')
            unw = np.memmap(unwfile, dtype=np.float32, mode='r', shape=(length, width))
            coh = read_coherence(ccfile, length, width)
            valid_unw = np.isfinite(unw[ys, xs]) & (unw[ys, xs] != 0)
            valid_coh = np.isfinite(coh[ys, xs]) & (coh[ys, xs] > 0) & (coh[ys, xs] <= 1)
            usable = valid_unw & valid_coh
            count = int(usable.sum())
            coverage = float(valid_unw.sum()/area)
            mean_coh = float(np.mean(coh[ys, xs][usable])) if count else float('nan')
            row.update(mean_coherence=f'{mean_coh:.6f}', target_coverage=f'{coverage:.6f}', valid_pixels=count)
            if coverage < args.min_coverage:
                row['reason'] = 'Insufficient valid-unwrapped coverage'
            elif not np.isfinite(mean_coh):
                row['reason'] = 'No valid coherence over retained pixels'
            elif mean_coh < args.coh_threshold:
                row['reason'] = 'Mean target coherence below threshold'
            else:
                row['decision'] = 'recover'
                row['reason'] = 'Target coherence and coverage pass'
                recovered.add(ifg)
        except (OSError, ValueError) as exc:
            row['reason'] = str(exc)
        report.append(row)
        print(f"{ifg}: {row['decision']} (coh={row['mean_coherence']}, coverage={row['target_coverage']})", flush=True)

    # Write independent report outside info so Step 11 won't inadvertently confuse it.
    outdir = tsdir/'11b_recovery'
    outdir.mkdir(exist_ok=True)
    with (outdir/'recovery_report.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=report[0].keys() if report else
                                ['ifg', 'mean_coherence', 'target_coverage', 'valid_pixels', 'target_pixels', 'decision', 'reason'])
        writer.writeheader()
        writer.writerows(report)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, width)
    ax.set_ylim(length, 0)
    ax.set_aspect('equal')
    ax.add_patch(Rectangle((xs.start, ys.start), xs.stop-xs.start, ys.stop-ys.start,
                           fill=False, edgecolor='red', linewidth=2))
    ax.set_title(f'Target region on GEOC grid ({area} pixels)')
    ax.set_xlabel('x pixel')
    ax.set_ylabel('y pixel')
    fig.savefig(outdir/'target_area.png', dpi=160)
    plt.close(fig)

    remaining = sorted(set(rejected) - recovered)
    print(f'\nRejected by Step 11: {len(rejected)}; recover: {len(recovered)}; remain rejected: {len(remaining)}')
    if not args.dry_run:
        # Unique backup name, never overwrite an existing backup.
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        shutil.copy2(bad_path, info/f'11bad_ifg.txt.before_11b_{stamp}')
        removed_path = info/'11removed_image.txt'
        if removed_path.exists():
            shutil.copy2(removed_path, info/f'11removed_image.txt.before_11b_{stamp}')
        atomic_lines(bad_path, remaining)
        # Keep Step 11 removed-image list consistent with the new accepted IFG network.
        good = sorted(set(all_ifgs) - set(remaining))
        all_dates = set(d for ifg in all_ifgs for d in (ifg[:8], ifg[-8:]))
        good_dates = set(d for ifg in good for d in (ifg[:8], ifg[-8:]))
        atomic_lines(removed_path, sorted(all_dates - good_dates))
        print(f'Updated {bad_path} and {removed_path}; backups saved in {info}')
    else:
        print('Dry run: Step 11 files unchanged')
    print(f'Report: {outdir}/recovery_report.csv; target PNG: {outdir}/target_area.png')
    print('Run Step 12 against the same cleaned stack (-d) and TS directory (-t).')


if __name__ == '__main__':
    main()
