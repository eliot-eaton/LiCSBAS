#!/usr/bin/env python3
"""LiCSBAS17_coherence.py: build pixelwise date-spanning coherence HDF5 and click viewer.

Build:
 python LiCSBAS17_coherence.py build -d GEOCml1 -t TS_GEOCml1
View:
 python LiCSBAS17_coherence.py view -i TS_GEOCml1/17coherence/coherence.h5

For each acquisition t, an IFG [start,end] contributes if start<=t<=end.
Mean, median, population std and count are computed per pixel/date.
The viewer additionally pools IFG coherence values over overlapping 1x1 to 9x9 windows.
All individual IFG coherence maps are retained for on-demand distributions.
"""
import argparse
import re
from pathlib import Path
from datetime import datetime

import h5py
import numpy as np

IFG_RE = re.compile(r"(?<!\d)(\d{8}_\d{8})(?!\d)")

def identifiers(path):
    if not path.exists():
        return set()
    return {s for line in path.read_text().splitlines()
            for s in IFG_RE.findall(line.split("#", 1)[0])}

def param(path, key):
    for line in path.read_text().splitlines():
        match = re.match(r"^\s*" + re.escape(key) + r"\s*[:=]?\s*(\S+)", line)
        if match:
            return float(match.group(1))
    raise ValueError(f"Missing {key}: {path}")

def dates_from_file(path):
    if not path.exists():
        return []
    return sorted({x.strip() for x in path.read_text().splitlines()
                   if re.fullmatch(r"\d{8}", x.strip())})

def build(args):
    ifgdir, tsadir = Path(args.ifgdir), Path(args.tsadir)
    info = tsadir / "info"
    width = int(param(ifgdir / "slc.mli.par", "range_samples"))
    length = int(param(ifgdir / "slc.mli.par", "azimuth_lines"))
    all_ifgs = {p.name for p in ifgdir.iterdir()
                if p.is_dir() and re.fullmatch(r"\d{8}_\d{8}", p.name)}
    retained = Path(args.retained) if args.retained else info / "120retained_links.txt"
    if retained.exists():
        selected = identifiers(retained)
        if not selected:
            raise ValueError(f"No YYYYMMDD_YYYYMMDD identifiers in {retained}")
        absent = selected - all_ifgs
        if absent:
            print(f"WARNING: {len(absent)} retained IFGs missing from GEOC directory")
        selected &= all_ifgs
        source = str(retained)
    else:
        if args.retained:
            raise FileNotFoundError(retained)
        bad = set().union(*(identifiers(info / x) for x in
                            ("11bad_ifg.txt", "12bad_ifg.txt", "120bad_ifg.txt")))
        selected = all_ifgs - bad
        source = "GEOC minus bad IFG lists (fallback)"
        print("WARNING: retained list missing; step 13 optional exclusions may differ")
    dates = dates_from_file(info / "13used_image.txt")
    if not dates:
        cumfile = tsadir / "cum.h5"
        if cumfile.exists():
            with h5py.File(cumfile) as f:
                dates = sorted(str(v.decode() if isinstance(v, bytes) else v)
                               for v in f["imdates"][:])
        else:
            dates = sorted({d for i in selected for d in (i[:8], i[9:])})
    selected = sorted(i for i in selected if i[:8] in dates and i[9:] in dates)
    if not selected:
        raise ValueError("No retained interferograms with acquisition dates")
    paths = [ifgdir / i / (i + ".cc") for i in selected]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} coherence files, first: {missing[0]}")
    out = Path(args.output) if args.output else tsadir / "17coherence" / "coherence.h5"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.overwrite:
        raise FileExistsError(f"{out} exists. Pass --overwrite to replace it.")
    spans = [np.array([min(i[:8], i[9:]) <= d <= max(i[:8], i[9:])
                       for i in selected], dtype=bool) for d in dates]
    dt = h5py.string_dtype("ascii")
    with h5py.File(out, "w") as f:
        f.attrs["source_ifg_list"] = source
        f.attrs["span_rule"] = "inclusive start <= acquisition <= end"
        f.attrs["std_ddof"] = 0
        f.create_dataset("imdates", data=np.asarray(dates, dtype=object), dtype=dt)
        f.create_dataset("ifgdates", data=np.asarray(selected, dtype=object), dtype=dt)
        f.create_dataset("date_ifg_mask", data=np.stack(spans), compression="gzip")
        for key in ("corner_lat", "corner_lon", "post_lat", "post_lon"):
            dem = ifgdir / "EQA.dem_par"
            if dem.exists():
                try:
                    f.create_dataset(key, data=param(dem, key))
                except ValueError:
                    pass
        chunks = (1, min(args.chunkrows, length), min(256, width))
        raw = f.create_dataset("ifg_coh", shape=(len(selected), length, width),
                               dtype="f4", chunks=chunks, compression="gzip",
                               compression_opts=2, fillvalue=np.nan)
        for j, (ifg, path) in enumerate(zip(selected, paths)):
            nbytes = path.stat().st_size
            if nbytes == length * width:
                dtype, scale = np.uint8, 255.0
            elif nbytes == length * width * 4:
                dtype, scale = np.dtype("<f4"), 1.0
            else:
                raise ValueError(f"Unrecognized coherence raster size: {path}")
            raster = np.memmap(path, mode="r", dtype=dtype, shape=(length, width))
            for y0 in range(0, length, args.chunkrows):
                y1 = min(length, y0 + args.chunkrows)
                a = np.asarray(raster[y0:y1, :], dtype=np.float32) / scale
                a[~np.isfinite(a) | (a <= 0) | (a > 1)] = np.nan
                raw[j, y0:y1, :] = a
            del raster
            if (j + 1) % 25 == 0 or j + 1 == len(selected):
                print(f"Loaded {j+1}/{len(selected)} IFGs", flush=True)
        shape = (len(dates), length, width)
        stats = {name: f.create_dataset(name, shape=shape,
                 dtype="f4" if name != "count" else "u2",
                 chunks=chunks, compression="gzip", compression_opts=2)
                 for name in ("mean", "median", "std", "count")}
        for k, (date, mask) in enumerate(zip(dates, spans)):
            inds = np.flatnonzero(mask)
            for y0 in range(0, length, args.chunkrows):
                y1 = min(length, y0 + args.chunkrows)
                # Restrict memory usage by processing small spatial stripes.
                stack = np.stack([raw[int(j), y0:y1, :] for j in inds]) if len(inds) else np.empty((0, y1-y0, width))
                n = np.sum(np.isfinite(stack), axis=0)
                if len(inds):
                    with np.errstate(invalid="ignore", divide="ignore"):
                        mean = np.nansum(stack, axis=0) / np.where(n, n, 1)
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=RuntimeWarning)
                            median = np.nanmedian(stack, axis=0)
                        std = np.sqrt(np.nansum((stack - mean[None, :, :])**2,
                                               axis=0) / np.where(n, n, 1))
                    mean[n == 0] = np.nan
                    std[n == 0] = np.nan
                else:
                    mean = median = std = np.full((y1-y0, width), np.nan)
                stats["mean"][k, y0:y1, :] = mean
                stats["median"][k, y0:y1, :] = median
                stats["std"][k, y0:y1, :] = std
                stats["count"][k, y0:y1, :] = n
            print(f"Calculated {k+1}/{len(dates)}: {date} ({len(inds)} spanning IFGs)", flush=True)
    print(f"Saved {out}")

def view(args):
    """Interactive point/window and signal-vs-reference polygon coherence viewer."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.widgets import Slider, Button, TextBox
    from matplotlib.patches import Rectangle, Polygon
    from matplotlib.path import Path as MplPath

    f = h5py.File(args.input, "r")
    dates = [datetime.strptime(v.decode() if isinstance(v, bytes) else str(v), "%Y%m%d")
             for v in f["imdates"][:]]
    length, width = f["mean"].shape[1:]
    masks = f["date_ifg_mask"][:].astype(bool)
    ifg_names = [v.decode() if isinstance(v, bytes) else str(v)
                 for v in f["ifgdates"][:]]
    ifg_duration_days = np.asarray([
        (datetime.strptime(name[9:17], "%Y%m%d")
         - datetime.strptime(name[:8], "%Y%m%d")).days
        for name in ifg_names], dtype=float)
    if len(ifg_duration_days) != masks.shape[1]:
        raise ValueError("IFG dates do not match date_ifg_mask")
    duration_min, duration_max = float(ifg_duration_days.min()), float(ifg_duration_days.max())
    cumf = h5py.File(args.cum_filt, "r") if args.cum_filt else None
    if cumf is not None:
        cumkey = "cum" if "cum" in cumf else "cum_model"
        cumdata = cumf[cumkey]
        if cumdata.shape[1:] != (length, width):
            raise ValueError(f"cum_filt spatial shape {cumdata.shape[1:]} does not match coherence {(length,width)}")
        cumdates = [datetime.strptime(v.decode() if isinstance(v, bytes) else str(v), "%Y%m%d")
                    for v in cumf["imdates"][:]]
        if len(cumdates) != cumdata.shape[0]:
            raise ValueError("cum_filt imdates length does not match cum dataset")
        gapdata = cumf.get("gap")
        if gapdata is not None and gapdata.shape != (len(cumdates)-1, length, width):
            raise ValueError(f"Unexpected gap shape {gapdata.shape}; expected {(len(cumdates)-1,length,width)}")
    fig = plt.figure(figsize=(19, 10))
    axmap = fig.add_axes([.045, .24, .35 if cumf is None else .185, .61])
    axdispmap = (fig.add_axes([.275, .24, .185, .61], sharex=axmap, sharey=axmap)
                 if cumf is not None else None)
    if cumf is None:
        axpoint = fig.add_axes([.50, .675, .46, .195])
        axsignal = fig.add_axes([.50, .440, .46, .195], sharex=axpoint)
        axreference = fig.add_axes([.50, .205, .46, .195], sharex=axpoint)
        axcoh, axcum = None, None
    else:
        axcoh = fig.add_axes([.56, .555, .39, .31])
        axcum = fig.add_axes([.56, .205, .39, .29], sharex=axcoh)
        axpoint = axsignal = axreference = None
    sliderax = fig.add_axes([.09, .105, .30, .035])
    fig.text(.05, .025, "Polygons: left-click vertices; right-click or press Enter to finish.",
             fontsize=10)
    windowax = fig.add_axes([.19, .935, .09, .04])
    pointax = fig.add_axes([.34, .935, .085, .04])
    signalax = fig.add_axes([.44, .935, .11, .04])
    refax = fig.add_axes([.56, .935, .11, .04])
    clearax = fig.add_axes([.68, .935, .10, .04])
    fig.text(.055, .957, "Window size:", fontsize=10, va="center")
    window_input = TextBox(windowax, "", initial=str(args.window))
    buttons = [("Point/window", Button(pointax, "Point/window")),
               ("Signal", Button(signalax, "Draw signal")),
               ("Reference", Button(refax, "Draw reference")),
               ("Clear", Button(clearax, "Clear areas"))]
    im = axmap.imshow(f["mean"][args.date_index], origin="upper", vmin=0, vmax=1,
                      cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=axmap, label="Mean coherence (pixel)", fraction=.045)
    axmap.set(xlabel="Pixel X", ylabel="Pixel Y")
    disp_im = None
    if cumf is not None:
        from matplotlib.colors import TwoSlopeNorm
        # Consistent colour limits for all acquisitions, sampled from the final map.
        final = np.asarray(cumdata[-1], dtype=float)
        finite = final[np.isfinite(final)]
        scale = float(np.nanpercentile(np.abs(finite), 98)) if finite.size else 1.0
        scale = max(scale, 1e-6)
        disp_im = axdispmap.imshow(np.asarray(cumdata[0], dtype=float), origin="upper",
                                   cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-scale, vcenter=0, vmax=scale),
                                   interpolation="nearest")
        fig.colorbar(disp_im, ax=axdispmap, label="Cumulative LOS (mm)", fraction=.045)
        axdispmap.set(xlabel="Pixel X")
    slider = Slider(sliderax, "Date index", 0, len(dates)-1,
                    valinit=args.date_index, valstep=1)
    point = axmap.plot([], [], "r+", ms=12, mew=2)[0]
    window_box = Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", lw=1.5)
    axmap.add_patch(window_box)
    disp_window_box = None
    disp_point = None
    if axdispmap is not None:
        disp_window_box = Rectangle((0, 0), 1, 1, fill=False, edgecolor="red", lw=1.5)
        axdispmap.add_patch(disp_window_box)
        disp_point = axdispmap.plot([], [], "r+", ms=12, mew=2)[0]
    selected = [width//2, length//2]
    current_window = [args.window]
    mode = ["Point/window"]
    polygons = {"Signal": None, "Reference": None}
    patches = {"Signal": None, "Reference": None}
    vertices = []
    draft, = axmap.plot([], [], "yellow", linestyle="--", marker="o", ms=5, lw=1.7, zorder=12)
    def update_draft():
        draft.set_data([p[0] for p in vertices], [p[1] for p in vertices])
        fig.canvas.draw_idle()
    # Spatially pooled IFG×pixel values, preserving the full distribution.
    def extract_region(mask2d):
        yy, xx = np.where(mask2d)
        if not len(xx):
            return None
        x0, x1 = int(xx.min()), int(xx.max()) + 1
        y0, y1 = int(yy.min()), int(yy.max()) + 1
        local = mask2d[y0:y1, x0:x1]
        data = np.asarray(f["ifg_coh"][:, y0:y1, x0:x1], dtype=np.float32)
        data = data[:, local]  # IFGs x selected pixels
        # One spatial mean per IFG, ignoring invalid pixels in the region.
        valid_per_ifg = np.isfinite(data).sum(axis=1)
        ifg_means = np.nansum(data, axis=1) / np.maximum(valid_per_ifg, 1)
        ifg_means[valid_per_ifg == 0] = np.nan
        means, medians, stds, counts, distributions, ifg_distributions, duration_distributions = [], [], [], [], [], [], []
        for spanning in masks:
            sample = data[spanning].ravel()
            sample = sample[np.isfinite(sample)]
            distributions.append(sample)
            # Each dot is the spatially averaged coherence of one IFG
            # spanning this date, rather than one dot per pixel/IFG.
            selected_ifg_means = ifg_means[spanning]
            valid_ifg = np.isfinite(selected_ifg_means)
            ifg_distributions.append(selected_ifg_means[valid_ifg])
            duration_distributions.append(ifg_duration_days[spanning][valid_ifg])
            counts.append(len(sample))
            means.append(float(np.mean(sample)) if len(sample) else np.nan)
            medians.append(float(np.median(sample)) if len(sample) else np.nan)
            stds.append(float(np.std(sample)) if len(sample) else np.nan)
        return dict(mean=np.array(means), median=np.array(medians),
                    std=np.array(stds), count=np.array(counts),
                    distributions=distributions, ifg_distributions=ifg_distributions,
                    duration_distributions=duration_distributions,
                    npix=int(mask2d.sum()))

    def window_mask():
        x, y = selected
        h = current_window[0] // 2
        x0, x1 = max(0, x-h), min(width, x+h+1)
        y0, y1 = max(0, y-h), min(length, y+h+1)
        m = np.zeros((length, width), dtype=bool)
        m[y0:y1, x0:x1] = True
        point.set_data([x], [y])
        window_box.set_bounds(x0-.5, y0-.5, x1-x0, y1-y0)
        if disp_point is not None:
            disp_point.set_data([x], [y])
            disp_window_box.set_bounds(x0-.5, y0-.5, x1-x0, y1-y0)
        return m

    def polygon_mask(verts):
        if verts is None:
            return None
        a = np.asarray(verts)
        x0 = max(0, int(np.floor(a[:, 0].min()))-1)
        x1 = min(width, int(np.ceil(a[:, 0].max()))+2)
        y0 = max(0, int(np.floor(a[:, 1].min()))-1)
        y1 = min(length, int(np.ceil(a[:, 1].max()))+2)
        m = np.zeros((length, width), dtype=bool)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m[y0:y1, x0:x1] = MplPath(verts).contains_points(
            np.column_stack([xx.ravel(), yy.ravel()]), radius=1e-7).reshape(xx.shape)
        return m

    def scatter_ifg_means(ax, stats, color):
        """Spatial mean per IFG; marker area scales with temporal baseline (days).

        Short-baseline IFGs have larger markers and are drawn first, so
        smaller long-baseline markers remain visible when points overlap.
        """
        xs, ys, durations = [], [], []
        for date, values, days in zip(dates, stats["ifg_distributions"],
                                      stats["duration_distributions"]):
            xs.extend([date] * len(values))
            ys.extend(values.tolist())
            durations.extend(days.tolist())
        if not ys:
            return
        durations = np.asarray(durations, dtype=float)
        if duration_max > duration_min:
            scaled = (duration_max - durations) / (duration_max - duration_min)
        else:
            scaled = np.full_like(durations, 0.5)
        # Subtle area scaling, 8 to 20 pt^2; based on IFG span, not coherence.
        sizes = 2.0 + 25 * scaled
        # Matplotlib renders later points on top within a scatter collection.
        # Draw shortest-duration / largest points first.
        order = np.argsort(-sizes, kind="stable")
        ax.scatter(np.asarray(xs, dtype=object)[order],
                   np.asarray(ys)[order], s=sizes[order],
                   color=color, alpha=.35, edgecolors="none",
                   zorder=2, rasterized=True,
                   label="IFG spatial means (larger = shorter span)")

    def draw_series(ax, stats, name, color):
        if stats is None:
            ax.text(.5, .5, f"Draw a {name.lower()} polygon to see its time series",
                    ha="center", va="center", transform=ax.transAxes,
                    color="gray")
            return
        ax.plot(dates, stats["mean"], "-o", ms=2.5, lw=1.5,
                color=color, label="Mean")
        ax.plot(dates, stats["median"], "--", lw=1.2,
                color=color, alpha=.85, label="Median")
        ax.fill_between(dates, np.clip(stats["mean"]-stats["std"], 0, 1),
                        np.clip(stats["mean"]+stats["std"], 0, 1),
                        color=color, alpha=.16, label="Mean ± 1 std")
        scatter_ifg_means(ax, stats, color)
        ax.legend(loc="upper right", fontsize=7, ncol=3)

    def draw_combined_coherence(ax, point_stats, signal_stats, reference_stats):
        for stats, label, color in (
            (point_stats, "Pixel/window", "tab:green"),
            (signal_stats, "Deforming area", "tab:orange"),
            (reference_stats, "Reference area", "tab:blue")
        ):
            if stats is None:
                continue
            ax.plot(dates, stats["mean"], "-o", ms=2.5, lw=1.6,
                    color=color, label=f"{label} mean")
            ax.plot(dates, stats["median"], "--", lw=1.0, alpha=.7,
                    color=color, label=f"{label} median")
            ax.fill_between(dates, np.clip(stats["mean"]-stats["std"],0,1),
                            np.clip(stats["mean"]+stats["std"],0,1),
                            color=color, alpha=.10)
            scatter_ifg_means(ax, stats, color)
        ax.set(title="Coherence time series (lines: pooled stats; dots: IFG area means, larger dots = shorter spans)",
               ylabel="Coherence", ylim=(0,1))
        ax.legend(loc="best", fontsize=8, ncol=2)
        ax.grid(alpha=.2)

    gap_colorbar = [None]
    gap_cmap = None
    if gapdata is not None:
        try:
            from cmcrameri import cm as cmc
            gap_cmap = cmc.batlow_r  # high gap fraction maps to the dark end
        except ImportError as exc:
            raise ImportError("Batlow gap colours require cmcrameri. Install: pip install cmcrameri") from exc
    def draw_displacement(ax, mask):
        yy, xx = np.where(mask)
        if not len(xx):
            return
        x0, x1, y0, y1 = int(xx.min()), int(xx.max())+1, int(yy.min()), int(yy.max())+1
        local = mask[y0:y1,x0:x1]
        # LiCSBAS cumulative displacement is stored in mm.
        cube = np.asarray(cumdata[:,y0:y1,x0:x1], dtype=float)[:,local]
        with np.errstate(invalid="ignore"):
            valid = np.isfinite(cube)
            counts = valid.sum(axis=1)
            disp = np.nansum(cube,axis=1) / np.where(counts, counts, 1)
            disp[counts == 0] = np.nan
        ax.plot(cumdates, disp, "-o", color="tab:purple", lw=1.5,
                ms=3, label=f"Mean cumulative LOS ({len(xx)} pixels)")
        if gapdata is not None:
            gap_cube = np.asarray(gapdata[:,y0:y1,x0:x1])[:,local]
            fraction = np.mean(gap_cube == 1, axis=1)
            # Midpoint of each acquisition interval, as in LiCSBAS_plot_ts.py.
            for i, fraction_gap in enumerate(fraction):
                if fraction_gap <= 0:
                    continue
                midpoint = cumdates[i] + (cumdates[i+1]-cumdates[i])/2
                ax.axvline(midpoint, color=gap_cmap(float(fraction_gap)), lw=1.7,
                           alpha=1.0, zorder=1)
            from matplotlib.cm import ScalarMappable
            from matplotlib.colors import Normalize
            if gap_colorbar[0] is None:
                cb = fig.colorbar(ScalarMappable(norm=Normalize(0,1), cmap=gap_cmap),
                                  ax=ax, fraction=.025, pad=.012)
                cb.set_label("Fraction of window pixels with gap", fontsize=8)
                gap_colorbar[0] = cb
        ax.set(title=f"Cumulative LOS displacement — selected {current_window[0]}×{current_window[0]} window",
               ylabel="LOS displacement (mm)", xlabel="Acquisition date")
        ax.grid(alpha=.2)
        ax.legend(loc="best", fontsize=8)

    def refresh():
        point_mask = window_mask()
        point_stats = extract_region(point_mask)
        signal_stats = (extract_region(polygon_mask(polygons["Signal"]))
                        if polygons["Signal"] is not None else None)
        reference_stats = (extract_region(polygon_mask(polygons["Reference"]))
                           if polygons["Reference"] is not None else None)
        if cumf is not None:
            axcoh.clear()
            axcum.clear()
            draw_combined_coherence(axcoh, point_stats, signal_stats, reference_stats)
            draw_displacement(axcum, point_mask)
            plt.setp(axcoh.get_xticklabels(), visible=False)
            bottom = axcum
        else:
            panels = [
                (axpoint, point_stats, "Pixel/window", "tab:green",
                 f"Pixel/window at ({selected[0]}, {selected[1]}) — {current_window[0]}×{current_window[0]}"),
                (axsignal, signal_stats, "Deforming area", "tab:orange", "Deforming area (signal)"),
                (axreference, reference_stats, "Reference area", "tab:blue", "Reference area"),
            ]
            for ax, stats, name, color, title in panels:
                ax.clear()
                draw_series(ax, stats, name, color)
                ax.set(title=title, ylabel="Coherence", ylim=(0,1))
                ax.grid(alpha=.2)
            plt.setp(axpoint.get_xticklabels(), visible=False)
            plt.setp(axsignal.get_xticklabels(), visible=False)
            bottom = axreference
        locator = mdates.AutoDateLocator()
        bottom.xaxis.set_major_locator(locator)
        bottom.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
        fig.canvas.draw_idle()

    def change_mode(new):
        mode[0] = new
        vertices.clear()
        update_draft()
        if new in ("Signal", "Reference"):
            print(f"{new}: left-click vertices; right-click or press Enter to finish the polygon.")
        else:
            print("Point/window mode: left-click to move the window.")
        fig.canvas.draw_idle()

    def click(event):
        if event.inaxes not in (axmap, axdispmap) or event.xdata is None or event.ydata is None:
            return
        if event.inaxes is axdispmap and mode[0] != "Point/window":
            return
        if mode[0] == "Point/window":
            if event.button == 1:
                x, y = int(round(event.xdata)), int(round(event.ydata))
                if 0 <= x < width and 0 <= y < length:
                    selected[:] = [x, y]
                    refresh()
        elif event.button == 1:
            vertices.append((float(event.xdata), float(event.ydata)))
            update_draft()
        elif event.button == 3:
            finish_polygon()

    def finish_polygon():
        if mode[0] not in ("Signal", "Reference"):
            return
        if len(vertices) < 3:
            print("Polygon needs at least 3 vertices.")
            return
        name = mode[0]
        polygons[name] = list(vertices)
        if patches[name] is not None:
            patches[name].remove()
        patches[name] = Polygon(vertices, closed=True, fill=False,
                                edgecolor=("darkorange" if name == "Signal" else "blue"),
                                lw=2.5, zorder=10)
        axmap.add_patch(patches[name])
        vertices.clear()
        update_draft()
        print(f"Completed {name.lower()} polygon. Draw the other polygon to compare.")
        change_mode("Point/window")
        refresh()


    def set_window(value):
        try:
            n = int(value.strip())
            if n < 1 or n % 2 != 1:
                raise ValueError
        except ValueError:
            print("Window size must be a positive odd integer (1, 3, 5, 7, ...).")
            return
        current_window[0] = n
        if n > max(width, length)*2:
            print("Note: window extends beyond image and will be clipped.")
        refresh()

    def clear(event):
        for name in polygons:
            polygons[name] = None
            if patches[name] is not None:
                patches[name].remove()
                patches[name] = None
        change_mode("Point/window")
        refresh()

    def set_date(k):
        k = int(k)
        im.set_data(f["mean"][k])
        axmap.set_title(f"Mean coherence: {dates[k]:%Y-%m-%d}")
        if disp_im is not None:
            # cum_filt can have different acquisition dates; use the nearest date.
            j = min(range(len(cumdates)), key=lambda q: abs((cumdates[q]-dates[k]).total_seconds()))
            disp_im.set_data(np.asarray(cumdata[j], dtype=float))
            axdispmap.set_title(f"Cumulative LOS: {cumdates[j]:%Y-%m-%d}")
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("button_press_event", click)
    # Enter provides a reliable alternative to right-click on trackpads and macOS.
    def on_key(event):
        if event.key in ("enter", "return") and mode[0] in ("Signal", "Reference"):
            finish_polygon()
    fig.canvas.mpl_connect("key_press_event", on_key)
    window_input.on_submit(set_window)
    for name, button in buttons:
        button.on_clicked(clear if name == "Clear" else
                          (lambda event, name=name: change_mode(name)))
    slider.on_changed(set_date)
    set_date(args.date_index)
    refresh()
    try:
        plt.show()
    finally:
        f.close()
    if cumf is not None:
        cumf.close()

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("-d", "--ifgdir", required=True)
    b.add_argument("-t", "--tsadir", required=True)
    b.add_argument("--retained")
    b.add_argument("-o", "--output")
    b.add_argument("--chunkrows", type=int, default=32)
    b.add_argument("--overwrite", action="store_true")
    v = sub.add_parser("view")
    v.add_argument("-i", "--input", required=True)
    v.add_argument("--date-index", type=int, default=0)
    v.add_argument("--cum-filt", help="Optional LiCSBAS cum_filt.h5: combined coherence and cumulative LOS plot")
    v.add_argument("--window", type=int, default=1,
                   help="Initial spatial window size (default 1)")
    args = parser.parse_args()
    if args.cmd == "view" and (args.window < 1 or args.window % 2 == 0):
        parser.error("--window must be a positive odd integer")
    (build if args.cmd == "build" else view)(args)

if __name__ == "__main__":
    main()

