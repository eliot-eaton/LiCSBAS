
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

h5file = "TS_new_GEOCml1/cum.h5"

with h5py.File(h5file, "r") as f:
    tsstd = f["tsstd"][:]
    imdates = f["imdates"][:]

# tsstd starts at the second acquisition
dates = [
    datetime.strptime(
        d.decode() if isinstance(d, bytes) else str(d),
        "%Y%m%d"
    )
    for d in imdates[1:]
]

median = []
p05 = []
p95 = []
pct_high = []

for a in tsstd:
    valid = a[np.isfinite(a)]

    if valid.size:
        median.append(np.median(valid))
        p05.append(np.percentile(valid, 5))
        p95.append(np.percentile(valid, 95))
        pct_high.append(100 * np.mean(valid > 100))
    else:
        median.append(np.nan)
        p05.append(np.nan)
        p95.append(np.nan)
        pct_high.append(np.nan)

fig, axes = plt.subplots(3, 1, figsize=(11, 11))

# Panel 1: Uncertainty through time
ax = axes[0]
ax.plot(dates, median, "o-", label="Median")
ax.plot(dates, p05, "o--", label="5th percentile")
ax.plot(dates, p95, "o-", label="95th percentile")

ax.set_yscale("log")
ax.set_ylabel("1σ uncertainty (mm)")
ax.set_title("LiCSBAS WLS cumulative displacement uncertainty")
ax.grid(alpha=0.3)
ax.legend()

# Panel 2: Distribution of all uncertainties
ax = axes[1]
valid = tsstd[np.isfinite(tsstd) & (tsstd > 0)]

ax.hist(np.log10(valid), bins=100)
ax.set_xlabel("log10(1σ uncertainty / mm)")
ax.set_ylabel("Number of values")
ax.set_title("Distribution of uncertainty")
ax.grid(alpha=0.3)

# Panel 3: Fraction of pixels with extreme uncertainty
ax = axes[2]
ax.plot(dates, pct_high, "o-")
ax.set_ylabel("Pixels with σ > 100 mm (%)")
ax.set_xlabel("Acquisition date")
ax.set_title("Spatial prevalence of extreme uncertainty")
ax.grid(alpha=0.3)

for ax in (axes[0], axes[2]):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.savefig("tsstd_diagnostics.png", dpi=200)
print("Saved tsstd_diagnostics.png")
