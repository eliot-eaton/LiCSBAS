LiCSBAS additional Steps 10b and 11b
=====================================
Dependencies: Python 3, numpy, scipy, matplotlib.
Designed for the GEOCml* float32 .unw and uint8/float32 .cc convention in the
COMET LiCSBAS fork. Original GEOCml1 is not modified.

1. Run island removal BEFORE Step 11:

python LiCSBAS10b_remove_islands.py \
    -d GEOCml1 -o GEOCml1_noislands \
    --min-island-pixels 100

Outputs:
  GEOCml1_noislands/ (copied stack, .unw isolated components zeroed)
  GEOCml1_noislands_island_comparisons/*_before_after.png
  GEOCml1_noislands_island_comparisons/island_removal_summary.csv

Connectivity defaults to 8 neighbours; use --connectivity 4 for 4 neighbours.
Components with size < threshold are removed, including within target region.
Do not rerun into an existing output directory: rename/remove explicitly.

2. Run the usual Step 11 on the CLEANED stack:

LiCSBAS11_check_unw.py -d GEOCml1_noislands -t TS_new_GEOCml1

3. Inspect proposed recoveries (NO changes to Step 11 files):

python LiCSBAS11b_recover_ifgs.py \
    -d GEOCml1_noislands -t TS_new_GEOCml1 \
    --target LAT1 LON1 LAT2 LON2 \
    --coh-threshold 0.3 --min-coverage 0.2 --dry-run

Outputs:
  TS_new_GEOCml1/11b_recovery/recovery_report.csv
  TS_new_GEOCml1/11b_recovery/target_area.png

Mean coherence is computed from finite positive coherence values at pixels
with valid nonzero unwrapped data INSIDE the target rectangle. Coverage is
fraction of target pixels with valid nonzero unwrapped data. Coordinates are
pixel-center inclusive; the target may be specified in either corner order.

4. Apply the decisions: repeat the command without --dry-run.

This backs up and updates:
  TS_new_GEOCml1/info/11bad_ifg.txt
  TS_new_GEOCml1/info/11removed_image.txt

It does NOT edit GEOCml1_noislands/.cc, .unw, or Step 11 statistics/network PNGs.
Thus the Step 11 network plots are historical; Step 12 produces its own plots.
This script intentionally considers all IFGs in 11bad_ifg.txt. If Step 11
excluded IFGs intentionally (e.g. skip dates, temporal baseline limits, or
coregistration errors), review recovery_report.csv before applying.
Step 12 will ALSO exclude any IFGs in info/120bad_ifg.txt if that file exists.

5. Run normal Step 12 with the SAME cleaned stack and TS directory:

LiCSBAS12_loop_closure.py -d GEOCml1_noislands -t TS_new_GEOCml1

IMPORTANT:
- Step 11 recreates parts of the TS directory; rerunning it requires
  rerunning Step 11b before Step 12.
- This does not fix unwrapping cycle errors in a retained connected region.
- Mean coherence does not guarantee correct unwrapping. Step 12 loop closure
  remains necessary.
- Verify your data uses native-endian float32 .unw and standard .cc encoding.
