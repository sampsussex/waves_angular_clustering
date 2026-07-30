# %%
"""
Find the sky footprint of VIRCAM detector 16 for every VIKING pawprint,
without downloading any image data.

Strategy
--------
1. Query the ESO TAP service for VIKING exposures (Z band in this example).
2. For each dp_id, fetch ONLY the FITS header via the ESO header service:
       https://archive.eso.org/hdr?DpId=<dp_id>
   (~100-300 kB of text per file, vs ~750 MB for the pixel data).
3. Parse the multi-extension header, find the HDU with
       HIERARCH ESO DET CHIP NO = 16   (EXTNAME 'DET1.CHIP16'),
   build an astropy WCS from it, and project the 2048x2048 chip corners
   onto the sky.
4. Write one row per pawprint: the exact corner quadrilateral plus an
   axis-aligned RA/Dec bounding box you can use as a rectangular mask.

Files with no chip-16 extension (e.g. Phase-3 *tiles*, which are single
mosaic HDUs) are skipped automatically -- you only get real pawprints.

Requires: pyvo, requests, astropy, numpy
"""

import os
import re
import html
import json
import numpy as np
import requests
import pyvo
from astropy.io.fits import Header
from astropy.wcs import WCS
from concurrent.futures import ThreadPoolExecutor, as_completed

TAP_URL = "https://archive.eso.org/tap_obs"
HDR_URL = "https://archive.eso.org/hdr"
CACHE_DIR = "eso_headers"          # headers are cached here so re-runs are free
N_WORKERS = 8                      # be polite to the archive
CHIP = 16
NX = NY = 2048                     # VIRCAM detector size in pixels

os.makedirs(CACHE_DIR, exist_ok=True)

# ----------------------------------------------------------------------
# 1. Get the list of dp_ids
# ----------------------------------------------------------------------

def query_phase3_products():
    """Phase-3 VIKING Z-band images (pawprints AND tiles; tiles get
    filtered out later because they have no chip-16 extension)."""
    service = pyvo.dal.TAPService(TAP_URL)
    query = """
    SELECT s_ra, s_dec, dp_id, obs_id
    FROM ivoa.ObsCore
    WHERE obs_collection='VIKING'
      AND dataproduct_type='image'
      AND em_min <= 0.878e-6
      AND em_max >= 0.878e-6
    """
    job = service.submit_job(query, maxrec=1_000_000)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
    job.raise_if_error()
    return job.fetch_result().to_table()


def query_raw_frames(band="Z"):
    """Raw VIKING science frames from dbo.raw.  179.A-2004 is the VIKING
    programme ID.  Unlike the Phase 3 collection (tiles only for VIKING),
    every raw VIRCAM frame is a 16-extension MEF, so chip 16 is always
    present.  Raw headers carry the telescope-model ZPN WCS, accurate to
    a few arcsec -- fine for masking (add a small buffer to be safe)."""
    service = pyvo.dal.TAPService(TAP_URL)
    query = f"""
    SELECT dp_id, ob_id, ra, dec, filter_path, tpl_start
    FROM dbo.raw
    WHERE prog_id LIKE '179.A-2004%'
      AND dp_cat = 'SCIENCE'
      AND instrument = 'VIRCAM'
      AND filter_path = '{band}'
    """
    job = service.submit_job(query, maxrec=1_000_000)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"])
    job.raise_if_error()
    return job.fetch_result().to_table()

# ----------------------------------------------------------------------
# 2. Header-only fetch + parse
# ----------------------------------------------------------------------

def fetch_header_text(dp_id):
    """Download (and cache) the full multi-HDU header dump for one file."""
    cache = os.path.join(CACHE_DIR, dp_id.replace(":", "_").replace("/", "_") + ".hdr")
    if os.path.exists(cache):
        with open(cache) as f:
            txt = f.read()
        if "SIMPLE" in txt or "XTENSION" in txt:
            return txt
        os.remove(cache)  # a cached error page from an earlier run -- refetch

    # NB: build the URL by hand.  requests' params= would encode ':' as
    # '%3A' and the ESO header service rejects that with
    # "Invalid format for dp_id".  Literal colons are fine in a query string.
    r = requests.get(f"{HDR_URL}?DpId={dp_id}", timeout=120)
    r.raise_for_status()
    txt = r.text
    # The service may wrap the header in a simple HTML page -- strip it.
    if "<html" in txt[:512].lower():
        m = re.search(r"<pre[^>]*>(.*?)</pre>", txt, re.S | re.I)
        txt = m.group(1) if m else re.sub(r"<[^>]+>", "", txt)
        txt = html.unescape(txt)
    if "SIMPLE" not in txt and "XTENSION" not in txt:
        raise RuntimeError(f"Header service returned no header for {dp_id}: "
                           f"{txt.strip()[:120]!r}")
    with open(cache, "w") as f:
        f.write(txt)
    return txt


def split_hdus(text):
    """Split a text header dump into per-HDU astropy Header objects."""
    blocks, cur = [], []
    for line in text.splitlines():
        if line.startswith(("SIMPLE  ", "XTENSION")) and cur:
            blocks.append(cur)
            cur = []
        if line.strip():
            cur.append(line)
    if cur:
        blocks.append(cur)
    headers = []
    for b in blocks:
        try:
            headers.append(Header.fromstring("\n".join(b), sep="\n"))
        except Exception:
            pass  # tolerate the odd malformed card in the dump
    return headers


def _chip_number(h):
    """Detector number from a header, trying every known convention:
    raw VIRCAM (ESO DET CHIP NO / EXTNAME DET1.CHIPnn) and CASU-processed
    pawprints (CAMNUM / DETNUM)."""
    for key in ("ESO DET CHIP NO", "CAMNUM", "DETNUM"):
        v = h.get(key)
        if v is not None:
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
    m = re.search(r"CHIP(\d+)", str(h.get("EXTNAME", "")))
    return int(m.group(1)) if m else None


def chip16_corners(dp_id):
    """Return (4,2) array of RA/Dec corners of detector 16, or None."""
    hdus = split_hdus(fetch_header_text(dp_id))
    target = None
    for h in hdus:
        if _chip_number(h) == CHIP:
            target = h
            break
    if target is None and len(hdus) == 17 and all(
            "CRVAL1" in h for h in hdus[1:]):
        # primary + 16 image extensions but no recognisable chip keyword:
        # VIRCAM MEFs store chips in order, so fall back to extension 16.
        target = hdus[16]
    if target is None:
        return None                                # e.g. a tile -> skip
    nx = target.get("NAXIS1") or NX
    ny = target.get("NAXIS2") or NY
    w = WCS(target)
    pix = np.array([[0.5, 0.5], [nx + 0.5, 0.5],
                    [nx + 0.5, ny + 0.5], [0.5, ny + 0.5]])
    return w.all_pix2world(pix, 1)                 # includes ZPN distortion


def diagnose(dp_id):
    """Print what the header service returned for one file, HDU by HDU.
    Run this on one dp_id before processing thousands."""
    txt = fetch_header_text(dp_id)
    print(f"--- {dp_id}: fetched {len(txt)} chars ---")
    print(txt[:300])
    hdus = split_hdus(txt)
    print(f"\n{len(hdus)} HDU(s) parsed:")
    for i, h in enumerate(hdus):
        print(f"  [{i:2d}] EXTNAME={h.get('EXTNAME')!r:20} "
              f"chip={_chip_number(h)!r:5} "
              f"NAXIS1x2={h.get('NAXIS1')}x{h.get('NAXIS2')} "
              f"CTYPE1={h.get('CTYPE1')!r} CRVAL={h.get('CRVAL1')},{h.get('CRVAL2')}")
    c = chip16_corners(dp_id)
    print("\nchip-16 corners:", c)

# ----------------------------------------------------------------------
# 3. Build the mask table
# ----------------------------------------------------------------------

def bbox_from_corners(c):
    """Axis-aligned RA/Dec bounding box, RA-wrap safe."""
    ra, dec = c[:, 0], c[:, 1]
    if ra.max() - ra.min() > 180:                 # straddles RA = 0
        ra = np.where(ra > 180, ra - 360, ra)
    return (ra.min() % 360, ra.max() % 360, dec.min(), dec.max())


def process(table, id_col="dp_id"):
    results, skipped = [], 0
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(chip16_corners, str(row[id_col])): row for row in table}
        for i, fut in enumerate(as_completed(futs), 1):
            row = futs[fut]
            try:
                corners = fut.result()
            except Exception as e:
                print(f"  ! {row[id_col]}: {e}")
                continue
            if corners is None:
                skipped += 1                       # tile / no chip 16
                continue
            ra_min, ra_max, dec_min, dec_max = bbox_from_corners(corners)
            results.append({
                "dp_id": str(row[id_col]),
                "corners_ra": corners[:, 0].tolist(),
                "corners_dec": corners[:, 1].tolist(),
                "ra_min": ra_min, "ra_max": ra_max,
                "dec_min": dec_min, "dec_max": dec_max,
            })
            if i % 100 == 0:
                print(f"  {i}/{len(futs)} done ({skipped} skipped)")
    print(f"Finished: {len(results)} chip-16 footprints, {skipped} files without chip 16 (tiles etc.)")
    return results


# ----------------------------------------------------------------------
# 4. Optional: keep only raw frames that went into the RELEASED tiles
# ----------------------------------------------------------------------

from astropy.time import Time


def _dp_id_mjd(dp_id):
    """Exposure start MJD encoded in a raw dp_id, e.g.
    'VCAM.2009-12-21T03:03:46.372' -> MJD."""
    return Time(dp_id.split(".", 1)[1], format="isot", scale="utc").mjd


def tile_provenance(tile_dp_id):
    """Fetch one released tile's primary header and return its
    provenance info: PROVn values, OB ids, and the MJD execution window."""
    hdus = split_hdus(fetch_header_text(tile_dp_id))
    h = hdus[0]
    provs = [str(h[k]).strip() for k in h if re.fullmatch(r"PROV\d+", k)]
    obids = [int(h[k]) for k in h if re.fullmatch(r"OBID\d+", k)]
    return {
        "tile": tile_dp_id,
        "provs": provs,
        "obids": obids,
        "mjd_obs": h.get("MJD-OBS"),
        "mjd_end": h.get("MJD-END"),
    }


def filter_raw_to_released(raw_table, tile_table,
                           pad_days=120.0 / 86400.0):
    """Keep only the raw frames that contributed to the released tiles.

    Strategy 1: if the tiles' PROVn entries are archive dp_ids
    (VCAM.*/VIRCAM.*), match directly on dp_id.
    Strategy 2 (usual case for CASU releases, where PROVn are internal
    filenames): keep a raw frame if its OB id matches a tile's OBIDn AND
    its exposure start falls inside that tile's [MJD-OBS, MJD-END]
    window (padded by `pad_days`, default 2 min).  This drops repeated /
    QC-rejected OB executions whose frames never entered a tile.
    """
    print(f"Fetching provenance for {len(tile_table)} tile headers...")
    tiles = []
    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(tile_provenance, str(r["dp_id"])) for r in tile_table]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                tiles.append(fut.result())
            except Exception as e:
                print(f"  ! tile header failed: {e}")
            if i % 200 == 0:
                print(f"  {i}/{len(futs)} tile headers done")

    # Strategy 1: direct dp_id provenance?
    prov_ids = {p for t in tiles for p in t["provs"]
                if p.startswith(("VCAM.", "VIRCAM."))}
    if prov_ids:
        keep = [i for i, r in enumerate(raw_table)
                if str(r["dp_id"]) in prov_ids]
        print(f"PROVn are dp_ids: kept {len(keep)}/{len(raw_table)} raw frames")
        return raw_table[keep]

    # Strategy 2: OB id + execution time window
    windows = {}                                   # ob_id -> [(mjd0, mjd1)]
    n_nowin = 0
    for t in tiles:
        if t["mjd_obs"] is None or t["mjd_end"] is None or not t["obids"]:
            n_nowin += 1
            continue
        for ob in t["obids"]:
            windows.setdefault(ob, []).append(
                (float(t["mjd_obs"]) - pad_days,
                 float(t["mjd_end"]) + pad_days))
    if n_nowin:
        print(f"  warning: {n_nowin} tiles lacked OBID/MJD keywords")

    keep = []
    for i, r in enumerate(raw_table):
        ob = int(r["ob_id"])
        if ob not in windows:
            continue
        mjd = _dp_id_mjd(str(r["dp_id"]))
        if any(m0 <= mjd <= m1 for m0, m1 in windows[ob]):
            keep.append(i)
    print(f"OB+time matching: kept {len(keep)}/{len(raw_table)} raw frames "
          f"({len(windows)} OBs with released tiles)")
    return raw_table[keep]


if __name__ == "__main__":
    # NB: the VIKING Phase 3 collection exposes only TILES as science
    # images (single-HDU mosaics -> no chip 16), so use the raw frames:
    table = query_raw_frames(band="Z")
    print(f"{len(table)} raw frames returned by TAP")

    # Restrict to frames that actually entered the released Z tiles
    # (drops repeated / QC-rejected OB executions):
    tiles = query_phase3_products()
    table = filter_raw_to_released(table, tiles)

    # Before a full run, sanity-check ONE file so an empty result
    # can be traced to fetching / parsing / chip identification:
    diagnose(str(table[0]["dp_id"]))

    masks = process(table)

    with open("viking_det16_masks.json", "w") as f:
        json.dump(masks, f, indent=1)
    print("Wrote viking_det16_masks.json")

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from numba import njit, prange
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ------------------------------------------------------------------
# Load catalog
# ------------------------------------------------------------------
cat = pd.read_parquet('/Users/sp624AA/Downloads/waves_data/reduced_cats/WAVES_N_Z_2125_reduced.parquet')
cat = cat[(cat['starmask'] == 0) & (cat['duplicate'] == 0) & (cat['class'] != 'artefact')]
cat = cat[(cat['mag_Zt'] > 20) & (cat['mag_Zt'] < 21)]

ra = cat['RAmax'].values.astype(np.float64)    # <- adjust column name if needed
dec = cat['Decmax'].values.astype(np.float64)  # <- adjust column name if needed

# ------------------------------------------------------------------
# Load region masks and filter to dec > -10
# ------------------------------------------------------------------
with open("viking_det16_masks_all.json") as f:
    masks = json.load(f)

masks = [m for m in masks if m['dec_min'] > -10]
masks = [m for m in masks if m['ra_min'] > 150]
masks = [m for m in masks if m['ra_min'] < 227]
print(f"{len(masks)} chip-16 masks remain after dec > -10 filter")

# ------------------------------------------------------------------
# Group overlapping chip-16 masks into continuous "pawprint" regions
# ------------------------------------------------------------------
def build_pawprint_regions(masks, min_frames=2):
    n = len(masks)
    ra_min = np.array([m['ra_min'] for m in masks])
    ra_max = np.array([m['ra_max'] for m in masks])
    dec_min = np.array([m['dec_min'] for m in masks])
    dec_max = np.array([m['dec_max'] for m in masks])
    ra_c = (ra_min + ra_max) / 2
    dec_c = (dec_min + dec_max) / 2
    width = ra_max - ra_min
    height = dec_max - dec_min

    dec0 = np.median(dec_c)
    cosdec0 = np.cos(np.deg2rad(dec0))

    x = ra_c * cosdec0
    y = dec_c
    tree = cKDTree(np.column_stack([x, y]))

    search_radius = 1.5 * max(width.max(), height.max())  # generous, catches full pawprint

    rows, cols = [], []
    for i in tqdm(range(n), desc="Finding overlapping chip-16 pairs"):
        cand = tree.query_ball_point([x[i], y[i]], r=search_radius)
        for j in cand:
            if j <= i:
                continue
            ix0, ix1 = ra_min[i] * cosdec0, ra_max[i] * cosdec0
            jx0, jx1 = ra_min[j] * cosdec0, ra_max[j] * cosdec0
            overlap_x = ix0 < jx1 and jx0 < ix1
            overlap_y = dec_min[i] < dec_max[j] and dec_min[j] < dec_max[i]
            if overlap_x and overlap_y:
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)

    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)

    regions = []
    frame_counts = []
    for comp_id in range(n_components):
        members = np.where(labels == comp_id)[0]
        frame_counts.append(len(members))
        if len(members) < min_frames:
            continue  # drop isolated/edge exposures with no dither partners

        # unwrap RA within this component relative to its first member's centre
        ref_ra = ra_c[members[0]]
        ra_min_unw = ((ra_min[members] - ref_ra + 180) % 360) - 180 + ref_ra
        ra_max_unw = ((ra_max[members] - ref_ra + 180) % 360) - 180 + ref_ra

        regions.append({
            "ra_min": ra_min_unw.min() % 360,
            "ra_max": ra_max_unw.max() % 360,
            "dec_min": dec_min[members].min(),
            "dec_max": dec_max[members].max(),
            "n_frames": len(members),
        })

    frame_counts = np.array(frame_counts)
    print(f"{n_components} connected components found")
    print(f"Frame-count distribution: {np.bincount(frame_counts)}")
    print(f"{len(regions)} pawprint regions kept (>= {min_frames} overlapping frames)")

    return regions


pawprints = build_pawprint_regions(masks, min_frames=2)

# ------------------------------------------------------------------
# Numba kernels: offsets normalized by each region's own bbox size
# ------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _count_in_regions(ra, dec, ra_min_arr, ra_max_arr, dec_min_arr, dec_max_arr, extent_factor):
    n_regions = ra_min_arr.shape[0]
    n_sources = ra.shape[0]
    counts = np.zeros(n_regions, dtype=np.int64)

    for i in prange(n_regions):
        ra_min = ra_min_arr[i]; ra_max = ra_max_arr[i]
        dec_min = dec_min_arr[i]; dec_max = dec_max_arr[i]

        width = ra_max - ra_min
        height = dec_max - dec_min
        ra_c = (ra_min + ra_max) / 2.0
        dec_c = (dec_min + dec_max) / 2.0
        cosdec = np.cos(np.deg2rad(dec_c))

        half_w = (width * extent_factor / 2.0) * cosdec
        half_h = height * extent_factor / 2.0

        c = 0
        for j in range(n_sources):
            dra = ((ra[j] - ra_c + 180.0) % 360.0) - 180.0
            dra *= cosdec
            ddec = dec[j] - dec_c
            if abs(dra) <= half_w and abs(ddec) <= half_h:
                c += 1
        counts[i] = c

    return counts


@njit(parallel=True, fastmath=True)
def _fill_offsets_normalized(ra, dec, ra_min_arr, ra_max_arr, dec_min_arr, dec_max_arr,
                              starts, dra_out, ddec_out, extent_factor):
    n_regions = ra_min_arr.shape[0]
    n_sources = ra.shape[0]

    for i in prange(n_regions):
        ra_min = ra_min_arr[i]; ra_max = ra_max_arr[i]
        dec_min = dec_min_arr[i]; dec_max = dec_max_arr[i]

        width = ra_max - ra_min
        height = dec_max - dec_min
        ra_c = (ra_min + ra_max) / 2.0
        dec_c = (dec_min + dec_max) / 2.0
        cosdec = np.cos(np.deg2rad(dec_c))

        half_w = (width * extent_factor / 2.0) * cosdec
        half_h = height * extent_factor / 2.0

        pos = starts[i]
        for j in range(n_sources):
            dra = ((ra[j] - ra_c + 180.0) % 360.0) - 180.0
            dra *= cosdec
            ddec = dec[j] - dec_c
            if abs(dra) <= half_w and abs(ddec) <= half_h:
                dra_out[pos] = dra / width
                ddec_out[pos] = ddec / height
                pos += 1


def stack_offsets_numba(ra, dec, regions, extent_factor=2.0, batch_size=50):
    ra = np.ascontiguousarray(ra, dtype=np.float64)
    dec = np.ascontiguousarray(dec, dtype=np.float64)
    ra_min_arr = np.array([r['ra_min'] for r in regions], dtype=np.float64)
    ra_max_arr = np.array([r['ra_max'] for r in regions], dtype=np.float64)
    dec_min_arr = np.array([r['dec_min'] for r in regions], dtype=np.float64)
    dec_max_arr = np.array([r['dec_max'] for r in regions], dtype=np.float64)

    n_regions = len(regions)
    batch_edges = list(range(0, n_regions, batch_size)) + [n_regions]

    counts = np.empty(n_regions, dtype=np.int64)
    for b in tqdm(range(len(batch_edges) - 1), desc="Counting sources per region"):
        lo, hi = batch_edges[b], batch_edges[b + 1]
        counts[lo:hi] = _count_in_regions(
            ra, dec,
            ra_min_arr[lo:hi], ra_max_arr[lo:hi],
            dec_min_arr[lo:hi], dec_max_arr[lo:hi],
            extent_factor
        )

    starts = np.zeros(n_regions, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    total = counts.sum()
    dra_out = np.empty(total, dtype=np.float64)
    ddec_out = np.empty(total, dtype=np.float64)

    for b in tqdm(range(len(batch_edges) - 1), desc="Collecting offsets"):
        lo, hi = batch_edges[b], batch_edges[b + 1]
        batch_starts = starts[lo:hi] - starts[lo]
        batch_total = int(counts[lo:hi].sum())
        batch_dra = np.empty(batch_total, dtype=np.float64)
        batch_ddec = np.empty(batch_total, dtype=np.float64)

        _fill_offsets_normalized(
            ra, dec,
            ra_min_arr[lo:hi], ra_max_arr[lo:hi],
            dec_min_arr[lo:hi], dec_max_arr[lo:hi],
            batch_starts, batch_dra, batch_ddec,
            extent_factor
        )

        dra_out[starts[lo]:starts[lo] + batch_total] = batch_dra
        ddec_out[starts[lo]:starts[lo] + batch_total] = batch_ddec

    return dra_out, ddec_out


# ------------------------------------------------------------------
# Plot 1: stacked density, normalized to each pawprint's bounding box
# ------------------------------------------------------------------
extent_factor = 4.0
half_extent = extent_factor / 2.0

all_dra, all_ddec = stack_offsets_numba(ra, dec, pawprints, extent_factor=extent_factor, batch_size=50)

extent = [-half_extent, half_extent, -half_extent, half_extent]
H, xedges, yedges = np.histogram2d(
    all_dra, all_ddec, bins=100,
    range=[[-half_extent, half_extent], [-half_extent, half_extent]]
)

np.savez(
    "stacked_density_hist.npz",
    H=H, xedges=xedges, yedges=yedges, extent=np.array(extent)
)
print("Saved histogram data to stacked_density_hist.npz")

nonzero = H[H > 0]
vmin = nonzero.min() if nonzero.size else 1
vmax = H.max()

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    H.T, origin='lower', extent=extent, aspect='auto',
    norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno'
)

# outline of the continuous pawprint bounding box (always unit square,
# since every region is normalized to its own bbox)
pawprint_outline = Rectangle(
    (-0.5, -0.5), 1.0, 1.0,
    edgecolor='black', facecolor='none', linewidth=1.5
)
ax.add_patch(pawprint_outline)

ax.set_xlabel(r'$\Delta$RA / pawprint width')
ax.set_ylabel(r'$\Delta$Dec / pawprint height')
ax.set_aspect('equal')
ax.set_title('Stacked 20<Z_VISTA<21 sources\nCentred on Detector 16 Imaged Regions, WWN')
fig.colorbar(im, ax=ax, label='N sources (log scale)')
#plt.tight_layout()
plt.savefig("wwn_chip16_stacked_density.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# Plot 2: sky footprint — individual chips (thin) + pawprint bboxes (black)
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

ax.hexbin(ra, dec, gridsize=300, mincnt=1, cmap='Greys', bins='log')

for m in masks:
    rect = Rectangle(
        (m['ra_min'], m['dec_min']), m['ra_max'] - m['ra_min'], m['dec_max'] - m['dec_min'],
        edgecolor='red', facecolor='none', linewidth=0.4, alpha=0.5
    )
    ax.add_patch(rect)

for r in pawprints:
    rect = Rectangle(
        (r['ra_min'], r['dec_min']), r['ra_max'] - r['ra_min'], r['dec_max'] - r['dec_min'],
        edgecolor='black', facecolor='none', linewidth=1.0, alpha=0.9
    )
    ax.add_patch(rect)

ax.set_xlabel('RA (deg)')
ax.set_ylabel('Dec (deg)')
ax.set_title('Footprint: sources, individual chip-16 frames (red), pawprints (black)')
ax.invert_xaxis()
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("wwn_chip16_footprint.png", dpi=150)
plt.show()

# %%
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from numba import njit, prange
from tqdm import tqdm
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

# ------------------------------------------------------------------
# RA wraparound handling
# ------------------------------------------------------------------
def circular_mean_deg(ra_deg):
    """Wrap-safe mean RA, used as the reference point for unwrapping."""
    rad = np.deg2rad(ra_deg)
    mean_sin = np.mean(np.sin(rad))
    mean_cos = np.mean(np.cos(rad))
    return np.rad2deg(np.arctan2(mean_sin, mean_cos)) % 360.0


def unwrap_ra(ra_deg, ref):
    """Shift RA values onto a contiguous line centred on `ref`, removing
    the 0/360 seam. Output is NOT wrapped back to [0, 360) — it stays
    contiguous, e.g. could run from -20 to 340. That's fine for all
    downstream math and plotting; only re-wrap if you need conventional
    RA labels/output."""
    ra_deg = np.asarray(ra_deg, dtype=np.float64)
    return ((ra_deg - ref + 180.0) % 360.0) - 180.0 + ref


# ------------------------------------------------------------------
# Load catalog
# ------------------------------------------------------------------
cat = pd.read_parquet('/Users/sp624AA/Downloads/waves_data/reduced_cats/WAVES_S_Z_2125_reduced.parquet')
cat = cat[(cat['starmask'] == 0) & (cat['duplicate'] == 0) & (cat['class'] != 'artefact')]
cat = cat[(cat['mag_Zt'] > 20) & (cat['mag_Zt'] < 21)]

ra_raw = cat['RAmax'].values.astype(np.float64)    # <- adjust column name if needed
dec = cat['Decmax'].values.astype(np.float64)      # <- adjust column name if needed

# ------------------------------------------------------------------
# Load region masks and filter to dec < -10
# ------------------------------------------------------------------
with open("viking_det16_masks_all.json") as f:
    masks_raw = json.load(f)

masks_raw = [m for m in masks_raw if m['dec_min'] < -10]
print(f"{len(masks_raw)} chip-16 masks remain after dec < -10 filter")

# ------------------------------------------------------------------
# Determine a single wrap-safe RA reference from ALL relevant RA
# values (catalog + mask corners), then unwrap everything onto one
# contiguous coordinate. This removes the 0/360 seam for every piece
# of downstream code (KD-tree, overlap tests, plotting).
# ------------------------------------------------------------------
mask_ra_values = np.concatenate([
    np.array([m['ra_min'] for m in masks_raw]),
    np.array([m['ra_max'] for m in masks_raw]),
])
ra_ref = circular_mean_deg(np.concatenate([ra_raw, mask_ra_values]))
print(f"RA unwrap reference: {ra_ref:.4f} deg")

ra = unwrap_ra(ra_raw, ra_ref)

masks = []
for m in masks_raw:
    ra_min_u = unwrap_ra(np.array([m['ra_min']]), ra_ref)[0]
    ra_max_u = unwrap_ra(np.array([m['ra_max']]), ra_ref)[0]
    # guard against a chip whose own bbox straddles the seam and
    # came out ra_min > ra_max after independent unwrapping
    if ra_min_u > ra_max_u:
        ra_min_u, ra_max_u = ra_max_u, ra_min_u
    masks.append({
        **m,
        "ra_min": ra_min_u,
        "ra_max": ra_max_u,
    })

# ------------------------------------------------------------------
# Group overlapping chip-16 masks into continuous "pawprint" regions
# ------------------------------------------------------------------
def build_pawprint_regions(masks, min_frames=2):
    n = len(masks)
    ra_min = np.array([m['ra_min'] for m in masks])
    ra_max = np.array([m['ra_max'] for m in masks])
    dec_min = np.array([m['dec_min'] for m in masks])
    dec_max = np.array([m['dec_max'] for m in masks])
    ra_c = (ra_min + ra_max) / 2
    dec_c = (dec_min + dec_max) / 2
    width = ra_max - ra_min
    height = dec_max - dec_min

    dec0 = np.median(dec_c)
    cosdec0 = np.cos(np.deg2rad(dec0))

    # RA is already unwrapped/contiguous at this point, so a plain
    # cos(dec)-scaled Cartesian KD-tree is safe.
    x = ra_c * cosdec0
    y = dec_c
    tree = cKDTree(np.column_stack([x, y]))

    search_radius = 1.5 * max(width.max(), height.max())

    rows, cols = [], []
    for i in tqdm(range(n), desc="Finding overlapping chip-16 pairs"):
        cand = tree.query_ball_point([x[i], y[i]], r=search_radius)
        for j in cand:
            if j <= i:
                continue
            ix0, ix1 = ra_min[i] * cosdec0, ra_max[i] * cosdec0
            jx0, jx1 = ra_min[j] * cosdec0, ra_max[j] * cosdec0
            overlap_x = ix0 < jx1 and jx0 < ix1
            overlap_y = dec_min[i] < dec_max[j] and dec_min[j] < dec_max[i]
            if overlap_x and overlap_y:
                rows.append(i); cols.append(j)
                rows.append(j); cols.append(i)

    adj = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    n_components, labels = connected_components(adj, directed=False)

    regions = []
    frame_counts = []
    for comp_id in range(n_components):
        members = np.where(labels == comp_id)[0]
        frame_counts.append(len(members))
        if len(members) < min_frames:
            continue

        # RA is already globally unwrapped, so a plain min/max over
        # members is safe here (no per-component re-unwrapping needed)
        regions.append({
            "ra_min": ra_min[members].min(),
            "ra_max": ra_max[members].max(),
            "dec_min": dec_min[members].min(),
            "dec_max": dec_max[members].max(),
            "n_frames": len(members),
        })

    frame_counts = np.array(frame_counts)
    print(f"{n_components} connected components found")
    print(f"Frame-count distribution: {np.bincount(frame_counts)}")
    print(f"{len(regions)} pawprint regions kept (>= {min_frames} overlapping frames)")

    return regions


pawprints = build_pawprint_regions(masks, min_frames=2)

# ------------------------------------------------------------------
# Numba kernels: offsets normalized by each region's own bbox size.
# The per-source dra wrap trick (((x+180)%360)-180) is kept as a
# safety net even though RA is now globally unwrapped — it's a no-op
# when there's no seam nearby, and free insurance if there is.
# ------------------------------------------------------------------
@njit(parallel=True, fastmath=True)
def _count_in_regions(ra, dec, ra_min_arr, ra_max_arr, dec_min_arr, dec_max_arr, extent_factor):
    n_regions = ra_min_arr.shape[0]
    n_sources = ra.shape[0]
    counts = np.zeros(n_regions, dtype=np.int64)

    for i in prange(n_regions):
        ra_min = ra_min_arr[i]; ra_max = ra_max_arr[i]
        dec_min = dec_min_arr[i]; dec_max = dec_max_arr[i]

        width = ra_max - ra_min
        height = dec_max - dec_min
        ra_c = (ra_min + ra_max) / 2.0
        dec_c = (dec_min + dec_max) / 2.0

        half_w = width * extent_factor / 2.0
        half_h = height * extent_factor / 2.0

        c = 0
        for j in range(n_sources):
            dra = ((ra[j] - ra_c + 180.0) % 360.0) - 180.0
            ddec = dec[j] - dec_c
            if abs(dra) <= half_w and abs(ddec) <= half_h:
                c += 1
        counts[i] = c

    return counts


@njit(parallel=True, fastmath=True)
def _fill_offsets_normalized(ra, dec, ra_min_arr, ra_max_arr, dec_min_arr, dec_max_arr,
                              starts, dra_out, ddec_out, extent_factor):
    n_regions = ra_min_arr.shape[0]
    n_sources = ra.shape[0]

    for i in prange(n_regions):
        ra_min = ra_min_arr[i]; ra_max = ra_max_arr[i]
        dec_min = dec_min_arr[i]; dec_max = dec_max_arr[i]

        width = ra_max - ra_min
        height = dec_max - dec_min
        ra_c = (ra_min + ra_max) / 2.0
        dec_c = (dec_min + dec_max) / 2.0

        half_w = width * extent_factor / 2.0
        half_h = height * extent_factor / 2.0

        pos = starts[i]
        for j in range(n_sources):
            dra = ((ra[j] - ra_c + 180.0) % 360.0) - 180.0
            ddec = dec[j] - dec_c
            if abs(dra) <= half_w and abs(ddec) <= half_h:
                dra_out[pos] = dra / width
                ddec_out[pos] = ddec / height
                pos += 1


def stack_offsets_numba(ra, dec, regions, extent_factor=2.0, batch_size=50):
    ra = np.ascontiguousarray(ra, dtype=np.float64)
    dec = np.ascontiguousarray(dec, dtype=np.float64)
    ra_min_arr = np.array([r['ra_min'] for r in regions], dtype=np.float64)
    ra_max_arr = np.array([r['ra_max'] for r in regions], dtype=np.float64)
    dec_min_arr = np.array([r['dec_min'] for r in regions], dtype=np.float64)
    dec_max_arr = np.array([r['dec_max'] for r in regions], dtype=np.float64)

    n_regions = len(regions)
    batch_edges = list(range(0, n_regions, batch_size)) + [n_regions]

    counts = np.empty(n_regions, dtype=np.int64)
    for b in tqdm(range(len(batch_edges) - 1), desc="Counting sources per region"):
        lo, hi = batch_edges[b], batch_edges[b + 1]
        counts[lo:hi] = _count_in_regions(
            ra, dec,
            ra_min_arr[lo:hi], ra_max_arr[lo:hi],
            dec_min_arr[lo:hi], dec_max_arr[lo:hi],
            extent_factor
        )

    starts = np.zeros(n_regions, dtype=np.int64)
    starts[1:] = np.cumsum(counts)[:-1]
    total = counts.sum()
    dra_out = np.empty(total, dtype=np.float64)
    ddec_out = np.empty(total, dtype=np.float64)

    for b in tqdm(range(len(batch_edges) - 1), desc="Collecting offsets"):
        lo, hi = batch_edges[b], batch_edges[b + 1]
        batch_starts = starts[lo:hi] - starts[lo]
        batch_total = int(counts[lo:hi].sum())
        batch_dra = np.empty(batch_total, dtype=np.float64)
        batch_ddec = np.empty(batch_total, dtype=np.float64)

        _fill_offsets_normalized(
            ra, dec,
            ra_min_arr[lo:hi], ra_max_arr[lo:hi],
            dec_min_arr[lo:hi], dec_max_arr[lo:hi],
            batch_starts, batch_dra, batch_ddec,
            extent_factor
        )

        dra_out[starts[lo]:starts[lo] + batch_total] = batch_dra
        ddec_out[starts[lo]:starts[lo] + batch_total] = batch_ddec

    return dra_out, ddec_out


# ------------------------------------------------------------------
# Plot 1: stacked density, normalized to each pawprint's bounding box
# ------------------------------------------------------------------
extent_factor = 3.0
half_extent = extent_factor / 2.0

all_dra, all_ddec = stack_offsets_numba(ra, dec, pawprints, extent_factor=extent_factor, batch_size=50)

extent = [-half_extent, half_extent, -half_extent, half_extent]
H, xedges, yedges = np.histogram2d(
    all_dra, all_ddec, bins=100,
    range=[[-half_extent, half_extent], [-half_extent, half_extent]]
)

np.savez(
    "stacked_density_hist.npz",
    H=H, xedges=xedges, yedges=yedges, extent=np.array(extent)
)
print("Saved histogram data to stacked_density_hist.npz")

nonzero = H[H > 0]
vmin = nonzero.min() if nonzero.size else 1
vmax = H.max()

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(
    H.T, origin='lower', extent=extent, aspect='auto',
    norm=LogNorm(vmin=vmin, vmax=vmax), cmap='inferno'
)

pawprint_outline = Rectangle(
    (-0.5, -0.5), 1.0, 1.0,
    edgecolor='black', facecolor='none', linewidth=1.5
)
ax.add_patch(pawprint_outline)

ax.set_xlabel(r'$\Delta$RA / pawprint width')
ax.set_ylabel(r'$\Delta$Dec / pawprint height')
ax.set_aspect('equal')
ax.set_title('Stacked 20<Z_VISTA<21 sources\nCentred on Detector 16 Imaged Regions, WWS')
fig.colorbar(im, ax=ax, label='N sources (log scale)')
plt.savefig("wws_chip16_stacked_density.png", dpi=150)
plt.show()

# ------------------------------------------------------------------
# Plot 2: sky footprint — individual chips (thin) + pawprint bboxes (black)
# Uses the unwrapped RA throughout, so the seam no longer splits the plot.
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 8))

ax.hexbin(ra, dec, gridsize=300, mincnt=1, cmap='Greys', bins='log')

for m in masks:
    rect = Rectangle(
        (m['ra_min'], m['dec_min']), m['ra_max'] - m['ra_min'], m['dec_max'] - m['dec_min'],
        edgecolor='red', facecolor='none', linewidth=0.4, alpha=0.5
    )
    ax.add_patch(rect)

for r in pawprints:
    rect = Rectangle(
        (r['ra_min'], r['dec_min']), r['ra_max'] - r['ra_min'], r['dec_max'] - r['dec_min'],
        edgecolor='black', facecolor='none', linewidth=1.0, alpha=0.9
    )
    ax.add_patch(rect)

ax.set_xlabel('RA (deg, unwrapped — see note below)')
ax.set_ylabel('Dec (deg)')
ax.set_title('Footprint: sources, individual chip-16 frames (red), pawprints (black)')
ax.invert_xaxis()
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig("wws_chip16_footprint.png", dpi=150)
plt.show()


