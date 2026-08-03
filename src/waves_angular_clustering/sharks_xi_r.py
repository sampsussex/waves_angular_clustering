"""
Measure the real-space 3D correlation function xi(r) for a WAVES-like mock
(WAVESwide or WAVESdeep selections), using real-space (redshift_cosmological)
positions, Landy-Szalay estimator, and 10-region jackknife errors.

NEW: for region='deep', you can now split the sample into Z-band apparent
magnitude slices (mag_Z_VISTA) and get xi(r) + jackknife errors for each
slice separately. This is useful for the shark/WAVES-deep mock where you
want to see how clustering evolves with (roughly) magnitude/luminosity/depth.

Dependencies:
    pip install Corrfunc astropy scikit-learn pandas pyarrow numpy

Usage:
    # full deep sample, single xi(r) (old behaviour)
    python measure_xi_r.py --input mock.parquet --region deep --outdir ./xi_output

    # full wide sample
    python measure_xi_r.py --input mock.parquet --region wide --outdir ./xi_output

    # deep sample split into the 5 standard Z-mag slices
    python measure_xi_r.py --input mock.parquet --region deep --outdir ./xi_output --magslices

    # deep sample, one custom magnitude slice
    python measure_xi_r.py --input mock.parquet --region deep --outdir ./xi_output --magmin 18.5 --magmax 19.5

NOTE ON COSMOLOGY:
    get_cosmology() below sets H0=100 so that comoving distances come out
    directly in Mpc/h. Check that Om0 (and any other params) match the
    fiducial cosmology used to build your mock -- getting this wrong will
    bias xi(r) and, more importantly, your comparison to any theory/HOD
    prediction computed in the "true" cosmology.
"""

import os
import argparse
import numpy as np
import pandas as pd

from astropy.cosmology import FlatLambdaCDM
from sklearn.cluster import KMeans

from Corrfunc.theory import DD
from Corrfunc.utils import convert_3d_counts_to_cf


# --------------------------------------------------------------------------
# Footprints
# --------------------------------------------------------------------------

# WAVES-wide sub-regions: (ra_min, ra_max, dec_min, dec_max)
WW_N = (157.25, 225.0, -3.95, 3.95)
WW_S = (330.0, 51.6, -35.6, -27.0)   # wraps through RA = 0/360

# WAVES-deep region
WD = (339.0, 351.0, -35.0, -30.0)

# Standard Z-band (mag_Z_VISTA) magnitude slices for the WAVES-deep / shark
# deep mock. Each tuple is (mag_min, mag_max), applied as mag_min <= mag < mag_max.
MAG_SLICES = [
    (16.0, 17.5),
    (17.5, 18.5),
    (18.5, 19.5),
    (19.5, 20.5),
    (20.5, 21.25),
]


def in_box(ra, dec, ra_min, ra_max, dec_min, dec_max):
    """RA/Dec box mask, correctly handling RA wraparound (ra_min > ra_max)."""
    if ra_min < ra_max:
        ra_mask = (ra >= ra_min) & (ra <= ra_max)
    else:
        ra_mask = (ra >= ra_min) | (ra <= ra_max)
    dec_mask = (dec >= dec_min) & (dec <= dec_max)
    return ra_mask & dec_mask


def in_wide_footprint(ra, dec):
    return in_box(ra, dec, *WW_N) | in_box(ra, dec, *WW_S)


def box_solid_angle(ra_min, ra_max, dec_min, dec_max):
    """Solid angle of an RA/Dec box in steradians (handles RA wrap)."""
    if ra_min < ra_max:
        dra = ra_max - ra_min
    else:
        dra = (360.0 - ra_min) + ra_max
    dra_rad = np.radians(dra)
    sin_dec_min = np.sin(np.radians(dec_min))
    sin_dec_max = np.sin(np.radians(dec_max))
    return dra_rad * (sin_dec_max - sin_dec_min)


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def apply_selection(df, region, mag_min=None, mag_max=None):
    """
    Apply the base WAVES-wide / WAVES-deep selection.

    mag_min / mag_max (optional): further restrict mag_Z_VISTA to
    [mag_min, mag_max) *on top of* the region's own magnitude limit. This is
    how the Z-magnitude slices for the deep mock are implemented -- the
    region cut (e.g. mag_Z_VISTA < 21.25 for deep) still applies, the slice
    just narrows it further.
    """
    df = df.copy()
    df['mass_stellar_total'] = np.log10(df['mass_stellar_total'])

    mask = (df['mass_stellar_total'] > 8) & (df['mag_Z_VISTA'] > -99)

    if region == 'wide':
        mask = mask & (df['mag_Z_VISTA'] < 21.1) & (df['redshift_observed'] < 0.2)
        mask = mask & in_wide_footprint(df['ra'].values, df['dec'].values)
    elif region == 'deep':
        mask = (
            mask
            & (df['mag_Z_VISTA'] < 21.25)
            & (df['dec'] > WD[2]) & (df['dec'] < WD[3])
            & (df['ra'] > WD[0]) & (df['ra'] < WD[1])
            & (df['redshift_observed'] < 0.8)
        )
    else:
        raise ValueError(f"Unknown region '{region}', must be 'wide' or 'deep'")

    if mag_min is not None:
        mask = mask & (df['mag_Z_VISTA'] >= mag_min)
    if mag_max is not None:
        mask = mask & (df['mag_Z_VISTA'] < mag_max)

    return df.loc[mask].reset_index(drop=True)


# --------------------------------------------------------------------------
# Randoms
# --------------------------------------------------------------------------

def sample_box_randoms(n, ra_min, ra_max, dec_min, dec_max, rng):
    """Uniform-on-sky sampling within an RA/Dec box (great-circle correct),
    handling RA wraparound."""
    if ra_min < ra_max:
        ra = rng.uniform(ra_min, ra_max, n)
    else:
        width = (360.0 - ra_min) + ra_max
        ra = (rng.uniform(0.0, width, n) + ra_min) % 360.0

    # uniform in sin(dec) -> uniform sky density, not uniform in dec itself
    sin_dec_min = np.sin(np.radians(dec_min))
    sin_dec_max = np.sin(np.radians(dec_max))
    sin_dec = rng.uniform(sin_dec_min, sin_dec_max, n)
    dec = np.degrees(np.arcsin(sin_dec))
    return ra, dec


def generate_randoms(data_df, region, factor=20, seed=None):
    rng = np.random.default_rng(seed)
    n_total = int(factor * len(data_df))

    if region == 'deep':
        ra, dec = sample_box_randoms(n_total, *WD, rng=rng)

    elif region == 'wide':
        omega_n = box_solid_angle(*WW_N)
        omega_s = box_solid_angle(*WW_S)
        frac_n = omega_n / (omega_n + omega_s)
        n_n = int(round(n_total * frac_n))
        n_s = n_total - n_n

        ra_n, dec_n = sample_box_randoms(n_n, *WW_N, rng=rng)
        ra_s, dec_s = sample_box_randoms(n_s, *WW_S, rng=rng)
        ra = np.concatenate([ra_n, ra_s])
        dec = np.concatenate([dec_n, dec_s])

    else:
        raise ValueError(f"Unknown region '{region}'")

    # reshuffle redshifts: resample (with replacement) from the data's own
    # n(z) to reproduce the selection function without an explicit model.
    # NOTE: for magnitude slices this means the randoms' n(z) tracks the
    # n(z) of that particular slice, which is the correct thing to do -- a
    # brighter/nearer slice and a fainter/deeper slice have different n(z)
    # and each needs its own randoms.
    z = rng.choice(data_df['redshift_cosmological'].values, size=len(ra), replace=True)

    return pd.DataFrame({'ra': ra, 'dec': dec, 'redshift_cosmological': z})


# --------------------------------------------------------------------------
# Coordinates
# --------------------------------------------------------------------------

def get_cosmology():
    # H0=100 -> comoving_distance() returns Mpc/h directly.
    # CHECK Om0 matches your mock's fiducial cosmology.
    return FlatLambdaCDM(H0=100.0, Om0=0.3121)


def to_cartesian(ra_deg, dec_deg, z, cosmo):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    d_c = cosmo.comoving_distance(z).value  # Mpc/h
    x = d_c * np.cos(dec) * np.cos(ra)
    y = d_c * np.cos(dec) * np.sin(ra)
    zc = d_c * np.sin(dec)
    return x, y, zc


# --------------------------------------------------------------------------
# Jackknife regions (KMeans on the unit sphere -- robust to RA wrap)
# --------------------------------------------------------------------------

def _unit_vec(ra, dec):
    # Force float64 regardless of input dtype. Parquet columns are often
    # float32, while the randoms (built with np.random / np.degrees /
    # np.arcsin) come out float64 -- feeding KMeans.fit() float32 data and
    # then .predict() float64 data (or vice versa) raises a Cython buffer
    # dtype mismatch, since sklearn's KMeans keeps cluster_centers_ in
    # whatever dtype it was fit on.
    ra_r = np.radians(np.asarray(ra, dtype=np.float64))
    dec_r = np.radians(np.asarray(dec, dtype=np.float64))
    return np.column_stack([
        np.cos(dec_r) * np.cos(ra_r),
        np.cos(dec_r) * np.sin(ra_r),
        np.sin(dec_r),
    ])


def assign_jackknife_regions(data_ra, data_dec, rand_ra, rand_dec, region,
                              n_regions=10, seed=42):
    """
    Returns (data_labels, rand_labels, n_regions_actual).

    'deep' is a single contiguous field, so a single KMeans run over all
    points is fine. This is re-run per magnitude slice (since each slice
    has its own galaxy positions), which is what you want -- the jackknife
    footprint should be re-tiled to the actual points being used.

    'wide' is two disjoint fields (WW-N, WW-S) separated by tens of
    degrees. Running one global KMeans over both risks an uneven N/S
    split (however many clusters happen to minimize total variance),
    which breaks the equal-sized-region assumption behind the delete-one
    jackknife covariance. Instead we force WW-N and WW-S to each get
    round(n_regions / 2) regions, clustered independently, with WW-S
    labels offset so the two fields never share a label.
    """
    if region == 'deep':
        km = KMeans(n_clusters=n_regions, random_state=seed, n_init=10)
        km.fit(_unit_vec(data_ra, data_dec))
        data_labels = km.labels_
        rand_labels = km.predict(_unit_vec(rand_ra, rand_dec))
        return data_labels, rand_labels, n_regions

    elif region == 'wide':
        n_per_field = int(round(n_regions / 2))
        n_total = 2 * n_per_field
        if n_total != n_regions:
            print(f"[wide] NOTE: requested n_jk={n_regions} is odd; forcing "
                  f"{n_per_field} regions per field -> using {n_total} "
                  f"jackknife regions total instead.")

        data_in_n = in_box(data_ra, data_dec, *WW_N)
        rand_in_n = in_box(rand_ra, rand_dec, *WW_N)
        data_in_s = ~data_in_n
        rand_in_s = ~rand_in_n

        data_labels = np.full(len(data_ra), -1, dtype=int)
        rand_labels = np.full(len(rand_ra), -1, dtype=int)

        # WW-N -> labels 0 .. n_per_field-1
        km_n = KMeans(n_clusters=n_per_field, random_state=seed, n_init=10)
        km_n.fit(_unit_vec(data_ra[data_in_n], data_dec[data_in_n]))
        data_labels[data_in_n] = km_n.labels_
        rand_labels[rand_in_n] = km_n.predict(
            _unit_vec(rand_ra[rand_in_n], rand_dec[rand_in_n])
        )

        # WW-S -> labels n_per_field .. 2*n_per_field-1
        km_s = KMeans(n_clusters=n_per_field, random_state=seed, n_init=10)
        km_s.fit(_unit_vec(data_ra[data_in_s], data_dec[data_in_s]))
        data_labels[data_in_s] = km_s.labels_ + n_per_field
        rand_labels[rand_in_s] = km_s.predict(
            _unit_vec(rand_ra[rand_in_s], rand_dec[rand_in_s])
        ) + n_per_field

        assert (data_labels >= 0).all() and (rand_labels >= 0).all(), \
            "some points were not assigned a jackknife region"
        return data_labels, rand_labels, n_total

    else:
        raise ValueError(f"Unknown region '{region}'")


def report_region_sizes(data_labels, rand_labels, n_regions):
    """Print galaxy/random counts per jackknife region as a balance check."""
    print("  jackknife region sizes (data / randoms):")
    for i in range(n_regions):
        nd = int((data_labels == i).sum())
        nr = int((rand_labels == i).sum())
        print(f"    region {i:2d}: {nd:6d} data, {nr:7d} randoms")


# --------------------------------------------------------------------------
# xi(r) via Corrfunc + Landy-Szalay
# --------------------------------------------------------------------------

def compute_xi(data_xyz, rand_xyz, r_edges, nthreads=4):
    dx, dy, dz = data_xyz
    rx, ry, rz = rand_xyz
    ND, NR = len(dx), len(rx)

    dd = DD(1, nthreads, r_edges, dx, dy, dz, periodic=False)
    dr = DD(0, nthreads, r_edges, dx, dy, dz,
            X2=rx, Y2=ry, Z2=rz, periodic=False)
    rr = DD(1, nthreads, r_edges, rx, ry, rz, periodic=False)

    xi = convert_3d_counts_to_cf(ND, ND, NR, NR, dd, dr, dr, rr, estimator='LS')
    return np.asarray(xi)


# --------------------------------------------------------------------------
# Full pipeline (single sample: full region, or one magnitude slice)
# --------------------------------------------------------------------------

def run_pipeline(df, region, out_dir, r_edges, tag=None,
                  mag_min=None, mag_max=None,
                  factor=20, nthreads=4, n_jk=10, seed=42):
    """
    df: the already-loaded parquet DataFrame (loaded once and reused across
        slices so we don't re-read the file from disk each time).
    tag: label used in output filenames / log messages, e.g. 'deep' or
         'deep_mag18.5-19.5'. Defaults to `region` if not given.
    """
    os.makedirs(out_dir, exist_ok=True)
    tag = tag or region

    data = apply_selection(df, region, mag_min=mag_min, mag_max=mag_max)
    print(f"[{tag}] {len(data)} galaxies after selection")

    if len(data) == 0:
        print(f"[{tag}] WARNING: no galaxies in this slice, skipping.")
        return None

    randoms = generate_randoms(data, region, factor=factor, seed=seed)
    print(f"[{tag}] generated {len(randoms)} randoms ({factor}x data)")

    cosmo = get_cosmology()
    dx, dy, dz = to_cartesian(data['ra'].values, data['dec'].values,
                               data['redshift_cosmological'].values, cosmo)
    rx, ry, rz = to_cartesian(randoms['ra'].values, randoms['dec'].values,
                               randoms['redshift_cosmological'].values, cosmo)

    print(f"[{tag}] computing full-sample xi(r) ...")
    xi_full = compute_xi((dx, dy, dz), (rx, ry, rz), r_edges, nthreads=nthreads)

    print(f"[{tag}] assigning {n_jk} jackknife regions ...")
    data_labels, rand_labels, n_jk_actual = assign_jackknife_regions(
        data['ra'].values, data['dec'].values,
        randoms['ra'].values, randoms['dec'].values,
        region, n_regions=n_jk, seed=seed
    )
    report_region_sizes(data_labels, rand_labels, n_jk_actual)

    nbins = len(r_edges) - 1
    xi_jk = np.zeros((n_jk_actual, nbins))
    for i in range(n_jk_actual):
        keep_d = data_labels != i
        keep_r = rand_labels != i
        xi_jk[i] = compute_xi(
            (dx[keep_d], dy[keep_d], dz[keep_d]),
            (rx[keep_r], ry[keep_r], rz[keep_r]),
            r_edges, nthreads=nthreads
        )
        print(f"[{tag}] jackknife sample {i+1}/{n_jk_actual} done "
              f"(dropped {(~keep_d).sum()} data, {(~keep_r).sum()} randoms)")

    xi_jk_mean = xi_jk.mean(axis=0)
    diff = xi_jk - xi_jk_mean
    cov = (n_jk_actual - 1) / n_jk_actual * (diff.T @ diff)
    err = np.sqrt(np.diag(cov))

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    out_npz = os.path.join(out_dir, f'xi_r_{tag}.npz')
    np.savez(out_npz,
             r_edges=r_edges, r_centers=r_centers,
             xi=xi_full, xi_jackknife=xi_jk, cov=cov, err=err,
             n_data=len(data), n_random=len(randoms),
             mag_min=np.nan if mag_min is None else mag_min,
             mag_max=np.nan if mag_max is None else mag_max)

    out_csv = os.path.join(out_dir, f'xi_r_{tag}.csv')
    pd.DataFrame({'r_Mpc_h': r_centers, 'xi': xi_full, 'xi_err_jk': err}).to_csv(out_csv, index=False)

    print(f"[{tag}] saved: {out_npz}")
    print(f"[{tag}] saved: {out_csv}")
    return {
        'tag': tag, 'mag_min': mag_min, 'mag_max': mag_max,
        'r_centers': r_centers, 'xi': xi_full, 'err': err, 'cov': cov,
        'n_data': len(data), 'n_random': len(randoms),
    }


# --------------------------------------------------------------------------
# Driver: full sample, one custom slice, or all standard mag slices
# --------------------------------------------------------------------------

def load_data(parquet_path):
    cols = ['ra', 'dec', 'redshift_cosmological', 'redshift_observed',
            'mass_stellar_total', 'mag_Z_VISTA']
    return pd.read_parquet(parquet_path, columns=cols)


def run_mag_slices(parquet_path, region, out_dir, r_edges, mag_slices=MAG_SLICES,
                    factor=20, nthreads=4, n_jk=10, seed=42):
    """
    Run the full jackknife xi(r) pipeline separately for each (mag_min, mag_max)
    slice in `mag_slices`, reusing a single load of the parquet file. Writes
    one npz/csv per slice, plus a combined summary CSV across all slices.
    """
    if region != 'deep':
        raise ValueError("Magnitude slicing is currently only wired up for "
                          "region='deep' (the shark/WAVES-deep mock).")

    df = load_data(parquet_path)
    results = []
    for mag_min, mag_max in mag_slices:
        tag = f'{region}_mag{mag_min:.2f}-{mag_max:.2f}'
        res = run_pipeline(
            df, region, out_dir, r_edges, tag=tag,
            mag_min=mag_min, mag_max=mag_max,
            factor=factor, nthreads=nthreads, n_jk=n_jk, seed=seed,
        )
        if res is not None:
            results.append(res)

    # Combined long-format summary across all slices, handy for plotting.
    rows = []
    for res in results:
        for r, xi, err in zip(res['r_centers'], res['xi'], res['err']):
            rows.append({
                'mag_min': res['mag_min'], 'mag_max': res['mag_max'],
                'n_data': res['n_data'], 'n_random': res['n_random'],
                'r_Mpc_h': r, 'xi': xi, 'xi_err_jk': err,
            })
    summary_path = os.path.join(out_dir, f'xi_r_{region}_magslices_summary.csv')
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"[{region}] saved combined slice summary: {summary_path}")

    return results


def run_single(parquet_path, region, out_dir, r_edges,
                mag_min=None, mag_max=None,
                factor=20, nthreads=4, n_jk=10, seed=42):
    """Run the pipeline once, either for the full region sample or one
    custom (mag_min, mag_max) slice."""
    df = load_data(parquet_path)
    tag = region
    if mag_min is not None or mag_max is not None:
        lo = f'{mag_min:.2f}' if mag_min is not None else 'x'
        hi = f'{mag_max:.2f}' if mag_max is not None else 'x'
        tag = f'{region}_mag{lo}-{hi}'
    return run_pipeline(
        df, region, out_dir, r_edges, tag=tag,
        mag_min=mag_min, mag_max=mag_max,
        factor=factor, nthreads=nthreads, n_jk=n_jk, seed=seed,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Measure xi(r) for a WAVES wide/deep mock.")
    p.add_argument('--input', required=True, help='Path to mock parquet file')
    p.add_argument('--region', choices=['wide', 'deep'], required=True)
    p.add_argument('--outdir', default='./xi_output')
    p.add_argument('--rmin', type=float, default=0.1, help='Mpc/h')
    p.add_argument('--rmax', type=float, default=50.0, help='Mpc/h')
    p.add_argument('--nbins', type=int, default=15)
    p.add_argument('--factor', type=int, default=20, help='random-to-data ratio')
    p.add_argument('--nthreads', type=int, default=4)
    p.add_argument('--njk', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--magslices', action='store_true',
                    help="Deep only: run all 5 standard Z-mag slices "
                         "(16-17.5, 17.5-18.5, 18.5-19.5, 19.5-20.5, 20.5-21.25) "
                         "instead of the full sample.")
    p.add_argument('--magmin', type=float, default=None,
                    help='Optional: run a single custom mag_Z_VISTA slice, lower bound.')
    p.add_argument('--magmax', type=float, default=None,
                    help='Optional: run a single custom mag_Z_VISTA slice, upper bound.')

    args = p.parse_args()

    r_edges = np.logspace(np.log10(args.rmin), np.log10(args.rmax), args.nbins + 1)

    if args.magslices:
        if args.region != 'deep':
            raise SystemExit("--magslices is only supported for --region deep")
        run_mag_slices(
            args.input, args.region, args.outdir, r_edges,
            factor=args.factor, nthreads=args.nthreads,
            n_jk=args.njk, seed=args.seed,
        )
    else:
        run_single(
            args.input, args.region, args.outdir, r_edges,
            mag_min=args.magmin, mag_max=args.magmax,
            factor=args.factor, nthreads=args.nthreads,
            n_jk=args.njk, seed=args.seed,
        )