"""
Real-space 3D correlation function xi(r) for the Magneticum lightcone mock.

 - Positions use the *true* (cosmological) redshift `z_true`, so this is xi(r)
   directly (no redshift-space distortions).
 - Magnitude cut on `mdust_z` (dust-attenuated apparent z-band):
        mdust_z < 21.1
        18.5 < mdust_z < 19.5
        19.5 < mdust_z < 20.5
 - Randoms: uniform on the sky inside the RA/Dec rectangle defined by the
   min/max of the catalogue (uniform in sin(dec), so the area change with
   declination is accounted for). Redshifts are resampled from the data n(z)
   of each sample.
 - Landy-Szalay estimator (Corrfunc) + delete-one jackknife errors
   (KMeans regions on the sphere).

Dependencies:  pip install Corrfunc astropy scikit-learn pandas pyarrow numpy scipy

Usage:
    python measure_xi_r_magneticum.py --outdir ./xi_output
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from astropy.cosmology import FlatLambdaCDM
from sklearn.cluster import KMeans

from Corrfunc.theory import DD
from Corrfunc.utils import convert_3d_counts_to_cf


PARQUET = '/mnt/lustre/projects/astro/general/sp624/magneticum/galaxies.parquet'

MAG_COL = 'm_z'
Z_COL = 'z_true'

# (mag_min, mag_max); None = no limit on that side.  mag_min <= m < mag_max
SAMPLES = [
    (None, 21.1),
    (18.5, 19.5),
    (19.5, 20.5),

]


# --------------------------------------------------------------------------
# Cosmology  (CHECK: Magneticum is WMAP7, Om0 ~ 0.272, h = 0.704)
# H0 = 100 so distances come out in Mpc/h.
# --------------------------------------------------------------------------
def get_cosmology():
    return FlatLambdaCDM(H0=100.0, Om0=0.272)


def to_cartesian(ra_deg, dec_deg, z, cosmo):
    ra = np.radians(ra_deg)
    dec = np.radians(dec_deg)
    d_c = cosmo.comoving_distance(z).value
    return (d_c * np.cos(dec) * np.cos(ra),
            d_c * np.cos(dec) * np.sin(ra),
            d_c * np.sin(dec))


def compute_z_eff(z_gal, cosmo, ngrid=2000):
    """Effective redshift: int z N^2/(dV/dz) dz / int N^2/(dV/dz) dz."""
    z_gal = np.asarray(z_gal, dtype=np.float64)
    z = np.linspace(z_gal.min(), z_gal.max(), ngrid)
    Nz = gaussian_kde(z_gal)(z)
    dVdz = cosmo.differential_comoving_volume(z).value
    w = Nz**2 / dVdz
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    return float(trapz(z * w, z) / trapz(w, z))


# --------------------------------------------------------------------------
# Footprint: rectangle from min/max (handles RA wrap)
# --------------------------------------------------------------------------
def get_rectangle(ra, dec):
    """Return (ra_lo, width, dec_lo, dec_hi). RA is handled relative to the
    circular mean so a field straddling RA=0/360 still gives a sensible box.
    A sample's RA is mapped to ra_rel = ((ra - ra_ref + 180) % 360) - 180."""
    ra = np.asarray(ra, dtype=np.float64)
    r = np.radians(ra)
    ra_ref = np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) % 360
    ra_rel = ((ra - ra_ref + 180.0) % 360.0) - 180.0
    return dict(ra_ref=ra_ref, rel_min=ra_rel.min(), rel_max=ra_rel.max(),
                dec_min=float(np.min(dec)), dec_max=float(np.max(dec)))


def sample_randoms_uniform(n, box, rng):
    ra_rel = rng.uniform(box['rel_min'], box['rel_max'], n)
    ra = (ra_rel + box['ra_ref']) % 360.0
    s0 = np.sin(np.radians(box['dec_min']))
    s1 = np.sin(np.radians(box['dec_max']))
    dec = np.degrees(np.arcsin(rng.uniform(s0, s1, n)))   # uniform in sin(dec)
    return ra, dec


def generate_randoms(data, box, factor, seed):
    rng = np.random.default_rng(seed)
    n = int(factor * len(data))
    ra, dec = sample_randoms_uniform(n, box, rng)
    z = rng.choice(data[Z_COL].values, size=n, replace=True)  # data n(z)
    return pd.DataFrame({'ra': ra, 'dec': dec, Z_COL: z})


# --------------------------------------------------------------------------
# Jackknife regions
# --------------------------------------------------------------------------
def _unit_vec(ra, dec):
    ra_r = np.radians(np.asarray(ra, dtype=np.float64))
    dec_r = np.radians(np.asarray(dec, dtype=np.float64))
    return np.column_stack([np.cos(dec_r) * np.cos(ra_r),
                            np.cos(dec_r) * np.sin(ra_r),
                            np.sin(dec_r)])


def assign_jackknife_regions(d_ra, d_dec, r_ra, r_dec, n_regions, seed):
    km = KMeans(n_clusters=n_regions, random_state=seed, n_init=10)
    km.fit(_unit_vec(d_ra, d_dec))
    return km.labels_, km.predict(_unit_vec(r_ra, r_dec))


# --------------------------------------------------------------------------
# xi(r)
# --------------------------------------------------------------------------
def compute_xi(data_xyz, rand_xyz, r_edges, nthreads):
    dx, dy, dz = data_xyz
    rx, ry, rz = rand_xyz
    ND, NR = len(dx), len(rx)
    dd = DD(1, nthreads, r_edges, dx, dy, dz, periodic=False)
    dr = DD(0, nthreads, r_edges, dx, dy, dz, X2=rx, Y2=ry, Z2=rz, periodic=False)
    rr = DD(1, nthreads, r_edges, rx, ry, rz, periodic=False)
    return np.asarray(convert_3d_counts_to_cf(ND, ND, NR, NR, dd, dr, dr, rr,
                                              estimator='LS'))


def run_sample(df, box, mag_min, mag_max, out_dir, r_edges,
               factor, nthreads, n_jk, seed):
    lo = 'x' if mag_min is None else f'{mag_min:.2f}'
    hi = 'x' if mag_max is None else f'{mag_max:.2f}'
    tag = f'{MAG_COL}_{lo}-{hi}'

    m = np.ones(len(df), dtype=bool)
    if mag_min is not None:
        m &= df[MAG_COL].values >= mag_min
    if mag_max is not None:
        m &= df[MAG_COL].values < mag_max
    data = df.loc[m].reset_index(drop=True)
    print(f"[{tag}] {len(data)} galaxies")
    if len(data) == 0:
        return None

    randoms = generate_randoms(data, box, factor, seed)
    cosmo = get_cosmology()
    z_eff = compute_z_eff(data[Z_COL].values, cosmo)
    print(f"[{tag}] {len(randoms)} randoms; z_eff = {z_eff:.4f}, "
          f"median z = {np.median(data[Z_COL]):.4f}")

    dx, dy, dz = to_cartesian(data['ra_deg'].values, data['dec_deg'].values,
                              data[Z_COL].values, cosmo)
    rx, ry, rz = to_cartesian(randoms['ra'].values, randoms['dec'].values,
                              randoms[Z_COL].values, cosmo)

    xi_full = compute_xi((dx, dy, dz), (rx, ry, rz), r_edges, nthreads)

    d_lab, r_lab = assign_jackknife_regions(
        data['ra_deg'].values, data['dec_deg'].values,
        randoms['ra'].values, randoms['dec'].values, n_jk, seed)

    nb = len(r_edges) - 1
    xi_jk = np.zeros((n_jk, nb))
    for i in range(n_jk):
        kd, kr = d_lab != i, r_lab != i
        xi_jk[i] = compute_xi((dx[kd], dy[kd], dz[kd]),
                              (rx[kr], ry[kr], rz[kr]), r_edges, nthreads)
        print(f"[{tag}] jackknife {i+1}/{n_jk} done")

    diff = xi_jk - xi_jk.mean(axis=0)
    cov = (n_jk - 1) / n_jk * (diff.T @ diff)
    err = np.sqrt(np.diag(cov))
    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])

    np.savez(os.path.join(out_dir, f'xi_r_{tag}.npz'),
             r_edges=r_edges, r_centers=r_centers, xi=xi_full,
             xi_jackknife=xi_jk, cov=cov, err=err, z_eff=z_eff,
             n_data=len(data), n_random=len(randoms),
             mag_min=np.nan if mag_min is None else mag_min,
             mag_max=np.nan if mag_max is None else mag_max)
    pd.DataFrame({'r_Mpc_h': r_centers, 'xi': xi_full, 'xi_err_jk': err,
                  'z_eff': z_eff}).to_csv(
        os.path.join(out_dir, f'xi_r_{tag}.csv'), index=False)
    print(f"[{tag}] saved")


def load_data(path):
    cols = ['ra_deg', 'dec_deg', Z_COL, MAG_COL]
    df = pd.read_parquet(path, columns=cols)
    for c in cols:
        df[c] = df[c].astype(np.float64)
    ok = np.isfinite(df[cols].values).all(axis=1) & (df[Z_COL] > 0)
    return df.loc[ok].reset_index(drop=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=PARQUET)
    p.add_argument('--outdir', default='./xi_output')
    p.add_argument('--rmin', type=float, default=0.1, help='Mpc/h')
    p.add_argument('--rmax', type=float, default=50.0, help='Mpc/h')
    p.add_argument('--nbins', type=int, default=15)
    p.add_argument('--factor', type=int, default=20, help='randoms / data')
    p.add_argument('--nthreads', type=int, default=4)
    p.add_argument('--njk', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    r_edges = np.logspace(np.log10(a.rmin), np.log10(a.rmax), a.nbins + 1)

    df = load_data(a.input)
    # Rectangle from the min/max of the whole catalogue, shared by all samples
    box = get_rectangle(df['ra_deg'].values, df['dec_deg'].values)
    print(f"Rectangle: RA {box['ra_ref'] + box['rel_min']:.3f} -> "
          f"{box['ra_ref'] + box['rel_max']:.3f} (mod 360), "
          f"Dec {box['dec_min']:.3f} -> {box['dec_max']:.3f}")

    for mag_min, mag_max in SAMPLES:
        run_sample(df, box, mag_min, mag_max, a.outdir, r_edges,
                   a.factor, a.nthreads, a.njk, a.seed)