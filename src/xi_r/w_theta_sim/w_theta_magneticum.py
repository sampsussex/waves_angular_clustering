"""
Angular correlation function w(theta) for the Magneticum lightcone mock.

 - Magnitude bins on `mdust_z`:
        16 < m < 17.5,  17.5 < m < 18.5,  18.5 < m < 19.5,
        19.5 < m < 20.5, 20.5 < m < 21.25, 21.25 < m < 22
 - Randoms: uniform on the sky inside the RA/Dec rectangle given by the
   min/max of the catalogue (uniform in sin(dec) to correct for area).
 - Landy-Szalay estimator (Corrfunc DDtheta_mocks) + delete-one jackknife
   errors (KMeans regions on the sphere).

Dependencies:  pip install Corrfunc scikit-learn pandas pyarrow numpy

Usage:
    python measure_w_theta_magneticum.py --outdir ./wtheta_output
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from Corrfunc.mocks.DDtheta_mocks import DDtheta_mocks
from Corrfunc.utils import convert_3d_counts_to_cf

PARQUET = '/mnt/lustre/projects/astro/general/sp624/magneticum/galaxies.parquet'
MAG_COL = 'm_z'

MAG_BINS = [
    (16.0, 17.5),
    (17.5, 18.5),
    (18.5, 19.5),
    (19.5, 20.5),
    (20.5, 21.25),
    (21.25, 22.0),
]


def get_rectangle(ra, dec):
    """Min/max rectangle; RA measured relative to the circular mean so a
    field straddling RA = 0/360 still works."""
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
    dec = np.degrees(np.arcsin(rng.uniform(s0, s1, n)))
    return ra, dec


def _unit_vec(ra, dec):
    ra_r = np.radians(np.asarray(ra, dtype=np.float64))
    dec_r = np.radians(np.asarray(dec, dtype=np.float64))
    return np.column_stack([np.cos(dec_r) * np.cos(ra_r),
                            np.cos(dec_r) * np.sin(ra_r),
                            np.sin(dec_r)])


def compute_w(d_ra, d_dec, r_ra, r_dec, theta_edges, nthreads):
    ND, NR = len(d_ra), len(r_ra)
    dd = DDtheta_mocks(1, nthreads, theta_edges, d_ra, d_dec)
    dr = DDtheta_mocks(0, nthreads, theta_edges, d_ra, d_dec,
                       RA2=r_ra, DEC2=r_dec)
    rr = DDtheta_mocks(1, nthreads, theta_edges, r_ra, r_dec)
    return np.asarray(convert_3d_counts_to_cf(ND, ND, NR, NR, dd, dr, dr, rr,
                                              estimator='LS'))


def run_bin(df, box, mag_min, mag_max, out_dir, theta_edges,
            factor, nthreads, n_jk, seed):
    tag = f'{MAG_COL}_{mag_min:.2f}-{mag_max:.2f}'
    m = (df[MAG_COL].values >= mag_min) & (df[MAG_COL].values < mag_max)
    data = df.loc[m]
    d_ra = np.ascontiguousarray(data['ra_deg'].values, dtype=np.float64)
    d_dec = np.ascontiguousarray(data['dec_deg'].values, dtype=np.float64)
    print(f"[{tag}] {len(d_ra)} galaxies")
    if len(d_ra) == 0:
        return

    rng = np.random.default_rng(seed)
    r_ra, r_dec = sample_randoms_uniform(int(factor * len(d_ra)), box, rng)
    print(f"[{tag}] {len(r_ra)} randoms")

    w_full = compute_w(d_ra, d_dec, r_ra, r_dec, theta_edges, nthreads)

    km = KMeans(n_clusters=n_jk, random_state=seed, n_init=10)
    km.fit(_unit_vec(d_ra, d_dec))
    d_lab = km.labels_
    r_lab = km.predict(_unit_vec(r_ra, r_dec))

    nb = len(theta_edges) - 1
    w_jk = np.zeros((n_jk, nb))
    for i in range(n_jk):
        kd, kr = d_lab != i, r_lab != i
        w_jk[i] = compute_w(d_ra[kd], d_dec[kd], r_ra[kr], r_dec[kr],
                            theta_edges, nthreads)
        print(f"[{tag}] jackknife {i+1}/{n_jk} done")

    diff = w_jk - w_jk.mean(axis=0)
    cov = (n_jk - 1) / n_jk * (diff.T @ diff)
    err = np.sqrt(np.diag(cov))
    # geometric bin centres (log-spaced bins)
    th = np.sqrt(theta_edges[:-1] * theta_edges[1:])

    np.savez(os.path.join(out_dir, f'w_theta_{tag}.npz'),
             theta_edges=theta_edges, theta_centers=th, w=w_full,
             w_jackknife=w_jk, cov=cov, err=err,
             n_data=len(d_ra), n_random=len(r_ra),
             mag_min=mag_min, mag_max=mag_max)
    pd.DataFrame({'theta_deg': th, 'w': w_full, 'w_err_jk': err}).to_csv(
        os.path.join(out_dir, f'w_theta_{tag}.csv'), index=False)
    print(f"[{tag}] saved")


def load_data(path):
    cols = ['ra_deg', 'dec_deg', MAG_COL]
    df = pd.read_parquet(path, columns=cols)
    for c in cols:
        df[c] = df[c].astype(np.float64)
    ok = np.isfinite(df[cols].values).all(axis=1)
    return df.loc[ok].reset_index(drop=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', default=PARQUET)
    p.add_argument('--outdir', default='./wtheta_output')
    p.add_argument('--thmin', type=float, default=0.01, help='deg')
    p.add_argument('--thmax', type=float, default=5.0, help='deg')
    p.add_argument('--nbins', type=int, default=15)
    p.add_argument('--factor', type=int, default=10, help='randoms / data')
    p.add_argument('--nthreads', type=int, default=4)
    p.add_argument('--njk', type=int, default=10)
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    theta_edges = np.logspace(np.log10(a.thmin), np.log10(a.thmax), a.nbins + 1)

    df = load_data(a.input)
    box = get_rectangle(df['ra_deg'].values, df['dec_deg'].values)
    print(f"Rectangle: RA {box['ra_ref'] + box['rel_min']:.3f} -> "
          f"{box['ra_ref'] + box['rel_max']:.3f} (mod 360), "
          f"Dec {box['dec_min']:.3f} -> {box['dec_max']:.3f}")

    for mag_min, mag_max in MAG_BINS:
        run_bin(df, box, mag_min, mag_max, a.outdir, theta_edges,
                a.factor, a.nthreads, a.njk, a.seed)