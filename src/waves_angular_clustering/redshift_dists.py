import numpy as np
import mpmath
import pandas as pd
from scipy.optimize import curve_fit
from scipy.special import gamma
from astropy.cosmology import LambdaCDM
from astropy import units as u
import matplotlib.pyplot as plt


def find_adaptive_zmax(model, mlo, mhi, tol=1e-4, z_start=0.2, growth=1.5, cap=2.0, n_probe=1000):
    """
    Find the SMALLEST z_max (starting from z_start, growing by `growth`
    each step) such that the mass beyond 0.9*z_max is a negligible
    fraction (tol) of the slice's total predicted mass -- i.e. find the
    grid extent that actually CONTAINS the distribution, without
    assuming any particular slice needs a wide range. z_start is
    intentionally small (0.2) so narrow, bright-slice distributions
    settle on a correspondingly small z_max (better peak resolution for
    a fixed n_points) rather than always growing outward from a
    z_start=1.0 floor regardless of whether that slice needed it.
    """
    z_max = z_start
    for _ in range(40):
        z_probe = np.linspace(1e-4, z_max, n_probe)
        dNdz = model.predict_dNdz_slice(z_probe, mlo, mhi, area_deg2=1.0)
        total = np.trapezoid(dNdz, z_probe)
        if total <= 0:
            z_max *= growth
            continue
        tail_mass = np.trapezoid(
            dNdz[z_probe >= 0.9 * z_max], z_probe[z_probe >= 0.9 * z_max]
        )
        if tail_mass / total < tol or z_max >= cap:
            return min(z_max, cap)
        z_max *= growth
    return z_max


# =====================================================================
# SHARKS mock loading (WAVES-Deep-sized region, complete to z = 0.8)
# =====================================================================
WD = (339.0, 351.0, -35.0, -30.0)   # (ra_min, ra_max, dec_min, dec_max) deg


def survey_area_deg2(ra_min, ra_max, dec_min, dec_max):
    """Exact spherical area of an RA/Dec rectangle, in deg^2."""
    dra = np.deg2rad(ra_max - ra_min)
    area_sr = dra * (np.sin(np.deg2rad(dec_max)) - np.sin(np.deg2rad(dec_min)))
    return area_sr * (180.0 / np.pi) ** 2


def apply_selection(df, region):
    df = df.copy()
    df['mass_stellar_total'] = np.log10(df['mass_stellar_total'])

    mask = (df['mass_stellar_total'] > 8) & (df['mag_Z_VISTA'] > -99)

    if region == 'wide':
        mask = mask & (df['mag_Z_VISTA'] < 21.1) & (df['redshift_observed'] < 0.2)
        mask = mask & in_wide_footprint(df['ra'].values, df['dec'].values)  # noqa: F821
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

    return df.loc[mask].reset_index(drop=True)


def load_sharks_mock(parquet_path, region="deep"):
    """
    Load the SHARKS lightcone mock and apply the WAVES selection.
    NOTE: the 'deep' region is complete only to z = 0.8 and Z < 21.25,
    over the WD footprint (~50.6 deg^2) -- comparisons against the
    forward model must respect both limits.
    """
    cols = ['ra', 'dec', 'redshift_cosmological', 'redshift_observed',
            'mass_stellar_total', 'mag_Z_VISTA']
    df = pd.read_parquet(parquet_path, columns=cols)
    return apply_selection(df, region)


def load_waves_n_photoz(photoz_filepath, photom_filepath, stargal_filepath):
    """
    Load WAVES-N photometry and photo-zs, and merge in the star/galaxy
    separation flags from the stargal catalogue.
    """
    df_photoz = pd.read_parquet(photoz_filepath)
    df_photom = pd.read_parquet(photom_filepath)
    df_stargal = pd.read_parquet(stargal_filepath)

    # merge in the star/galaxy separation flags
    df = df_photoz.merge(df_stargal[["TARGETID", "stargal_flag"]], on="TARGETID", how="left")
    df = df.merge(df_photom[["TARGETID", "Z", "Z_1", "Z_2"]], on="TARGETID", how="left")
    return df


# =====================================================================
# 1. Forward model: evolving Schechter LF -> predicted dN/dz
# =====================================================================
class SchechterNzModel:
    """
    Forward-models dN/dz for a flux-limited sample from an evolving
    Schechter luminosity function.

    Provides both the cumulative flux-limited prediction (all galaxies
    brighter than a single apparent-magnitude limit) and the sliced
    prediction (galaxies within an apparent-magnitude range), the
    latter being what's needed to build synthetic n(z | m) histograms
    analogous to real catalogue data.
    """

    def __init__(
        self,
        H0=100.0, Om0=0.3, Ode0=0.7,
        Mstar0=-21.814943193345457,
        alpha=-1.3166859304357672,
        phistar0=0.004938332759020672,   # Mpc^-3 mag^-1 (h=1)
        P=1.625,       # density evolution
        Q=-0.07875,    # luminosity evolution
        Mmin=-24.25, Mmax=-13.5,          # valid abs-mag range of the LF fit
        zfit_max=0.5,                      # valid redshift range of the LF fit
        freeze_evolution=True,             # freeze M*(z), phi*(z) beyond zfit_max
        kcorr=None,
    ):
        self.cosmo = LambdaCDM(H0=H0, Om0=Om0, Ode0=Ode0)
        self.Mstar0 = Mstar0
        self.alpha = alpha
        self.phistar0 = phistar0
        self.P = P
        self.Q = Q
        self.Mmin = Mmin
        self.Mmax = Mmax
        self.zfit_max = zfit_max
        # If True, the P/Q evolution is evaluated at min(z, zfit_max):
        # beyond the LF's fitted redshift range the Schechter parameters
        # are held at their zfit_max values instead of being
        # exponentially extrapolated (phi* alone would otherwise grow
        # by x4.5 at z=1 and x20 at z=2 with P=1.625, manufacturing
        # high-z tails in the faint slices).
        self.freeze_evolution = freeze_evolution
        # Approximate population-mean VISTA Z-band K(z) by default
        # (see make_polynomial_kcorr); swap in a kcorrect-derived
        # polynomial fit from WAVES photometry via the kcorr= kwarg.
        self.kcorr = kcorr if kcorr is not None else self.make_polynomial_kcorr()

    # -----------------------------------------------------------
    # K-corrections
    # -----------------------------------------------------------
    @staticmethod
    def make_polynomial_kcorr(coeffs=(0.20, 1.00)):
        """
        Build K(z) = coeffs[0]*z + coeffs[1]*z^2 + ... (K(0) = 0 by
        construction). The default (0.20, 1.00) is an APPROXIMATE
        population-mean K(z) for the VISTA Z band -- e.g. K ~ 0.08 mag
        at z=0.2, ~0.35 at z=0.5, ~0.80 at z=0.8 -- chosen to be
        broadly consistent with kcorrect-style values for a mixed
        red/blue population, and to grow faster than pure bandwidth
        compression at z > 0.5 where the observed band moves into the
        rest-frame blue. Replace the coefficients with a proper
        polynomial fit of kcorrect K-corrections from WAVES/GAMA
        photometry (ideally colour-dependent) before doing precision
        work.
        """
        coeffs = np.asarray(coeffs, dtype=float)

        def kcorr(z):
            z = np.asarray(z, dtype=float)
            out = np.zeros_like(z)
            for i, c in enumerate(coeffs):
                out = out + c * z ** (i + 1)
            return out if out.ndim else float(out)

        return kcorr

    @staticmethod
    def bandwidth_kcorr(z):
        """Legacy pure bandpass-compression K-correction, 2.5 log10(1+z).
        Kept for comparison; underestimates real Z-band K at z > ~0.5."""
        return 2.5 * np.log10(1.0 + z)

    # -----------------------------------------------------------
    def schechter_params(self, z):
        """
        Evolve M*, phi* to redshift z. alpha held fixed.

        If freeze_evolution is set (default), evolution is evaluated at
        z_eff = min(z, zfit_max): the LF fit has no support beyond
        zfit_max, so rather than extrapolating M*(z) linearly and
        phi*(z) exponentially, both are held at their zfit_max values.
        """
        z_eff = min(z, self.zfit_max) if self.freeze_evolution else z
        Mstar = self.Mstar0 - self.Q * z_eff
        phistar = self.phistar0 * 10 ** (0.4 * self.P * z_eff)
        return Mstar, phistar, self.alpha

    def n_brighter_than(self, Mlim, z):
        """
        Number density (Mpc^-3) of galaxies with M < Mlim at redshift z,
        via the unnormalized upper incomplete gamma function (mpmath
        supports the negative, non-integer order alpha+1 that scipy
        cannot).
        """
        Mstar, phistar, a = self.schechter_params(z)
        x = 10 ** (0.4 * (Mstar - Mlim))
        if x <= 0:
            x = 1e-8  # avoid divergence as x -> 0 for alpha+1 <= 0
        val = mpmath.gammainc(a + 1, x, mpmath.inf)
        return phistar * float(val)

    # -----------------------------------------------------------
    def distance_modulus(self, z):
        d_L = self.cosmo.luminosity_distance(z).to(u.Mpc).value
        return 5 * np.log10(d_L) + 25

    def Mlim_of_z(self, z, Zlim):
        """M_lim(z) = Zlim - DM(z) - K(z). Q-evolution is NOT re-applied
        here since it's already folded into schechter_params via M*(z)."""
        return Zlim - self.distance_modulus(z) - self.kcorr(z)

    def dVdz_per_sr(self, z):
        return self.cosmo.differential_comoving_volume(z).to(u.Mpc**3 / u.sr).value

    # -----------------------------------------------------------
    def predict_dNdz(self, z_array, Zlim, area_deg2):
        """dN/dz for all galaxies brighter than a single flux limit Zlim."""
        area_sr = area_deg2 * (np.pi / 180.0) ** 2
        dNdz = np.zeros_like(np.asarray(z_array, dtype=float))
        for i, z in enumerate(z_array):
            if z <= 0:
                continue
            Ml = np.clip(self.Mlim_of_z(z, Zlim), self.Mmin, self.Mmax)
            n_z = self.n_brighter_than(Ml, z)
            dV = self.dVdz_per_sr(z) * area_sr
            dNdz[i] = n_z * dV
        return dNdz

    def predict_dNdz_slice(self, z_array, mag_lo, mag_hi, area_deg2):
        """
        dN/dz for galaxies with apparent magnitude in [mag_lo, mag_hi):
        the difference of two cumulative flux-limited predictions
        (fainter limit minus brighter limit).
        """
        dNdz_hi = self.predict_dNdz(z_array, mag_hi, area_deg2)
        dNdz_lo = self.predict_dNdz(z_array, mag_lo, area_deg2)
        return np.clip(dNdz_hi - dNdz_lo, 0.0, None)

    # -----------------------------------------------------------
    def plot_flux_limited(self, z_grid, samples, filename="predicted_Nz.png", dpi=150):
        """
        samples: list of (Zlim, area_deg2, label) tuples, e.g.
        [(21.1, 1200.0, "WAVES-Wide"), (21.25, 65.0, "WAVES-Deep")]
        """
        fig, ax = plt.subplots(figsize=(7, 5))
        for Zlim, area, label in samples:
            dNdz = self.predict_dNdz(z_grid, Zlim, area)
            ax.plot(z_grid, dNdz, label=f"{label} (Z<{Zlim})")
        ax.set_xlabel("z")
        ax.set_ylabel("dN/dz")
        ax.legend()
        ax.set_title("Predicted N(z) from evolving Schechter LF")
        fig.tight_layout()
        fig.savefig(filename, dpi=dpi)
        print(f"Saved plot to {filename}")


# =====================================================================
# 3. Generalized 4-parameter fit: A * z^alpha * exp[-(z/z_c)^beta]
# =====================================================================
class GeneralNzFitter:
    """
    Fits

        dN/dz(z) = A * z^alpha * exp[-(z / z_c)^beta]

    independently to EACH magnitude slice, with A, alpha, z_c, beta all
    free (4 parameters per slice, not shared/joint across slices). This
    is a strict generalization of Baugh & Efstathiou (1993)
    """

    def __init__(self, mag_edges=None):
        self.mag_edges = np.asarray(mag_edges) if mag_edges is not None else np.arange(16, 23, 1)
        self.mag_centres = self.mag_edges[:-1] + 0.5

        self.hist_list = []          # normalized (unit-area) target dN/dz per slice
        self.z_grids = []            # each slice gets ITS OWN z_grid (different extent)
        self.valid_slices = []       # list of (mlo, mhi) with data
        self.slice_totals = []       # total predicted counts per deg^2 per slice
        self.results = []            # list of dicts: {A, alpha, zc, beta, perr, pcov}

    # -----------------------------------------------------------
    @classmethod
    def from_model(cls, model, mag_edges=None, n_points=500, tail_tol=1e-2):
        """
        Build target dN/dz shapes per magnitude slice from a
        SchechterNzModel, with each slice's z_grid adaptively widened
        (via find_adaptive_zmax) until it captures >= (1 - tail_tol) of
        that slice's predicted mass, rather than sharing one fixed
        range across all slices.
        """
        obj = cls(mag_edges=mag_edges)
        any_beyond_fit = False

        for mlo, mhi in zip(obj.mag_edges[:-1], obj.mag_edges[1:]):
            z_max = find_adaptive_zmax(model, mlo, mhi, tol=tail_tol)
            z_grid = np.linspace(1e-4, z_max, n_points)
            dNdz = model.predict_dNdz_slice(z_grid, mlo, mhi, area_deg2=1.0)
            total = np.trapezoid(dNdz, z_grid)
            if total <= 0:
                continue

            if z_max > model.zfit_max:
                any_beyond_fit = True

            obj.hist_list.append(dNdz / total)   # normalize to unit area
            obj.z_grids.append(z_grid)
            obj.valid_slices.append((mlo, mhi))
            obj.slice_totals.append(total)       # counts deg^-2 in the slice

        if any_beyond_fit:
            if model.freeze_evolution:
                print(f"NOTE: some slices needed z_grid extents beyond the LF's fitted "
                      f"range (z <= {model.zfit_max}); M*(z)/phi*(z) are FROZEN at their "
                      f"z={model.zfit_max} values there (freeze_evolution=True), not "
                      f"extrapolated.")
            else:
                print(f"NOTE: some slices needed z_grid extents beyond the LF's fitted "
                      f"range (z <= {model.zfit_max}) to fully contain their mass -- "
                      f"M*(z)/phi*(z) are being extrapolated there.")

        return obj

    # -----------------------------------------------------------
    @staticmethod
    def analytic_A(alpha, zc, beta):
        """
        A that makes A*z^alpha*exp[-(z/zc)^beta] integrate to 1 over
        z in [0, inf):

            integral = (zc^(alpha+1) / beta) * Gamma((alpha+1)/beta)
            A = 1 / integral
        """
        integral = (zc ** (alpha + 1) / beta) * gamma((alpha + 1.0) / beta)
        return 1.0 / integral

    @classmethod
    def model_func(cls, z, alpha, zc, beta):
        """Unit-area-normalized A*z^alpha*exp[-(z/zc)^beta], with A solved analytically."""
        A = cls.analytic_A(alpha, zc, beta)
        return A * z**alpha * np.exp(-(z / zc) ** beta)

    # -----------------------------------------------------------
    def fit(
        self,
        p0=(2.0, 0.15, 1.5),
        bounds=((0.1, 1e-4, 0.2), (8.0, 5.0, 8.0)),
    ):
        """
        Fit each slice independently for (alpha, zc, beta) by
        UNWEIGHTED least squares in LINEAR density space. A is not a
        free parameter -- it's fixed analytically so the curve
        integrates to 1 over [0, inf) (see analytic_A).

        Known caveats of this scheme (deliberate, documented rather
        than "fixed"):
        - Linear-space unweighted L2 means the tall peak dominates the
          loss; the low- and high-z tails carry little weight and can
          be visibly poorly fit even when the peak is excellent.
        - No `sigma` is passed to curve_fit, so pcov / the *_err values
          reflect only the residual scatter of the unweighted fit and
          should be treated as indicative, not as proper parameter
          uncertainties.
        - analytic_A normalizes over z in [0, inf) while the target
          histograms are unit-normalized over their finite (truncated)
          z_grid; with tail_tol=1e-2 up to ~1% of the fitted curve's
          mass can lie beyond the grid, a small systematic in zc/beta.
        """
        self.results = []
        for (mlo, mhi), density, z_grid in zip(self.valid_slices, self.hist_list, self.z_grids):
            popt, pcov = curve_fit(
                self.model_func, z_grid, density,
                p0=p0, bounds=bounds, maxfev=20000,
            )
            perr = np.sqrt(np.diag(pcov))
            alpha, zc, beta = popt
            A = self.analytic_A(alpha, zc, beta)
            self.results.append({
                "mlo": mlo, "mhi": mhi,
                "A": A, "alpha": alpha, "zc": zc, "beta": beta,
                "alpha_err": perr[0], "zc_err": perr[1], "beta_err": perr[2],
                "z_max_used": z_grid[-1],
                "popt": popt, "pcov": pcov,
            })
        return self.results

    # -----------------------------------------------------------
    def summary(self):
        if not self.results:
            raise RuntimeError("Call .fit() first.")
        print(f"{'slice':>8} {'A':>12} {'alpha':>10} {'z_c':>10} {'beta':>10} {'peak z':>10} {'z_max used':>12}")
        for r in self.results:
            # mode of A*z^alpha*exp[-(z/zc)^beta] (dlnf/dz=0): z_peak = zc*(alpha/beta)^(1/beta)
            z_peak = r["zc"] * (r["alpha"] / r["beta"]) ** (1.0 / r["beta"]) if r["alpha"] > 0 else 0.0
            print(f"{r['mlo']:>4.0f}-{r['mhi']:<3.0f} {r['A']:12.4f} {r['alpha']:10.4f} "
                  f"{r['zc']:10.4f} {r['beta']:10.4f} {z_peak:10.4f} {r['z_max_used']:12.2f}")

    # -----------------------------------------------------------
    def plot(self, filename="general_nz_fit.png", dpi=150):
        n_slices = len(self.results)
        fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3), sharey=True)

        for idx, (r, density, z_grid) in enumerate(zip(self.results, self.hist_list, self.z_grids)):
            ax = axes[idx] if n_slices > 1 else axes
            ax.plot(z_grid, density, "o", ms=2, label="model shape")
            fitted = self.model_func(z_grid, r["alpha"], r["zc"], r["beta"])
            ax.plot(z_grid, fitted, "-", label="fitted template")
            ax.set_title(f"{r['mlo']:.0f}-{r['mhi']:.0f}")
            ax.set_xlabel("z")

        (axes[0] if n_slices > 1 else axes).set_ylabel("dN/dz (normalized)")
        (axes[0] if n_slices > 1 else axes).legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        print(f"\nSaved plot to {filename}")

    # -----------------------------------------------------------
    def plot_vs_mock(
        self,
        model,
        mock_df,
        mock_area_deg2=None,
        z_mock_max=0.8,
        mock_maglim=21.25,
        dz=0.02,
        z_col="redshift_observed",
        mag_col="mag_Z_VISTA",
        filename="nz_vs_sharks.png",
        dpi=150,
    ):
        """
        Overplot the SHARKS mock n(z|m) histograms on the LF-model
        predictions and fitted templates, per magnitude slice, in
        ABSOLUTE units of counts / dz / deg^2 (so no normalization
        ambiguity from the mock's z < z_mock_max truncation).

        The SHARKS deep mock is only complete to z = 0.8 and Z < 21.25
        over the WAVES-Deep-sized WD footprint (~50.6 deg^2):
        - the mock histogram is only drawn up to z_mock_max, with a
          dotted vertical line marking the truncation; the model curve
          continues beyond it,
        - slices that straddle the Z < 21.25 flux limit are flagged as
          INCOMPLETE (only the m < 21.25 part of the slice is present
          in the mock, so the mock histogram is a lower bound there),
        - slices entirely fainter than 21.25 get no mock overlay.
        """
        if not self.results:
            raise RuntimeError("Call .fit() first.")
        if mock_area_deg2 is None:
            mock_area_deg2 = survey_area_deg2(*WD)

        bins = np.arange(0.0, z_mock_max + dz, dz)
        centres = 0.5 * (bins[:-1] + bins[1:])
        n_slices = len(self.results)
        fig, axes = plt.subplots(1, n_slices, figsize=(3 * n_slices, 3.2))
        if n_slices == 1:
            axes = [axes]

        for idx, (r, z_grid) in enumerate(zip(self.results, self.z_grids)):
            ax = axes[idx]
            mlo, mhi = r["mlo"], r["mhi"]

            # model prediction and fitted template, per deg^2
            model_abs = model.predict_dNdz_slice(z_grid, mlo, mhi, area_deg2=1.0)
            total = self.slice_totals[idx]
            fitted_abs = total * self.model_func(z_grid, r["alpha"], r["zc"], r["beta"])
            ax.plot(z_grid, model_abs, "-", lw=1.2, label="LF model")
            ax.plot(z_grid, fitted_abs, "--", lw=1.2, label="fitted template")

            # mock overlay
            if mlo < mock_maglim:
                sel = (mock_df[mag_col] >= mlo) & (mock_df[mag_col] < mhi)
                z_vals = mock_df.loc[sel, z_col].to_numpy()
                counts, _ = np.histogram(z_vals, bins=bins)
                y = counts / (dz * mock_area_deg2)
                yerr = np.sqrt(counts) / (dz * mock_area_deg2)
                ax.errorbar(centres, y, yerr=yerr, fmt=".", ms=3, lw=0.8,
                            label="SHARKS deep", zorder=5)
                if mhi > mock_maglim:
                    ax.set_title(f"{mlo:.0f}-{mhi:.0f}  (mock incomplete: Z<{mock_maglim})",
                                 fontsize=9)
                else:
                    ax.set_title(f"{mlo:.0f}-{mhi:.0f}", fontsize=10)
            else:
                ax.set_title(f"{mlo:.0f}-{mhi:.0f}  (no mock: Z<{mock_maglim})", fontsize=9)

            ax.axvline(z_mock_max, ls=":", lw=0.8, color="grey")
            ax.set_xlabel("z")
            ax.set_xlim(0, max(z_grid[-1], z_mock_max))

        axes[0].set_ylabel(r"dN/dz  [deg$^{-2}$]")
        axes[0].legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(filename, dpi=dpi)
        print(f"\nSaved model-vs-SHARKS comparison to {filename} "
              f"(mock area = {mock_area_deg2:.2f} deg^2, truncated at z = {z_mock_max})")

    # -----------------------------------------------------------
    def plot_paper(
        self,
        model,
        mock_df,
        mock_area_deg2=None,
        z_mock_max=0.8,
        mock_maglim=21.25,
        dz=0.02,
        z_col="redshift_observed",
        mag_col="mag_Z_VISTA",
        z_plot_max=None,
        yscale="log",
        cmap="viridis",
        filename="nz_vs_sharks_paper",
        dpi=300,
    ):
        """
        Publication-quality single-panel version of the SHARKS
        comparison: every magnitude slice overlaid on ONE set of axes,
        one colour per slice, with

            solid line   = LF forward-model dN/dz,
            dashed line  = fitted A z^alpha exp[-(z/zc)^beta] template,
            step histogram (same colour) = SHARKS deep mock,

        all in absolute counts / dz / deg^2. Colour encodes the slice;
        line style / histogram encodes the component -- so the figure
        carries two legends (slices by colour, components by style).

        SHARKS caveats are handled as in plot_vs_mock: the mock is only
        drawn to z_mock_max (dotted vertical line), slices straddling
        the Z < mock_maglim flux limit are marked incomplete in the
        legend, and slices entirely fainter than the limit get no
        histogram.

        Saves both <filename>.pdf (vector, for the paper) and
        <filename>.png.
        """
        from matplotlib.lines import Line2D

        if not self.results:
            raise RuntimeError("Call .fit() first.")
        if mock_area_deg2 is None:
            mock_area_deg2 = survey_area_deg2(*WD)
        if z_plot_max is None:
            z_plot_max = max(max(zg[-1] for zg in self.z_grids), z_mock_max)

        bins = np.arange(0.0, z_mock_max + dz, dz)
        n_slices = len(self.results)
        colours = plt.get_cmap(cmap)(np.linspace(0.0, 0.9, n_slices))

        with plt.rc_context({
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.8,
            "xtick.direction": "in", "ytick.direction": "in",
            "xtick.top": True, "ytick.right": True,
            "xtick.minor.visible": True, "ytick.minor.visible": True,
        }):
            fig, ax = plt.subplots(figsize=(7.0, 5.0))

            slice_handles = []
            for r, z_grid, total, col in zip(self.results, self.z_grids,
                                             self.slice_totals, colours):
                mlo, mhi = r["mlo"], r["mhi"]

                # LF forward model (solid) and fitted template (dashed)
                model_abs = model.predict_dNdz_slice(z_grid, mlo, mhi, area_deg2=1.0)
                fitted_abs = total * self.model_func(z_grid, r["alpha"], r["zc"], r["beta"])
                ax.plot(z_grid, model_abs, "-", color=col, lw=1.6, zorder=3)
                ax.plot(z_grid, fitted_abs, "--", color=col, lw=1.4, zorder=4)

                # SHARKS mock as a step histogram in the same colour
                label = rf"${mlo:.0f} \leq Z < {mhi:.0f}$"
                if mlo < mock_maglim:
                    sel = (mock_df[mag_col] >= mlo) & (mock_df[mag_col] < mhi)
                    counts, _ = np.histogram(mock_df.loc[sel, z_col].to_numpy(), bins=bins)
                    ax.stairs(counts / (dz * mock_area_deg2), bins,
                              color=col, lw=1.1, alpha=0.85, zorder=2)
                    if mhi > mock_maglim:
                        label += rf" (mock $Z<{mock_maglim}$)"
                else:
                    label += " (no mock)"
                slice_handles.append(Line2D([], [], color=col, lw=3, label=label))

            ax.axvline(z_mock_max, ls=":", lw=0.9, color="0.4", zorder=1)
            ax.annotate("SHARKS limit", xy=(z_mock_max, 0.985), xycoords=("data", "axes fraction"),
                        xytext=(4, 0), textcoords="offset points",
                        rotation=90, va="top", ha="left", fontsize=8, color="0.4")

            #ax.set_yscale(yscale)
            ax.set_xlim(0.0, z_plot_max)
            ax.set_xlabel(r"redshift $z$", fontsize=12)
            ax.set_ylabel(r"$\mathrm{d}N/\mathrm{d}z\;\;[\mathrm{deg}^{-2}]$", fontsize=12)

            # legend 1: colour -> magnitude slice
            leg1 = ax.legend(handles=slice_handles, loc="upper right", fontsize=8,
                             frameon=False, title="magnitude slice", title_fontsize=9)
            ax.add_artist(leg1)
            # legend 2: style -> component
            style_handles = [
                Line2D([], [], color="k", ls="-", lw=1.6, label="LF forward model"),
                Line2D([], [], color="k", ls="--", lw=1.4, label="fitted template"),
                Line2D([], [], color="k", ls="-", lw=1.1, drawstyle="steps-mid",
                       alpha=0.85, label="SHARKS deep mock"),
            ]
            ax.legend(handles=style_handles, loc="lower right", fontsize=8, frameon=False)

            fig.tight_layout()
            for ext in ("pdf", "png"):
                fig.savefig(f"{filename}.{ext}", dpi=dpi, bbox_inches="tight")
        print(f"\nSaved paper figure to {filename}.pdf / .png "
              f"(mock area = {mock_area_deg2:.2f} deg^2)")


# =====================================================================
# 4. Example usage
# =====================================================================
if __name__ == "__main__":
    model = SchechterNzModel()   # freeze_evolution=True, polynomial VISTA-Z K(z)

    # sanity-check plot of the original cumulative flux-limited predictions
    z_grid = np.linspace(0.001, 0.5, 200)
    model.plot_flux_limited(
        z_grid,
        samples=[(21.1, 1200.0, "WAVES-Wide"), (21.25, 65.0, "WAVES-Deep")],
        filename="predicted_Nz.png",
    )

    gen_fitter = GeneralNzFitter.from_model(
        model,
        mag_edges=np.arange(16, 23, 1),
    )
    gen_fitter.fit()
    gen_fitter.summary()
    gen_fitter.plot(filename="general_nz_fit.png")

    # ------------------------------------------------------------
    # SHARKS mock comparison (WAVES-Deep region, z < 0.8, Z < 21.25)
    # ------------------------------------------------------------
    sharks_path = "/Users/sp624AA/Downloads/groupfinding_comp_mocks/fibre_incomplete_mocks.parquet"   # <-- set to your mock file
    try:
        mock = load_sharks_mock(sharks_path, region="deep")
    except (FileNotFoundError, OSError):
        print(f"\nSHARKS mock not found at '{sharks_path}' -- skipping overlay.")
    else:
        gen_fitter.plot_vs_mock(
            model,
            mock,
            mock_area_deg2=survey_area_deg2(*WD),   # ~50.6 deg^2, exact spherical area
            z_mock_max=0.8,
            mock_maglim=21.25,
            filename="nz_vs_sharks.png",
        )
        gen_fitter.plot_paper(
            model,
            mock,
            z_mock_max=0.8,
            mock_maglim=21.25,
            filename="nz_vs_sharks_paper",   # saves .pdf and .png
        )