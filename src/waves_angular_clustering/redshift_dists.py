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


def load_sharks_mock(filepath):
    df = pd.read_parquet(filepath)
    # cut to waves deep boundary, as only this extends to 0.8 z.
    # rename columns
    # apply k correction. 
    return df


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
        # placeholder pure bandpass-compression K-correction by default;
        # swap in a real VISTA-native K(z) polynomial via the kcorr= kwarg
        self.kcorr = kcorr if kcorr is not None else self._default_kcorr

    @staticmethod
    def _default_kcorr(z):
        return 2.5 * np.log10(1.0 + z)

    # -----------------------------------------------------------
    def schechter_params(self, z):
        """Evolve M*, phi* to redshift z. alpha held fixed."""
        Mstar = self.Mstar0 - self.Q * z
        phistar = self.phistar0 * 10 ** (0.4 * self.P * z)
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
        any_extrapolated = False

        for mlo, mhi in zip(obj.mag_edges[:-1], obj.mag_edges[1:]):
            z_max = find_adaptive_zmax(model, mlo, mhi, tol=tail_tol)
            z_grid = np.linspace(1e-4, z_max, n_points)
            dNdz = model.predict_dNdz_slice(z_grid, mlo, mhi, area_deg2=1.0)
            total = np.trapezoid(dNdz, z_grid)
            if total <= 0:
                continue

            if z_max > model.zfit_max:
                any_extrapolated = True

            obj.hist_list.append(dNdz / total)   # normalize to unit area
            obj.z_grids.append(z_grid)
            obj.valid_slices.append((mlo, mhi))

        if any_extrapolated:
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
        Fit each slice independently for (alpha, zc, beta); A is not a
        free parameter -- it's fixed analytically so the curve
        integrates to 1 (see analytic_A). log_space=True (default) fits
        log(density) vs log(model), giving the peak and the tail
        comparable weight in the loss rather than letting the tall peak
        dominate an unweighted L2 fit.
        """
        self.results = []
        for (mlo, mhi), density, z_grid in zip(self.valid_slices, self.hist_list, self.z_grids):
            floor = 1e-4 * density.max()


            target = density
            fit_func = self.model_func

            popt, pcov = curve_fit(
                fit_func, z_grid, target,
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


# =====================================================================
# 4. Example usage
# =====================================================================
if __name__ == "__main__":
    model = SchechterNzModel()

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