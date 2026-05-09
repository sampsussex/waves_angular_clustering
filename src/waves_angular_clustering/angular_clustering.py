import numpy as np
import matplotlib.pyplot as plt
import treecorr
import pandas as pd
import json
import os
import itertools


class AngularClustering:
    def __init__(self, ra_cat, dec_cat, ra_rand, dec_rand, selection_dic,
                 min_sep=0.01, max_sep=10, nbins=100, sep_units='degrees',
                 cat_units='degrees', rand_units='degrees',
                 n_patch=20, var_method='jackknife'):
        self.ra_cat = ra_cat
        self.dec_cat = dec_cat
        self.ra_rand = ra_rand
        self.dec_rand = dec_rand
        self.selection_name = selection_dic
        self.n_patch = n_patch
        self.var_method = var_method
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.sep_units = sep_units
        self.cat_units = cat_units
        self.rand_units = rand_units
        self.nbins = nbins

        self._make_catalogs()

        self.results = {
            'selection': selection_dic,
            'columns': {
                'xi': None,
                'varxi': None,
                'meanlogr': None
            }
        }

    def _make_catalogs(self):
        self.data_cat = treecorr.Catalog(
            ra=self.ra_cat, dec=self.dec_cat,
            ra_units=self.cat_units, dec_units=self.cat_units,
            npatch=self.n_patch
        )
        self.rand_cat = treecorr.Catalog(
            ra=self.ra_rand, dec=self.dec_rand,
            ra_units=self.rand_units, dec_units=self.rand_units,
            patch_centers=self.data_cat.patch_centers
        )

    def do_correlations(self):
        """
        Run DD, DR, and RR correlations and compute xi.
        """
        dd = treecorr.NNCorrelation(
            min_sep=self.min_sep, max_sep=self.max_sep,
            nbins=self.nbins, sep_units=self.sep_units,
            var_method=self.var_method
        )
        dr = treecorr.NNCorrelation(
            min_sep=self.min_sep, max_sep=self.max_sep,
            nbins=self.nbins, sep_units=self.sep_units,
            var_method=self.var_method
        )
        dd.process(self.data_cat)
        dr.process(self.data_cat, self.rand_cat)


        rr = treecorr.NNCorrelation(
            min_sep=self.min_sep, max_sep=self.max_sep,
            nbins=self.nbins, sep_units=self.sep_units, 
            var_method=self.var_method
        )
        rr.process(self.rand_cat)

        self.xi, self.varxi = dd.calculateXi(rr=rr, dr=dr)
        self.meanlogr = dd.meanlogr

        # Store as lists so they are JSON-serialisable
        self.results['columns']['xi'] = self.xi.tolist()
        self.results['columns']['varxi'] = self.varxi.tolist()
        self.results['columns']['meanlogr'] = self.meanlogr.tolist()

    def save_results(self, save_location):
        with open(save_location, 'w') as f:
            json.dump(self.results, f)

    def clean_up(self):
        """Release treecorr catalog memory."""
        del self.data_cat
        del self.rand_cat


class WavesWideClustering:
    def __init__(self, n_photom_filepath=None, s_photom_filepath=None,
                 n_stargal_filepath=None, s_stargal_filepath=None,
                 n_randoms_filepath=None, s_randoms_filepath=None,
                 results_directory=None):

        self.n_photom_filepath = n_photom_filepath
        self.s_photom_filepath = s_photom_filepath

        self.n_stargal_filepath = n_stargal_filepath
        self.s_stargal_filepath = s_stargal_filepath

        self.n_randoms_filepath = n_randoms_filepath
        self.s_randoms_filepath = s_randoms_filepath
        self.randoms_realisation_to_load = [0, 1, 2, 3, 4]

        self.results_directory = results_directory

        # Treecorr binning settings — shared by all AngularClustering instances
        # and used when reconstructing an RR object from cache.
        self.min_sep   = 0.01
        self.max_sep   = 10
        self.nbins     = 100
        self.sep_units = 'degrees'

        self.data_ra_col = 'RAmax'
        self.data_dec_col = 'Decmax'
        self.randoms_ra_col = 'ra'
        self.randoms_dec_col = 'dec'

        self.columns_to_load_photom = [
            'uberID', self.data_ra_col, self.data_dec_col,
            'class', 'mag_Zt', 'mask', 'starmask', 'ghostmask',
            'duplicate'
        ]
        self.columns_to_load_stargal = ['uberID', 'stargal']
        # NOTE: 'ghostmask' added here — it is used in _load_randoms but was
        # missing from the original columns list.
        self.columns_to_load_randoms = [
            self.randoms_ra_col, self.randoms_dec_col,
            'starmask', 'ghostmask', 'polygon_mask', 'realisation'
        ]

        # ------------------------------------------------------------------ #
        # Selection definitions
        # ------------------------------------------------------------------ #
        # NOTE: unified naming — was 'TOPZ+SFM' in possible_selections but
        # 'TOPZ/SFM/R50' in selections_to_run. Standardised to 'TOPZ/SFM/R50'.
        self.possible_selections = {
            'target_selection':   ['galaxy', 'galaxy/ambiguous', 'star'],
            'ghostmask_selection':['no ghostmask', 'with ghostmask'],
            'survey_depth':       ['Z<21.1', 'Z<21.25', 'Z<22'],
            'star_gal_method':    ['TOPZ/SFM/R50', 'baseline'],
            'region':             ['WWN', 'WWS', 'WW combined'],
        }

        selections_to_run = {
            'target_selection':   ['galaxy'],#, 'galaxy/ambiguous', 'star'],
            'ghostmask_selection':['no ghostmask', 'with ghostmask'],
            'survey_depth':       ['Z<21.1'],#, 'Z<21.25', 'Z<22'],
            'star_gal_method':    ['TOPZ/SFM/R50', 'baseline'],
            'region':             ['WWN', 'WWS'],
        }

        self._validate_selections(selections_to_run)

        # Expand the dict-of-lists into a flat list of individual selection dicts,
        # one per combination (Cartesian product).
        self.selections_to_run = self._expand_selections(selections_to_run)

    # ---------------------------------------------------------------------- #
    # Private helpers
    # ---------------------------------------------------------------------- #

    def _validate_selections(self, selections_to_run):
        """Raise ValueError if any value in selections_to_run is not in possible_selections."""
        for key, values in selections_to_run.items():
            if key not in self.possible_selections:
                raise ValueError(f"Unknown selection key: '{key}'")
            for v in values:
                if v not in self.possible_selections[key]:
                    raise ValueError(
                        f"Invalid value '{v}' for key '{key}'. "
                        f"Allowed values: {self.possible_selections[key]}"
                    )

    @staticmethod
    def _expand_selections(selections_dict):
        """
        Convert a dict-of-lists into a list of individual selection dicts.

        Example
        -------
        {'a': [1, 2], 'b': ['x']}  ->  [{'a': 1, 'b': 'x'}, {'a': 2, 'b': 'x'}]
        """
        keys = list(selections_dict.keys())
        value_lists = [selections_dict[k] for k in keys]
        return [dict(zip(keys, combo)) for combo in itertools.product(*value_lists)]

    @staticmethod
    def _selection_to_filename(selection):
        """Create a safe filename string from a selection dict."""
        parts = [f"{k}={v}" for k, v in sorted(selection.items())]
        name = "__".join(parts).replace(' ', '_').replace('<', 'lt').replace('/', '-')
        return f"clustering__{name}.json"

    def _get_results_path(self, selection):
        return os.path.join(self.results_directory, self._selection_to_filename(selection))

    def _check_if_results_exist(self, selection):
        """Return True if results have already been saved for this selection."""
        return os.path.isfile(self._get_results_path(selection))

    def _get_filepaths_for_selection(self, selection):
        """Return (photom_fp, stargal_fp, randoms_fp) for a given region."""
        region = selection['region']
        if region == 'WWN':
            return self.n_photom_filepath, self.n_stargal_filepath, self.n_randoms_filepath
        elif region == 'WWS':
            return self.s_photom_filepath, self.s_stargal_filepath, self.s_randoms_filepath
        elif region == 'WW combined':
            return None, None, None   # handled separately via _load_WWC_*
        else:
            raise ValueError(f"Unknown region: '{region}'")

    # ---------------------------------------------------------------------- #
    # Data loading
    # ---------------------------------------------------------------------- #

    def _load_dataset(self, photom_filepath, stargal_filepath, selection):
        print(f"  Loading photometric data from {photom_filepath}...")
        df = pd.read_parquet(photom_filepath, columns=self.columns_to_load_photom)
        print(f"  Loaded {len(df)} rows from photometric catalogue.")
        df['uberID'] = df['uberID'].astype(np.int64)

        # ------------------------------------------------------------------ #
        # Star/galaxy separation
        # ------------------------------------------------------------------ #
        # Initialise stargal column to NaN so the base_selection mask works
        # correctly even for rows that don't match the chosen method.
        df['stargal'] = np.nan

        if selection['star_gal_method'] == 'TOPZ/SFM/R50':
            print(f"  Loading stargal classification from {stargal_filepath}...")
            df_stargal = pd.read_parquet(stargal_filepath, columns=self.columns_to_load_stargal)
            df_stargal['uberID'] = df_stargal['uberID'].astype(np.int64)
            print(f"  Loaded {len(df_stargal)} rows from stargal catalogue.")
            # Merge brings in the external stargal classification
            df = df.merge(df_stargal, on='uberID', how='left', suffixes=('', '_ext'))
            del df_stargal
            # Use the external column if present, fall back to the initialised NaN
            if 'stargal_ext' in df.columns:
                df['stargal'] = df['stargal_ext']
                df.drop(columns=['stargal_ext'], inplace=True)

        elif selection['star_gal_method'] == 'baseline':
            # Use the photometric 'class' column directly
            df['stargal'] = df['class']

        # ------------------------------------------------------------------ #
        # Build selection mask
        # ------------------------------------------------------------------ #
        base_selection = (
            (df['duplicate'] == False) &
            (df['mask'] == False) &
            (df['starmask'] == False)
        )

        target = selection['target_selection']
        if target == 'galaxy':
            base_selection &= df['stargal'] == 'galaxy'
        elif target == 'galaxy/ambiguous':
            base_selection &= df['stargal'].isin(['galaxy', 'ambiguous'])
        elif target == 'star':
            base_selection &= df['stargal'] == 'star'

        if selection['ghostmask_selection'] == 'with ghostmask':
            base_selection &= df['ghostmask'] == False

        depth = selection['survey_depth']
        if depth == 'Z<21.1':
            base_selection &= df['mag_Zt'] < 21.1
        elif depth == 'Z<21.25':
            base_selection &= df['mag_Zt'] < 21.25
        elif depth == 'Z<22':
            base_selection &= df['mag_Zt'] < 22

        df_sel = df.loc[base_selection].copy()
        del base_selection
        del df  # free memory

        if len(df_sel) == 0:
            raise ValueError(
                f"Dataset is empty after applying selection: {selection}"
            )

        ra_data = df_sel[self.data_ra_col].to_numpy(copy=True)
        dec_data = df_sel[self.data_dec_col].to_numpy(copy=True)
        del df_sel  # free memory

        if np.any(np.isnan(ra_data)) or np.any(np.isnan(dec_data)):
            raise ValueError(
                "NaN values found in RA/Dec after applying selection. "
                "Check input catalogue."
            )

        return ra_data, dec_data

    def _load_randoms(self, randoms_filepath, selection):
        print(f"  Loading randoms from {randoms_filepath} with selection {selection}...")
        df = pd.read_parquet(randoms_filepath, columns=self.columns_to_load_randoms)

        base_selection = (
            (df['starmask'] == False) &
            (df['polygon_mask'] == False) &
            (df['realisation'].isin(self.randoms_realisation_to_load))
        )

        if selection['ghostmask_selection'] == 'with ghostmask':
            base_selection &= df['ghostmask'] == False

        df_sel = df.loc[base_selection].copy()
        del df
        if len(df_sel) == 0:
            raise ValueError(
                f"Randoms catalogue is empty after applying selection: {selection}"
            )

        ra_randoms = df_sel[self.randoms_ra_col].to_numpy(copy=True)
        dec_randoms = df_sel[self.randoms_dec_col].to_numpy(copy=True)
        print(f"  Loaded {len(ra_randoms)} random points after selection.")
        del df_sel  # free memory
        return ra_randoms, dec_randoms

    def _load_WWC_data(self, selection):
        """Load and concatenate north + south data for the WW combined region."""
        ra_n, dec_n = self._load_dataset(
            self.n_photom_filepath, self.n_stargal_filepath, selection
        )
        ra_s, dec_s = self._load_dataset(
            self.s_photom_filepath, self.s_stargal_filepath, selection
        )
        return np.concatenate([ra_n, ra_s]), np.concatenate([dec_n, dec_s])

    def _load_WWC_randoms(self, selection):
        """Load and concatenate north + south randoms for the WW combined region."""
        ra_n, dec_n = self._load_randoms(self.n_randoms_filepath, selection)
        ra_s, dec_s = self._load_randoms(self.s_randoms_filepath, selection)
        return np.concatenate([ra_n, ra_s]), np.concatenate([dec_n, dec_s])

    # ---------------------------------------------------------------------- #
    # Results I/O
    # ---------------------------------------------------------------------- #

    def load_results(self, results_filepath):
        with open(results_filepath, 'r') as f:
            results = json.load(f)
        return results

    def get_previously_run_results(self):
        """
        Return a list of result dicts for all selections that already have
        saved output. Also removes those selections from self.selections_to_run
        so they are not recomputed.
        """
        already_run = []
        remaining = []
        for selection in self.selections_to_run:
            if self._check_if_results_exist(selection):
                results = self.load_results(self._get_results_path(selection))
                already_run.append(results)
            else:
                remaining.append(selection)
        self.selections_to_run = remaining
        return already_run

    # ---------------------------------------------------------------------- #
    # Clustering runners
    # ---------------------------------------------------------------------- #

    def get_clustering_for_selection(self, selection):
        """
        Run the angular clustering pipeline for a single selection dict and
        save the result to disk. Returns the results dict.
        """
        print(f"Running clustering for: {selection}")

        region = selection['region']
        if region == 'WW combined':
            print("  Loading and concatenating north + south data for WW combined region...")
            ra_data, dec_data = self._load_WWC_data(selection)
            ra_rand, dec_rand = self._load_WWC_randoms(selection)
            print(f"  Loaded {len(ra_data)} data points and {len(ra_rand)} randoms for WW combined.")
        else:
            print("  Loading data and randoms...")
            photom_fp, stargal_fp, randoms_fp = self._get_filepaths_for_selection(selection)
            ra_data, dec_data = self._load_dataset(photom_fp, stargal_fp, selection)
            ra_rand, dec_rand = self._load_randoms(randoms_fp, selection)
            print(f"  Loaded {len(ra_data)} data points and {len(ra_rand)} randoms.")

        self._diagnose_catalog("Data and Randoms", ra_data, dec_data, ra_rand, dec_rand)

        print("  Generating diagnostic plots...")
        self.plot_ra_dec_histograms(
            ra_data,
            dec_data,
            ra_rand,
            dec_rand,
            selection,
        )
        print("  Saved RA/Dec histogram diagnostics.")
        print("  Generating RA/Dec density diagnostic...")
        self.plot_ra_dec_density(
            ra_data,
            dec_data,
            ra_rand,
            dec_rand,
            selection,
        )

        print("  Initial diagnostics complete.")
        print("  Initialising AngularClustering instance...")
        ac = AngularClustering(
            ra_cat=ra_data, dec_cat=dec_data,
            ra_rand=ra_rand, dec_rand=dec_rand,
            selection_dic=selection,
            min_sep=self.min_sep, max_sep=self.max_sep,
            nbins=self.nbins, sep_units=self.sep_units,
        )
        print("  Computing correlations...")
        print("  Computing DD, DR, RR, and xi...")
        ac.do_correlations()
        print("  Clustering computation complete.")
        save_path = self._get_results_path(selection)
        print(f"  Saving results to {save_path}...")
        os.makedirs(self.results_directory, exist_ok=True)
        ac.save_results(save_path)
        print(f"  Saved to {save_path}")

        results = ac.results
        print("  Cleaning up treecorr catalogs from memory...")
        ac.clean_up()
        del ac  # free memory
        del ra_data, dec_data, ra_rand, dec_rand  # free memory
        print("  Done.")
        print("-" * 50)
        return results

    def get_clustering_for_all_selections_to_run(self):
        """
        Run the angular clustering pipeline for every selection in
        self.selections_to_run. Skips any for which results already exist.
        Returns a list of result dicts (new + previously saved).
        """
        all_results = self.get_previously_run_results()

        for selection in self.selections_to_run:
            try:
                result = self.get_clustering_for_selection(selection)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR for selection {selection}: {e}")

        return all_results
    
    def _diagnose_catalog(self, name, ra_data, dec_data, ra_rand, dec_rand):
        print(f"\n{name} diagnostics")
        print(f"  data:    N={len(ra_data)}, RA=({ra_data.min():.3f}, {ra_data.max():.3f}), Dec=({dec_data.min():.3f}, {dec_data.max():.3f})")
        print(f"  randoms: N={len(ra_rand)}, RA=({ra_rand.min():.3f}, {ra_rand.max():.3f}), Dec=({dec_rand.min():.3f}, {dec_rand.max():.3f})")
        print(f"  random/data ratio = {len(ra_rand)/len(ra_data):.2f}")

    def _get_diagnostics_directory(self):
        diag_dir = os.path.join(self.results_directory, "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)
        return diag_dir


    def _get_diagnostic_plot_path(self, selection, plot_type):
        """
        plot_type examples:
            'hist1d'
            'density2d'
        """
        base = self._selection_to_filename(selection)
        base = base.replace(".json", "")

        filename = f"{base}__{plot_type}.png"

        return os.path.join(
            self._get_diagnostics_directory(),
            filename
        )

    def plot_ra_dec_histograms(
        self,
        ra_data,
        dec_data,
        ra_rand,
        dec_rand,
        selection,
        bins=100,
        normalise=True,
    ):
        """
        Save-only 1D RA/Dec histogram comparison.
        """

        save_path = self._get_diagnostic_plot_path(
            selection,
            "hist1d"
        )

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        density = normalise

        # --------------------------------------------------
        # RA histogram
        # --------------------------------------------------
        axes[0].hist(
            ra_data,
            bins=bins,
            histtype='step',
            linewidth=2,
            density=density,
            label='Data'
        )

        axes[0].hist(
            ra_rand,
            bins=bins,
            histtype='step',
            linewidth=2,
            density=density,
            label='Randoms'
        )

        axes[0].set_xlabel('RA [deg]')
        axes[0].set_ylabel('Density' if density else 'Counts')
        axes[0].set_title('RA distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        # --------------------------------------------------
        # Dec histogram
        # --------------------------------------------------
        axes[1].hist(
            dec_data,
            bins=bins,
            histtype='step',
            linewidth=2,
            density=density,
            label='Data'
        )

        axes[1].hist(
            dec_rand,
            bins=bins,
            histtype='step',
            linewidth=2,
            density=density,
            label='Randoms'
        )

        axes[1].set_xlabel('Dec [deg]')
        axes[1].set_ylabel('Density' if density else 'Counts')
        axes[1].set_title('Dec distribution')
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved histogram diagnostic to {save_path}")

    def plot_ra_dec_density(
        self,
        ra_data,
        dec_data,
        ra_rand,
        dec_rand,
        selection,
        bins=200,
    ):
        """
        Save-only 2D density diagnostic.
        """

        save_path = self._get_diagnostic_plot_path(
            selection,
            "density2d"
        )

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # --------------------------------------------------
        # Data density
        # --------------------------------------------------
        h1 = axes[0, 0].hist2d(
            ra_data,
            dec_data,
            bins=bins,
        )

        axes[0, 0].set_title('Data')
        axes[0, 0].set_xlabel('RA [deg]')
        axes[0, 0].set_ylabel('Dec [deg]')
        fig.colorbar(h1[3], ax=axes[0, 0])

        # --------------------------------------------------
        # Random density
        # --------------------------------------------------
        h2 = axes[0, 1].hist2d(
            ra_rand,
            dec_rand,
            bins=bins,
        )

        axes[0, 1].set_title('Randoms')
        axes[0, 1].set_xlabel('RA [deg]')
        axes[0, 1].set_ylabel('Dec [deg]')
        fig.colorbar(h2[3], ax=axes[0, 1])

        # --------------------------------------------------
        # Difference map
        # --------------------------------------------------
        data_hist, xedges, yedges = np.histogram2d(
            ra_data,
            dec_data,
            bins=bins
        )

        rand_hist, _, _ = np.histogram2d(
            ra_rand,
            dec_rand,
            bins=[xedges, yedges]
        )

        data_hist /= np.sum(data_hist)
        rand_hist /= np.sum(rand_hist)

        diff = data_hist - rand_hist

        im = axes[1, 0].imshow(
            diff.T,
            origin='lower',
            aspect='auto',
            extent=[
                xedges[0], xedges[-1],
                yedges[0], yedges[-1]
            ],
        )

        axes[1, 0].set_title('Data - Randoms')
        axes[1, 0].set_xlabel('RA [deg]')
        axes[1, 0].set_ylabel('Dec [deg]')
        fig.colorbar(im, ax=axes[1, 0])

        # --------------------------------------------------
        # Scatter comparison
        # --------------------------------------------------
        step_data = max(1, len(ra_data) // 50000)
        step_rand = max(1, len(ra_rand) // 50000)

        axes[1, 1].scatter(
            ra_rand[::step_rand],
            dec_rand[::step_rand],
            s=1,
            alpha=0.3,
            label='Randoms'
        )

        axes[1, 1].scatter(
            ra_data[::step_data],
            dec_data[::step_data],
            s=1,
            alpha=0.3,
            label='Data'
        )

        axes[1, 1].set_title('Scatter comparison')
        axes[1, 1].set_xlabel('RA [deg]')
        axes[1, 1].set_ylabel('Dec [deg]')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

        print(f"  Saved density diagnostic to {save_path}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

# Colour keyed on target_selection value
_COLOUR_BY_TARGET = {
    'star':               'red',
    'galaxy':             'blue',
    'galaxy/ambiguous':   'green',
}
_DEFAULT_COLOUR = 'grey'   # fallback for unrecognised target_selection values

# Line style keyed on star_gal_method value
_LINESTYLE_BY_METHOD = {
    'TOPZ/SFM/R50': '-',    # solid
    'baseline':     '--',   # dashed
    'UMAP':         ':',    # double-dashed (dotted)
}
_DEFAULT_LINESTYLE = '-'    # fallback


def _colour_for(selection: dict) -> str:
    target = selection.get('target_selection', '')
    return _COLOUR_BY_TARGET.get(target, _DEFAULT_COLOUR)


def _linestyle_for(selection: dict) -> str:
    method = selection.get('star_gal_method', '')
    return _LINESTYLE_BY_METHOD.get(method, _DEFAULT_LINESTYLE)


def _label_for(selection: dict, title_keys: set) -> str:
    """
    Build a legend label from *selection*, omitting:
      - keys whose value appears in title_keys (already shown in the panel title)
      - keys whose value is None

    Only the VALUES are shown (no 'key=' prefix).
    """
    parts = [
        str(v)
        for k, v in selection.items()
        if v is not None and str(v) not in title_keys
    ]
    return ', '.join(parts) if parts else 'default'


def _build_panel_title(panel_results: list) -> tuple[str, set]:
    """
    For a list of result dicts sharing a panel, identify which keys have only a
    single unique value across all results.  Those values go into the panel
    title (as bare values, no key names).  Returns (title_string, title_value_set).
    """
    if not panel_results:
        return '', set()

    # Collect all unique values per key across every result on this panel
    from collections import defaultdict
    values_per_key = defaultdict(set)
    for r in panel_results:
        for k, v in r.get('selection', {}).items():
            if v is not None:
                values_per_key[k].add(str(v))

    # Keys with exactly one unique value → go in the title
    title_parts = [
        next(iter(vals))
        for vals in values_per_key.values()
        if len(vals) == 1
    ]
    title_value_set = set(title_parts)
    title = ', '.join(title_parts)
    return title, title_value_set


class AngularClusteringPlots:
    def __init__(self, clustering_results, num_panels, save_location=None, log_scale=True):
        """
        Parameters
        ----------
        clustering_results : list of dict
            Each dict is a result as returned by AngularClustering (i.e. has
            keys 'selection' and 'columns').
        num_panels : int
            Number of subplot panels to create.
        save_location : str or None
            If given, the figure is saved here instead of shown.
        log_scale : bool
            Whether to use a logarithmic scale for the x and y-axis.
        """
        self.clustering_results = clustering_results
        self.save_location = save_location
        self.num_panels = num_panels
        self.log_scale = log_scale

        # selections_per_panel maps panel index -> list of result dicts to plot.
        # Populated via assign_results_to_panel().
        self.selections_per_panel = {panel: [] for panel in range(num_panels)}

    def assign_results_to_panel(self, panel_index, selection_filters):
        """
        Assign clustering results matching *all* key/value pairs in
        selection_filters to a specific panel.

        Each value in selection_filters may be either:
          - a single value   e.g. 'no ghostmask'
          - a list of values e.g. ['galaxy', 'galaxy/ambiguous', 'star']

        A result matches a key if its selection[key] is equal to the scalar
        value, or is contained in the list of values.

        Parameters
        ----------
        panel_index : int
        selection_filters : dict
            e.g. {
                'target_selection':    ['galaxy', 'star'],
                'survey_depth':        ['Z<21.1'],
                'ghostmask_selection': 'no ghostmask',
            }
        """
        if panel_index not in self.selections_per_panel:
            raise ValueError(
                f"panel_index {panel_index} out of range (0..{self.num_panels - 1})"
            )

        def _matches(result_selection, filters):
            for k, allowed in filters.items():
                val = result_selection.get(k)
                if isinstance(allowed, list):
                    if val not in allowed:
                        return False
                else:
                    if val != allowed:
                        return False
            return True

        matched = [
            r for r in self.clustering_results
            if _matches(r.get('selection', {}), selection_filters)
        ]
        self.selections_per_panel[panel_index].extend(matched)

    # ---------------------------------------------------------------------- #

    def plot_correlation_figure(self, ncols=None, figsize=None):
        """
        Draw all panels in a single figure.

        Parameters
        ----------
        ncols : int or None
            Number of columns in the subplot grid. Defaults to num_panels
            (single row).
        figsize : tuple or None
            Passed to plt.subplots.
        """
        ncols = ncols or self.num_panels
        nrows = int(np.ceil(self.num_panels / ncols))
        figsize = figsize or (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False, sharex=True, sharey=True, constrained_layout=True)
        axes_flat = axes.flatten()

        for panel_idx in range(self.num_panels):
            ax = axes_flat[panel_idx]
            results_for_panel = self.selections_per_panel[panel_idx]
            if results_for_panel:
                self._plot_correlation_function_subplot(ax, results_for_panel)
            else:
                ax.set_visible(False)

        # Hide any unused axes beyond num_panels
        for ax in axes_flat[self.num_panels:]:
            ax.set_visible(False)

        for i, ax in enumerate(axes_flat[:self.num_panels]):
            if not ax.get_visible():
                continue
            row = i // ncols
            col = i % ncols

            if col == 0:
                ax.set_ylabel(r'$w(\theta)$')

            # Label x-axis if there's no visible panel directly below
            next_row_idx = i + ncols
            bottom_edge = (next_row_idx >= self.num_panels)
            if bottom_edge:
                ax.set_xlabel(r'$\theta$ [degrees]')

        if self.save_location:
            fig.savefig(self.save_location, dpi=150, bbox_inches='tight')
            print(f"Figure saved to {self.save_location}")
        else:
            plt.show()

        return fig, axes

    def _plot_correlation_function_subplot(self, ax, clustering_result_per_plot):
        """
        Plot one or more w(theta) curves on a single Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
        clustering_result_per_plot : list of dict
            Each dict has keys 'selection' and 'columns'
            (with sub-keys 'xi', 'varxi', 'meanlogr').
        """
        # Work out which values are shared across all results on this panel so
        # they can be shown in the title rather than repeated in every label.
        panel_title, title_value_set = _build_panel_title(clustering_result_per_plot)

        for result in clustering_result_per_plot:
            columns  = result['columns']
            xi       = np.array(columns['xi'])
            varxi    = np.array(columns['varxi'])
            meanlogr = np.array(columns['meanlogr'])

            r   = np.exp(meanlogr)
            sel = result.get('selection', {})

            colour    = _colour_for(sel)
            linestyle = _linestyle_for(sel)
            label     = _label_for(sel, title_value_set)

            # Only plot positive xi values on a log-log scale
            if self.log_scale:
                pos_mask = xi > 0
                if not np.any(pos_mask):
                    print(f"  Warning: no positive xi values for selection {sel}. Skipping.")
                    continue
            else:                
                pos_mask = np.ones_like(xi, dtype=bool)

            ax.plot(
                r[pos_mask], xi[pos_mask],
                label=label,
                color=colour,
                linestyle=linestyle,
            )
            ax.errorbar(
                r[pos_mask], xi[pos_mask],
                yerr=np.sqrt(varxi[pos_mask]),
                lw=1.5, alpha = 0.25, ls='', color=colour,
            )
        if self.log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(0.01, 10)
        else:
            ax.set_xlim(0.1, 3)
        #ax.set_xlabel(r'$\theta$ [degrees]')
        #ax.set_ylabel(r'$w(\theta)$')
        ax.legend(fontsize=7)
        ax.grid()

        if panel_title:
            ax.set_title(panel_title, fontsize=8)