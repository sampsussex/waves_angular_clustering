import numpy as np
import matplotlib.pyplot as plt
import treecorr
import pandas as pd
import json
import os
import itertools

# Keys from the selection dict that uniquely determine which randoms are loaded.
# target_selection / survey_depth / star_gal_method only affect the data catalogue,
# so RR can be shared across all selections that agree on these two keys.
_RR_CACHE_KEYS = ('region', 'ghostmask_selection')


class AngularClustering:
    def __init__(self, ra_cat, dec_cat, ra_rand, dec_rand, selection_dic,
                 min_sep=0.01, max_sep=10, nbins=100, sep_units='degrees',
                 cat_units='degrees', rand_units='degrees',
                 n_patch=10, var_method='jackknife'):
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

    def do_correlations(self, precomputed_rr=None):
        """
        Run DD, DR, and RR correlations and compute xi.
        TODO: Get rid of precomuted stuff, patches makes this not work. 

        Parameters
        ----------
        precomputed_rr : treecorr.NNCorrelation or None
            If supplied, skip the RR processing step and use this object
            directly. The caller is responsible for ensuring it was computed
            with identical binning settings and the same randoms catalogue.
            
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
        # RR correlation files are stored in a sub-directory of results_directory.
        # They are named by the subset of selection keys that affect the randoms,
        # so they can be reused across selections that differ only in data-side cuts.
        self.rr_cache_dir = os.path.join(results_directory, 'rr_cache') if results_directory else None

        # Treecorr binning settings — shared by all AngularClustering instances
        # and used when reconstructing an RR object from cache.
        self.min_sep   = 0.01
        self.max_sep   = 10
        self.nbins     = 100
        self.sep_units = 'degrees'

        self.data_ra_col = 'RAGAIA'
        self.data_dec_col = 'DecGAIA'
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
            'target_selection':   ['galaxy', 'galaxy/ambiguous', 'star'],
            'ghostmask_selection':['no ghostmask', 'with ghostmask'],
            'survey_depth':       ['Z<21.1'],
            'star_gal_method':    ['TOPZ/SFM/R50'],
            'region':             ['WWN'],
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
    # RR cache helpers
    # ---------------------------------------------------------------------- #

    @staticmethod
    def _rr_cache_key(selection):
        """
        Return the subset of the selection dict that uniquely identifies
        which randoms catalogue was used, and therefore which RR file applies.
        """
        return {k: selection[k] for k in _RR_CACHE_KEYS if k in selection}

    @staticmethod
    def _rr_key_to_filename(rr_key):
        """Convert an RR cache-key dict to a safe filename (FITS format)."""
        parts = [f"{k}={v}" for k, v in sorted(rr_key.items())]
        name = "__".join(parts).replace(' ', '_').replace('<', 'lt').replace('/', '-')
        return f"rr__{name}.fits"

    def _get_rr_cache_path(self, selection):
        """Full path for the RR FITS file corresponding to this selection."""
        rr_key = self._rr_cache_key(selection)
        return os.path.join(self.rr_cache_dir, self._rr_key_to_filename(rr_key))

    def _rr_cache_exists(self, selection):
        """Return True if a cached RR file exists for this selection's randoms."""
        return os.path.isfile(self._get_rr_cache_path(selection))

    def _load_rr_from_cache(self, selection):
        """
        Load and return a treecorr.NNCorrelation from the RR cache.
        The correlation config must match; treecorr will raise if it doesn't.
        """
        path = self._get_rr_cache_path(selection)
        print(f"  Loading cached RR from {path}")
        rr = treecorr.NNCorrelation(
            min_sep=self.min_sep,
            max_sep=self.max_sep,
            nbins=self.nbins,
            sep_units=self.sep_units,
        )
        rr.read(path)
        return rr

    def _save_rr_to_cache(self, rr, selection):
        """Write an RR NNCorrelation object to the cache directory as FITS."""
        os.makedirs(self.rr_cache_dir, exist_ok=True)
        path = self._get_rr_cache_path(selection)
        rr.write(path)
        print(f"  Cached RR saved to {path}")

    def _get_or_compute_rr(self, rand_cat, selection, ac_instance):
        """
        Return an RR NNCorrelation, loading from cache if available or
        computing (and caching) it if not.

        Parameters
        ----------
        rand_cat : treecorr.Catalog
            The randoms catalogue already built by the AngularClustering instance.
        selection : dict
        ac_instance : AngularClustering
            Used to read binning parameters for constructing the RR object.
        """
        if self._rr_cache_exists(selection):
            return self._load_rr_from_cache(selection)

        print(f"  Computing RR for cache key: {self._rr_cache_key(selection)}")
        rr = treecorr.NNCorrelation(
            min_sep=ac_instance.min_sep,
            max_sep=ac_instance.max_sep,
            nbins=ac_instance.nbins,
            sep_units=ac_instance.sep_units,
        )
        rr.process(rand_cat)
        self._save_rr_to_cache(rr, selection)
        return rr

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

        df_sel = df.loc[base_selection]

        if len(df_sel) == 0:
            raise ValueError(
                f"Dataset is empty after applying selection: {selection}"
            )

        ra_data = df_sel[self.data_ra_col].values
        dec_data = df_sel[self.data_dec_col].values

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

        df_sel = df.loc[base_selection]

        if len(df_sel) == 0:
            raise ValueError(
                f"Randoms catalogue is empty after applying selection: {selection}"
            )

        ra_randoms = df_sel[self.randoms_ra_col].values
        dec_randoms = df_sel[self.randoms_dec_col].values
        print(f"  Loaded {len(ra_randoms)} random points after selection.")
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

        print("  Initialising AngularClustering instance...")
        ac = AngularClustering(
            ra_cat=ra_data, dec_cat=dec_data,
            ra_rand=ra_rand, dec_rand=dec_rand,
            selection_dic=selection,
            min_sep=self.min_sep, max_sep=self.max_sep,
            nbins=self.nbins, sep_units=self.sep_units,
        )
        print("  Computing correlations...")
        print("  Checking for cached RR...")
        rr = self._get_or_compute_rr(ac.rand_cat, selection, ac)
        print("  Computing DD, DR, and xi using RR...")
        ac.do_correlations(precomputed_rr=rr)
        print("  Clustering computation complete.")
        save_path = self._get_results_path(selection)
        print(f"  Saving results to {save_path}...")
        os.makedirs(self.results_directory, exist_ok=True)
        ac.save_results(save_path)
        print(f"  Saved to {save_path}")

        results = ac.results
        print("  Cleaning up treecorr catalogs from memory...")
        ac.clean_up()
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


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

# Default colour cycle used when no colour is specified per panel entry
_DEFAULT_COLOURS = plt.rcParams['axes.prop_cycle'].by_key()['color']


class AngularClusteringPlots:
    def __init__(self, clustering_results, num_panels, save_location=None):
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
        """
        self.clustering_results = clustering_results
        self.save_location = save_location
        self.num_panels = num_panels

        # selections_per_panel maps panel index -> list of result dicts to plot.
        # Populated via assign_results_to_panel().
        self.selections_per_panel = {panel: [] for panel in range(num_panels)}

    def assign_results_to_panel(self, panel_index, selection_filters):
        """
        Assign clustering results matching all key/value pairs in
        selection_filters to a specific panel.

        Parameters
        ----------
        panel_index : int
        selection_filters : dict
            e.g. {'region': 'WWN', 'survey_depth': 'Z<21.1'}
        """
        if panel_index not in self.selections_per_panel:
            raise ValueError(f"panel_index {panel_index} out of range (0..{self.num_panels-1})")

        matched = [
            r for r in self.clustering_results
            if all(r['selection'].get(k) == v for k, v in selection_filters.items())
        ]
        self.selections_per_panel[panel_index].extend(matched)

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

        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
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

        plt.tight_layout()

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
        for idx, result in enumerate(clustering_result_per_plot):
            columns = result['columns']
            xi      = np.array(columns['xi'])
            varxi   = np.array(columns['varxi'])
            meanlogr = np.array(columns['meanlogr'])

            r = np.exp(meanlogr)
            colour = _DEFAULT_COLOURS[idx % len(_DEFAULT_COLOURS)]

            # Build a human-readable label from the selection dict
            sel = result.get('selection', {})
            label = ', '.join(f"{k}={v}" for k, v in sel.items())

            # Only plot positive xi values on a log-log scale
            pos_mask = xi > 0
            if not np.any(pos_mask):
                print(f"  Warning: no positive xi values for selection {sel}. Skipping.")
                continue

            ax.plot(r[pos_mask], xi[pos_mask], label=label, color=colour)
            ax.errorbar(
                r[pos_mask], xi[pos_mask],
                yerr=np.sqrt(varxi[pos_mask]),
                lw=0.1, ls='', color=colour
            )

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$w(\theta)$')
        ax.legend(fontsize=7)
        ax.set_xlim(0.01, 10)
        ax.grid(True, which='both', ls='--', alpha=0.4)