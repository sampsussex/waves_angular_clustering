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

        self.dd = dd   # expose for diagnostics
        self.dr = dr
        self.rr = rr

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
        self.randoms_realisation_to_load = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

        self.results_directory = results_directory

        # Treecorr binning settings — shared by all AngularClustering instances
        # and used when reconstructing an RR object from cache.
        self.min_sep   = 0.01
        self.max_sep   = 10
        self.nbins     = 50
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
        self.columns_to_load_randoms = [
            self.randoms_ra_col, self.randoms_dec_col,
            'starmask', 'ghostmask', 'polygon_mask', 'realisation'
        ]

        # ------------------------------------------------------------------ #
        # Selection definitions
        # ------------------------------------------------------------------ #
        self.possible_selections = {
            'target_selection':   ['galaxy', 'galaxy/ambiguous', 'star'],
            'ghostmask_selection':['no ghostmask', 'with ghostmask'],
            'survey_depth':       ['Z<21.1', 'Z<21.25', 'Z<22'],
            'star_gal_method':    ['TOPZ/SFM/R50', 'baseline'],
            'region':             ['WWN', 'WWS', 'WW combined'],
        }

        selections_to_run = {
            'target_selection':   ['galaxy'],
            'ghostmask_selection':['no ghostmask', 'with ghostmask'],
            'survey_depth':       ['Z<21.1'],
            'star_gal_method':    ['TOPZ/SFM/R50'],
            'region':             ['WWN', 'WWS'],
        }

        self._validate_selections(selections_to_run)

        self.selections_to_run = self._expand_selections(selections_to_run)

        # Number of RA strips for the jackknife-style spatial split
        self.n_ra_strips = 10

        self.extra_rec_masks = [
            [[165.9, 165.95], [-3.95, -3.7]],
            [[215.4, 215.5], [3.7, 3.95]],
            [[17.85, 17.95], [-30.15, -30.05]],
            [[18.4, 18.5], [-31.80, -31.70]]
        ]

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

    def _get_results_path(self, selection, strip_index=None):
        """
        Return the save path for a given selection.

        Parameters
        ----------
        selection : dict
        strip_index : int or None
            If given, appends ``__strip_N`` to the filename stem so that each
            RA strip is saved as a separate file.
        """
        base = self._selection_to_filename(selection)
        if strip_index is not None:
            # Insert strip suffix before the .json extension
            base = base.replace('.json', f'__strip_{strip_index}.json')
        return os.path.join(self.results_directory, base)

    def _check_if_results_exist(self, selection, strip_index=None):
        """Return True if results have already been saved for this selection (and strip)."""
        return os.path.isfile(self._get_results_path(selection, strip_index=strip_index))

    def _check_if_all_strips_exist(self, selection):
        """Return True only when every strip result file already exists on disk."""
        return all(
            self._check_if_results_exist(selection, strip_index=i)
            for i in range(self.n_ra_strips)
        )

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
    # RA wrapping
    # ---------------------------------------------------------------------- #

    def _wrap_ra_for_region(self, ra, selection):
        """
        For WWS, wrap RA values > 180 deg into negative RA values so that the
        survey footprint is contiguous in the wrapped coordinate system.

        Example:  359 -> -1,  270 -> -90,  181 -> -179
        """
        ra = np.asarray(ra).copy()
        if selection.get("region") == "WWS":
            ra[ra > 180] -= 360
        return ra

    def _unwrap_ra(self, ra_wrapped, selection):
        """
        Reverse _wrap_ra_for_region: bring negative RA values back to [0, 360).
        """
        ra = np.asarray(ra_wrapped).copy()
        if selection.get("region") == "WWS":
            ra[ra < 0] += 360
        return ra

    # ---------------------------------------------------------------------- #
    # RA-strip splitting
    # ---------------------------------------------------------------------- #

    def _split_catalog_by_ra_strips(self, ra_data, dec_data, ra_rand, dec_rand,
                                    selection, n_strips=None):
        """
        Split data and randoms catalogs into *n_strips* equal-width RA strips.

        RA wrapping is applied before computing strip edges so that WWS sources
        that straddle RA=0 are handled correctly, then unwrapped before
        returning so that treecorr always receives native (0–360) coordinates.

        Parameters
        ----------
        ra_data, dec_data : ndarray
        ra_rand, dec_rand : ndarray
        selection : dict
        n_strips : int or None
            Defaults to self.n_ra_strips.

        Yields
        ------
        strip_index : int
        ra_data_strip, dec_data_strip, ra_rand_strip, dec_rand_strip : ndarray
            Native (unwrapped) RA coordinates and corresponding Dec arrays for
            the objects that fall inside this RA strip.
        """
        if n_strips is None:
            n_strips = self.n_ra_strips

        # Work in wrapped coordinates so the footprint is contiguous
        ra_data_w = self._wrap_ra_for_region(ra_data, selection)
        ra_rand_w = self._wrap_ra_for_region(ra_rand, selection)

        # Derive strip edges from the *combined* RA extent of data + randoms
        # so that every strip is the same angular width regardless of whether
        # the data or the randoms happen to extend slightly further.
        ra_all_w  = np.concatenate([ra_data_w, ra_rand_w])
        ra_min_w  = ra_all_w.min()
        ra_max_w  = ra_all_w.max()
        edges     = np.linspace(ra_min_w, ra_max_w, n_strips + 1)

        print(f"  RA strip edges (wrapped): {edges}")

        for i in range(n_strips):
            lo, hi = edges[i], edges[i + 1]

            # Include the upper boundary only for the last strip so that no
            # source is double-counted.
            if i < n_strips - 1:
                data_mask = (ra_data_w >= lo) & (ra_data_w < hi)
                rand_mask = (ra_rand_w >= lo) & (ra_rand_w < hi)
            else:
                data_mask = (ra_data_w >= lo) & (ra_data_w <= hi)
                rand_mask = (ra_rand_w >= lo) & (ra_rand_w <= hi)

            n_data = data_mask.sum()
            n_rand = rand_mask.sum()

            if n_data == 0:
                print(f"  Strip {i}: no data points in RA [{lo:.3f}, {hi:.3f}] — skipping.")
                continue
            if n_rand == 0:
                print(f"  Strip {i}: no random points in RA [{lo:.3f}, {hi:.3f}] — skipping.")
                continue

            print(f"  Strip {i}: RA [{lo:.3f}, {hi:.3f}]  data={n_data}  randoms={n_rand}")

            # Unwrap back to native coordinates before handing to treecorr
            ra_data_strip = self._unwrap_ra(ra_data_w[data_mask], selection)
            ra_rand_strip = self._unwrap_ra(ra_rand_w[rand_mask], selection)
            dec_data_strip = dec_data[data_mask]
            dec_rand_strip = dec_rand[rand_mask]

            yield i, ra_data_strip, dec_data_strip, ra_rand_strip, dec_rand_strip

    # ---------------------------------------------------------------------- #
    # Data loading
    # ---------------------------------------------------------------------- #

    def _get_extra_rec_masks(self, ra, dec):
        """Return boolean mask excluding regions in self.extra_rec_masks."""
        if not self.extra_rec_masks:
            return np.ones(len(ra), dtype=bool)

        mask = np.ones(len(ra), dtype=bool)

        for (ramin, ramax), (decmin, decmax) in self.extra_rec_masks:
            mask &= ~(
                (ra >= ramin) & (ra <= ramax) &
                (dec >= decmin) & (dec <= decmax)
            )

        return mask

    def _load_dataset(self, photom_filepath, stargal_filepath, selection):
        print(f"  Loading photometric data from {photom_filepath}...")
        df = pd.read_parquet(photom_filepath, columns=self.columns_to_load_photom)
        print(f"  Loaded {len(df)} rows from photometric catalogue.")
        df['uberID'] = df['uberID'].astype(np.int64)

        # ------------------------------------------------------------------ #
        # Star/galaxy separation
        # ------------------------------------------------------------------ #
        df['stargal'] = np.nan

        if selection['star_gal_method'] == 'TOPZ/SFM/R50':
            print(f"  Loading stargal classification from {stargal_filepath}...")
            df_stargal = pd.read_parquet(stargal_filepath, columns=self.columns_to_load_stargal)
            df_stargal['uberID'] = df_stargal['uberID'].astype(np.int64)
            print(f"  Loaded {len(df_stargal)} rows from stargal catalogue.")
            df = df.merge(df_stargal, on='uberID', how='left', suffixes=('', '_ext'))
            del df_stargal
            if 'stargal_ext' in df.columns:
                df['stargal'] = df['stargal_ext']
                df.drop(columns=['stargal_ext'], inplace=True)

        elif selection['star_gal_method'] == 'baseline':
            df['stargal'] = df['class']

        # ------------------------------------------------------------------ #
        # Build selection mask
        # ------------------------------------------------------------------ #
        base_selection = (
            (df['duplicate'] == False) &
            (df['mask'] == False) &
            (df['starmask'] == False)
        )
        extra_rec_masks = self._get_extra_rec_masks(
            df[self.data_ra_col].to_numpy(), df[self.data_dec_col].to_numpy()
        )
        base_selection &= extra_rec_masks

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
        del df

        if len(df_sel) == 0:
            raise ValueError(
                f"Dataset is empty after applying selection: {selection}"
            )

        ra_data  = df_sel[self.data_ra_col].to_numpy(copy=True)
        dec_data = df_sel[self.data_dec_col].to_numpy(copy=True)
        del df_sel

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

        extra_rec_masks = self._get_extra_rec_masks(
            df[self.randoms_ra_col].to_numpy(), df[self.randoms_dec_col].to_numpy()
        )
        base_selection &= extra_rec_masks

        if selection['ghostmask_selection'] == 'with ghostmask':
            base_selection &= df['ghostmask'] == False

        df_sel = df.loc[base_selection].copy()
        del df
        if len(df_sel) == 0:
            raise ValueError(
                f"Randoms catalogue is empty after applying selection: {selection}"
            )

        ra_randoms  = df_sel[self.randoms_ra_col].to_numpy(copy=True)
        dec_randoms = df_sel[self.randoms_dec_col].to_numpy(copy=True)
        print(f"  Loaded {len(ra_randoms)} random points after selection.")
        del df_sel
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
        Return a list of (selection, strip_index, result) tuples for strips
        that already have saved output on disk.  Also removes fully-completed
        selections from self.selections_to_run so they are not recomputed.
        """
        already_run   = []   # list of result dicts
        remaining     = []   # selections still needing at least one strip run

        for selection in self.selections_to_run:
            if self._check_if_all_strips_exist(selection):
                for i in range(self.n_ra_strips):
                    path = self._get_results_path(selection, strip_index=i)
                    if os.path.isfile(path):
                        already_run.append(self.load_results(path))
            else:
                remaining.append(selection)

        self.selections_to_run = remaining
        return already_run

    # ---------------------------------------------------------------------- #
    # Clustering runners
    # ---------------------------------------------------------------------- #

    def get_clustering_for_selection(self, selection):
        """
        Run the angular clustering pipeline for a single selection dict,
        splitting the catalogs into RA strips and saving each strip's result
        to a separate file.  Returns a list of result dicts (one per strip).
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
        self.plot_ra_dec_histograms(ra_data, dec_data, ra_rand, dec_rand, selection)
        print("  Saved RA/Dec histogram diagnostics.")
        self.plot_ra_dec_density(ra_data, dec_data, ra_rand, dec_rand, selection)
        print("  Initial diagnostics complete.")

        strip_results = []

        for strip_index, ra_d, dec_d, ra_r, dec_r in self._split_catalog_by_ra_strips(
            ra_data, dec_data, ra_rand, dec_rand, selection
        ):
            # Skip strips whose results are already on disk
            if self._check_if_results_exist(selection, strip_index=strip_index):
                print(f"  Strip {strip_index}: results already exist — loading from disk.")
                result = self.load_results(
                    self._get_results_path(selection, strip_index=strip_index)
                )
                strip_results.append(result)
                continue

            print(f"  Strip {strip_index}: initialising AngularClustering instance...")

            # Embed the strip index in the selection metadata so it is stored
            # alongside the correlation function in the JSON file.
            selection_with_strip = dict(selection, ra_strip=strip_index)

            ac = AngularClustering(
                ra_cat=ra_d, dec_cat=dec_d,
                ra_rand=ra_r, dec_rand=dec_r,
                selection_dic=selection_with_strip,
                min_sep=self.min_sep, max_sep=self.max_sep,
                nbins=self.nbins, sep_units=self.sep_units,
            )
            print(f"  Strip {strip_index}: computing DD, DR, RR, and xi...")
            ac.do_correlations()
            print(f"  Strip {strip_index}: clustering computation complete.")

            print(f"  Strip {strip_index}: saving DD/DR/RR diagnostic...")
            self.plot_dd_dr_rr(ac.dd, ac.dr, ac.rr, selection_with_strip)

            save_path = self._get_results_path(selection, strip_index=strip_index)
            os.makedirs(self.results_directory, exist_ok=True)
            ac.save_results(save_path)
            print(f"  Strip {strip_index}: results saved to {save_path}")

            strip_results.append(ac.results)
            ac.clean_up()
            del ac

        del ra_data, dec_data, ra_rand, dec_rand
        print(f"  All {len(strip_results)} strips complete for selection: {selection}")
        print("-" * 50)
        return strip_results

    def get_clustering_for_all_selections_to_run(self):
        """
        Run the angular clustering pipeline for every selection in
        self.selections_to_run. Skips any for which all strip results already
        exist.  Returns a flat list of result dicts (one per strip, all
        selections combined).
        """
        all_results = self.get_previously_run_results()

        for selection in self.selections_to_run:
            try:
                strip_results = self.get_clustering_for_selection(selection)
                all_results.extend(strip_results)
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
        return os.path.join(self._get_diagnostics_directory(), filename)

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
        save_path = self._get_diagnostic_plot_path(selection, "hist1d")

        ra_data_plot = self._wrap_ra_for_region(ra_data, selection)
        ra_rand_plot = self._wrap_ra_for_region(ra_rand, selection)

        fig, axes = plt.subplots(1, 2, figsize=(5, 5))
        density = normalise

        axes[0].hist(ra_data_plot, bins=bins, histtype='step', linewidth=2,
                     density=density, label='Data')
        axes[0].hist(ra_rand_plot, bins=bins, histtype='step', linewidth=2,
                     density=density, label='Randoms')
        axes[0].set_xlabel('RA [deg]')
        axes[0].set_ylabel('Density' if density else 'Counts')
        axes[0].set_title('RA distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        axes[1].hist(dec_data, bins=bins, histtype='step', linewidth=2,
                     density=density, label='Data')
        axes[1].hist(dec_rand, bins=bins, histtype='step', linewidth=2,
                     density=density, label='Randoms')
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
        bins=500,
    ):
        """
        Save-only 2D density diagnostic.
        """
        save_path = self._get_diagnostic_plot_path(selection, "density2d")
        ra_data_plot = self._wrap_ra_for_region(ra_data, selection)
        ra_rand_plot = self._wrap_ra_for_region(ra_rand, selection)

        ra_all  = np.concatenate([ra_data_plot, ra_rand_plot])
        dec_all = np.concatenate([dec_data,     dec_rand])

        ra_min,  ra_max  = ra_all.min(),  ra_all.max()
        dec_min, dec_max = dec_all.min(), dec_all.max()

        ra_range  = ra_max  - ra_min
        dec_range = dec_max - dec_min

        bin_deg    = ra_range / bins
        n_ra_bins  = bins
        n_dec_bins = max(1, int(np.round(dec_range / bin_deg)))

        x_bins = np.linspace(ra_min,  ra_max,  n_ra_bins  + 1)
        y_bins = np.linspace(dec_min, dec_max, n_dec_bins + 1)

        aspect_ratio    = n_ra_bins / n_dec_bins
        min_panel_height = 4.0
        panel_height    = max(min_panel_height, 18.0 / aspect_ratio)
        panel_width     = panel_height * aspect_ratio

        fig, axes = plt.subplots(3, 1, figsize=(panel_width, panel_height * 3))

        h1 = axes[0].hist2d(ra_data_plot, dec_data, bins=[x_bins, y_bins], cmap='coolwarm')
        axes[0].set_aspect('equal')
        axes[0].set_title('Data')
        axes[0].set_xlabel('RA [deg]')
        axes[0].set_ylabel('Dec [deg]')
        fig.colorbar(h1[3], ax=axes[0])

        h2 = axes[1].hist2d(ra_rand_plot, dec_rand, bins=[x_bins, y_bins], cmap='coolwarm')
        axes[1].set_aspect('equal')
        axes[1].set_title('Randoms')
        axes[1].set_xlabel('RA [deg]')
        axes[1].set_ylabel('Dec [deg]')
        fig.colorbar(h2[3], ax=axes[1])

        data_hist, xedges, yedges = np.histogram2d(
            ra_data_plot, dec_data, bins=[x_bins, y_bins]
        )
        rand_hist, _, _ = np.histogram2d(
            ra_rand_plot, dec_rand, bins=[xedges, yedges]
        )

        data_sum = np.sum(data_hist)
        rand_sum = np.sum(rand_hist)
        if data_sum > 0:
            data_hist = data_hist / data_sum
        if rand_sum > 0:
            rand_hist = rand_hist / rand_sum

        diff = data_hist - rand_hist
        im = axes[2].imshow(
            diff.T,
            origin='lower',
            aspect='equal',
            cmap='coolwarm',
            extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
        )
        axes[2].set_title('Data - Randoms')
        axes[2].set_xlabel('RA [deg]')
        axes[2].set_ylabel('Dec [deg]')
        fig.colorbar(im, ax=axes[2])

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved density diagnostic to {save_path}")

    def plot_dd_dr_rr(self, dd, dr, rr, selection):
        """
        Plot and save DD, DR, RR pair counts (weight and npairs) as a function
        of mean separation, and save the raw values to a JSON file.
        Both are written to the diagnostics directory.
        """
        diag_dir = self._get_diagnostics_directory()
        base = self._selection_to_filename(selection).replace(".json", "")

        raw = {}
        for label, corr in [('DD', dd), ('DR', dr), ('RR', rr)]:
            raw[label] = {
                'meanr':    corr.meanr.tolist(),
                'meanlogr': corr.meanlogr.tolist(),
                'weight':   corr.weight.tolist(),
                'npairs':   corr.npairs.tolist(),
            }

        raw_path = os.path.join(diag_dir, f"{base}__dd_dr_rr_raw.json")
        with open(raw_path, 'w') as f:
            json.dump(raw, f, indent=2)
        print(f"  Saved DD/DR/RR raw values to {raw_path}")

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Pair counts: DD / DR / RR\n" +
            " | ".join(f"{k}={v}" for k, v in selection.items()),
            fontsize=9
        )

        colours = {'DD': 'steelblue', 'DR': 'darkorange', 'RR': 'seagreen'}

        for label, corr in [('DD', dd), ('DR', dr), ('RR', rr)]:
            sep = corr.meanr
            axes[0].plot(sep, corr.weight, marker='o', ms=3, lw=1,
                        color=colours[label], label=label)
            axes[1].plot(sep, corr.npairs, marker='o', ms=3, lw=1,
                        color=colours[label], label=label)

        for ax, ylabel, title in zip(
            axes,
            ['Weighted pair counts  (weight)', 'Raw pair counts  (npairs)'],
            ['weight', 'npairs'],
        ):
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Mean separation (degrees)')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(framealpha=0.7)
            ax.grid(True, which='both', ls=':', alpha=0.4)

        plt.tight_layout()
        plot_path = os.path.join(diag_dir, f"{base}__dd_dr_rr.png")
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved DD/DR/RR diagnostic plot to {plot_path}")


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #

_COLOUR_BY_TARGET = {
    'star':               'red',
    'galaxy':             'blue',
    'galaxy/ambiguous':   'green',
}
_DEFAULT_COLOUR = 'grey'

_LINESTYLE_BY_METHOD = {
    'TOPZ/SFM/R50': '-',
    'baseline':     '--',
    'UMAP':         ':',
}
_DEFAULT_LINESTYLE = '-'


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
      - the 'ra_strip' key (handled separately if needed)

    Only the VALUES are shown (no 'key=' prefix).
    """
    parts = [
        str(v)
        for k, v in selection.items()
        if v is not None and str(v) not in title_keys and k != 'ra_strip'
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

    from collections import defaultdict
    values_per_key = defaultdict(set)
    for r in panel_results:
        for k, v in r.get('selection', {}).items():
            if v is not None and k != 'ra_strip':
                values_per_key[k].add(str(v))

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

    def plot_correlation_figure(self, ncols=None, figsize=None):
        """
        Draw all panels in a single figure.
        """
        ncols = ncols or self.num_panels
        nrows = int(np.ceil(self.num_panels / ncols))
        figsize = figsize or (5 * ncols, 4 * nrows)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize,
            squeeze=False, sharex=True, sharey=True,
            constrained_layout=True
        )
        axes_flat = axes.flatten()

        for panel_idx in range(self.num_panels):
            ax = axes_flat[panel_idx]
            results_for_panel = self.selections_per_panel[panel_idx]
            if results_for_panel:
                self._plot_correlation_function_subplot(ax, results_for_panel)
            else:
                ax.set_visible(False)

        for ax in axes_flat[self.num_panels:]:
            ax.set_visible(False)

        for i, ax in enumerate(axes_flat[:self.num_panels]):
            if not ax.get_visible():
                continue
            row = i // ncols
            col = i % ncols
            if col == 0:
                ax.set_ylabel(r'$w(\theta)$')
            next_row_idx = i + ncols
            if next_row_idx >= self.num_panels:
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
        """
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
                lw=1.5, alpha=0.25, ls='', color=colour,
            )

        if self.log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlim(0.01, 10)
        else:
            ax.set_xlim(0.1, 3)

        ax.legend(fontsize=7)
        ax.grid()

        if panel_title:
            ax.set_title(panel_title, fontsize=8)