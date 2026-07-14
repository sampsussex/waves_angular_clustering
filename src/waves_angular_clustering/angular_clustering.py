import os
import json
import itertools
import numpy as np
import pandas as pd
import treecorr
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection



class AngularClustering:
    def __init__(self, ra_cat, dec_cat, ra_rand, dec_rand, selection_dic,
                 min_sep=0.01, max_sep=10, nbins=30, sep_units='degrees',
                 cat_units='degrees', rand_units='degrees',
                 n_patch=20, var_method='jackknife', ):
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

                # In AngularClustering.do_correlations(), after rr.process(self.rand_cat):
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
                 results_directory=None, photom_type = 'colour',
                 additional_masking = False):

        self.n_photom_filepath = n_photom_filepath
        self.s_photom_filepath = s_photom_filepath

        self.n_stargal_filepath = n_stargal_filepath
        self.s_stargal_filepath = s_stargal_filepath

        self.n_randoms_filepath = n_randoms_filepath
        self.s_randoms_filepath = s_randoms_filepath
        self.randoms_realisation_to_load = [0, 1, 2, 3, 4]

        self.results_directory = results_directory

        if photom_type not in ['total', 'colour']:
            raise ValueError(f"Invalid photom_type: '{photom_type}'. Must be 'total' or 'colour'.")
        
        self.photom_type = photom_type


        # Treecorr binning settings — shared by all AngularClustering instances
        # and used when reconstructing an RR object from cache.
        self.min_sep   = 0.01
        self.max_sep   = 10
        self.nbins     = 30
        self.sep_units = 'degrees'

        self.additional_masking = additional_masking

        # ------------------------------------------------------------------ #
        # Parameters for the additional 'streak' artefact masking (only used
        # when additional_masking=True). Streaks arise where total-aperture
        # photometry is artificially dilated relative to colour-aperture
        # photometry, producing spatially clustered spurious detections.
        # Sources (and randoms) that fall inside a box around a flagged
        # 'streak candidate' position are removed.
        # ------------------------------------------------------------------ #
        self.streak_half_width_ra  = 0.1   # deg, half box width in RA
        self.streak_half_width_dec = 0.1   # deg, half box width in Dec
        self.streak_threshold      = 10    # neighbour count above which a region is flagged
        self.streak_dmagZ_max      = -3    # mag_Zt - mag_Zc must be below this
        self.streak_magZt_min      = 19.0
        self.streak_magZt_max      = 21.25

        self.data_ra_col = 'RAmax'
        self.data_dec_col = 'Decmax'
        self.randoms_ra_col = 'ra'
        self.randoms_dec_col = 'dec'
        if additional_masking:
            self.columns_to_load_photom = [
                'uberID', self.data_ra_col, self.data_dec_col,
                'class', 'mag_Zt', 'flux_ic', 'flux_Yc', 'flux_rc', 'flux_Zc',
                'mask', 'starmask', 'ghostmask', 'duplicate'
            ]
        else:
            if self.photom_type == 'total':
                self.columns_to_load_photom = [
                    'uberID', self.data_ra_col, self.data_dec_col,
                    'class', 'mag_Zt', 'mask', 'starmask', 'ghostmask',
                    'duplicate'
                ]
            elif self.photom_type == 'colour':
                self.columns_to_load_photom = [
                    'uberID', self.data_ra_col, self.data_dec_col,
                    'class', 'flux_ic', 'flux_Yc', 'flux_rc', 'flux_Zc',
                    'mask', 'starmask', 'ghostmask', 'duplicate'
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
            'survey_depth':       ['Z<21.1', 'Z<21.25', 'Z<22',
                                   '16<Z<17', '17<Z<18', '18<Z<19', '19<Z<20', '20<Z<21', '21<Z<22'],
            'star_gal_method':    ['TOPZ/SFM/R50', 'baseline'],
            'region':             ['WWN', 'WWS', 'WW combined'],
        }

        selections_to_run = {
            'target_selection':   ['galaxy'],
            'ghostmask_selection':['no ghostmask', 'with ghostmask'],
            'survey_depth':       ['Z<21.1', '16<Z<17', '17<Z<18', '18<Z<19', '19<Z<20', '20<Z<21', '21<Z<22'],
            'star_gal_method':    ['TOPZ/SFM/R50'],
            'region':             ['WWN'],#, 'WWS'],
        }

        self._validate_selections(selections_to_run)

        # Expand the dict-of-lists into a flat list of individual selection dicts,
        # one per combination (Cartesian product).
        self.selections_to_run = self._expand_selections(selections_to_run)

        self.extra_rec_masks = [
            [[165.9, 165.95], [-3.95, -3.7]], # in north, ramin, ramax, decmin, decmax
            [[215.4, 215.5], [3.7, 3.95]], # in north, ramin, ramax, decmin, decmax
            [[17.85, 17.95], [-30.15, -30.05]], # in south, ramin, ramax, decmin, decmax
            [[18.4, 18.5], [-31.80, -31.70]], # in south, ramin, ramax, decmin, decmax
            #[[157.25, 225], [-3.95, -3.5]], not using the large slab at the bottom for now
            [[201.8, 202], [-3.3, -3.1]], 
            [[205.4, 205.5], [3.9, 3.95]], 
            [[222, 222.2], [-2.6, -2.4]] 
        ]
        # Ive put in these extra masks as there are some iffy regions that may need additional masking.
        # for certain the 1st, and 3rd region here are needed. Need to check on the 
        # seg viewer that the others are justified. Perhaps also
        # the snugness of the masks might be causing some isses as well.
        # the ghostmasks may also be a bit too smug. I guess i need to go back to the
        # other stacked plots to check on this more thoroughly.

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

    def _add_colour_magnitudes(self, df):
        """
        Convert colour-aperture fluxes to magnitudes (mag_ic, mag_Yc, mag_rc,
        mag_Zc), estimating mag_Zc from neighbouring bands where the Z colour
        flux itself is missing. Only computed where the underlying flux is
        finite and positive. Used both for the 'colour' depth selection and
        for the additional (streak) masking, which needs mag_Zc regardless
        of photom_type.
        """
        print("  Converting colour fluxes to magnitudes...")
        df['mag_ic'] = np.nan
        df['mag_Yc'] = np.nan
        df['mag_rc'] = np.nan
        df['mag_Zc'] = np.nan
        print("checking is finite and positive for flux_ic, flux_Yc, flux_rc, flux_Zc")
        valid_i = np.isfinite(df['flux_ic']) & (df['flux_ic'] > 0)
        valid_Y = np.isfinite(df['flux_Yc']) & (df['flux_Yc'] > 0)
        valid_r = np.isfinite(df['flux_rc']) & (df['flux_rc'] > 0)
        valid_Z = np.isfinite(df['flux_Zc']) & (df['flux_Zc'] > 0)
        print("  Converting fluxes to magnitudes where valid...")
        df.loc[valid_i, 'mag_ic'] = 8.9 - 2.5 * np.log10(df.loc[valid_i, 'flux_ic'])
        df.loc[valid_Y, 'mag_Yc'] = 8.9 - 2.5 * np.log10(df.loc[valid_Y, 'flux_Yc'])
        df.loc[valid_r, 'mag_rc'] = 8.9 - 2.5 * np.log10(df.loc[valid_r, 'flux_rc'])
        df.loc[valid_Z, 'mag_Zc'] = 8.9 - 2.5 * np.log10(df.loc[valid_Z, 'flux_Zc'])
        print("  Estimating missing Z magnitudes where possible...")
        # If Z colour flux is missing, estimate Z from i and Y.
        use_iY = (~valid_Z) & valid_i & valid_Y
        df.loc[use_iY, 'mag_Zc'] = (
            df.loc[use_iY, 'mag_Yc']
            - 0.4912 * (df.loc[use_iY, 'mag_ic'] - df.loc[use_iY, 'mag_Yc'])
            - 0.0281
        )
        print("  Estimating missing Z magnitudes from r and i where possible...")
        # If both Z and Y colour fluxes are missing, estimate Z from r and i.
        use_ri = (~valid_Z) & (~valid_Y) & valid_r & valid_i
        df.loc[use_ri, 'mag_Zc'] = (
            df.loc[use_ri, 'mag_ic']
            - 0.7044 * (df.loc[use_ri, 'mag_rc'] - df.loc[use_ri, 'mag_ic'])
            + 0.004
        )
        return df

    def _get_additional_mask(self, df):
        """
        Identify 'streak' artefacts using the total-vs-colour Z magnitude
        difference (mag_Zt - mag_Zc), which flags sources whose photometric
        apertures have been artificially dilated (e.g. by satellite/asteroid
        streaks or similar image defects). A source is flagged as a streak
        artefact if it lies within a small RA/Dec box of more than
        `self.streak_threshold` other 'streak candidate' sources.

        Returns
        -------
        keep_mask : boolean array, len(df) — True to keep, False to mask out
        ra_streak_cand, dec_streak_cand : RA/Dec of the d_magZ-selected
            'streak candidate' sources (used to also mask the randoms, and
            for diagnostics)
        is_streak_point : boolean array over the candidates, flagging which
            candidates themselves sit in dense ('streak') regions (diagnostics only)
        """
        if 'mag_Zc' not in df.columns:
            df = self._add_colour_magnitudes(df)

        # Only search for streak candidates among sources that are already
        # 'clean' — i.e. not masked, not star-masked, not duplicates, and
        # not flagged as artefacts by the pipeline's own 'class' column.
        # This keeps the d_magZ streak search from being contaminated by
        # sources that would be cut for other reasons anyway.
        clean_selection = (
            (df['mask'] == False) &
            (df['starmask'] == False) &
            (df['duplicate'] == False) &
            (df['class'] != 'artefact')
        )

        d_magZ = df['mag_Zt'] - df['mag_Zc']
        sel = (
            clean_selection &
            (d_magZ < self.streak_dmagZ_max) &
            (df['mag_Zt'] < self.streak_magZt_max) &
            (df['mag_Zt'] > self.streak_magZt_min)
        )

        ra_sel = df.loc[sel, self.data_ra_col].to_numpy()
        dec_sel = df.loc[sel, self.data_dec_col].to_numpy()

        if len(ra_sel) == 0:
            # No streak candidates found — nothing to mask.
            return np.ones(len(df), dtype=bool), ra_sel, dec_sel, np.zeros(0, dtype=bool)

        # Scale coordinates so a box of +/- half_width becomes a unit
        # Chebyshev ball, matching the diagnostic snippet's approach.
        scaled_sel = np.column_stack([
            ra_sel / self.streak_half_width_ra,
            dec_sel / self.streak_half_width_dec,
        ])
        tree = cKDTree(scaled_sel)

        counts_per_point = tree.query_ball_point(scaled_sel, r=1.0, p=np.inf, return_length=True)
        is_streak_point = counts_per_point > self.streak_threshold

        ra_all = df[self.data_ra_col].to_numpy()
        dec_all = df[self.data_dec_col].to_numpy()
        scaled_all = np.column_stack([
            ra_all / self.streak_half_width_ra,
            dec_all / self.streak_half_width_dec,
        ])
        counts_all = tree.query_ball_point(scaled_all, r=1.0, p=np.inf, return_length=True)
        streak_mask = counts_all > self.streak_threshold

        keep_mask = ~streak_mask

        return keep_mask, ra_sel, dec_sel, is_streak_point

    def _apply_additional_mask_to_points(self, ra, dec, ra_streak_cand, dec_streak_cand):
        """
        Apply the same streak exclusion (built from the data catalogue's
        d_magZ candidate positions) to an arbitrary set of RA/Dec points.
        Used to mask the randoms catalogue in the same way as the data,
        since the randoms have no photometry of their own to derive
        mag_Zt/mag_Zc from.
        """
        ra = np.asarray(ra)
        dec = np.asarray(dec)

        if ra_streak_cand is None or len(ra_streak_cand) == 0:
            return np.ones(len(ra), dtype=bool)

        scaled_cand = np.column_stack([
            np.asarray(ra_streak_cand) / self.streak_half_width_ra,
            np.asarray(dec_streak_cand) / self.streak_half_width_dec,
        ])
        tree = cKDTree(scaled_cand)

        scaled_points = np.column_stack([
            ra / self.streak_half_width_ra,
            dec / self.streak_half_width_dec,
        ])
        counts = tree.query_ball_point(scaled_points, r=1.0, p=np.inf, return_length=True)

        return counts <= self.streak_threshold

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
        # Additional masking (streak-artefact removal)
        # ------------------------------------------------------------------ #
        # Computed here (rather than inside the depth-selection branch below)
        # because it needs mag_Zt/mag_Zc regardless of self.photom_type, and
        # because the resulting candidate positions are also needed to mask
        # the randoms catalogue for this selection.
        ra_streak_cand = np.array([])
        dec_streak_cand = np.array([])
        additional_keep_mask = None

        if self.additional_masking:
            print("  Computing additional (streak) mask...")
            df = self._add_colour_magnitudes(df)
            additional_keep_mask, ra_streak_cand, dec_streak_cand, is_streak_point = self._get_additional_mask(df)
            n_flagged = (~additional_keep_mask).sum()
            print(f"  Additional masking flags {n_flagged} / {len(df)} sources as streak artefacts.")
            self.plot_streak_mask(ra_streak_cand, dec_streak_cand, is_streak_point, selection)

        # ------------------------------------------------------------------ #
        # Build selection mask
        # ------------------------------------------------------------------ #
        base_selection = (
            (df['duplicate'] == False) &
            (df['mask'] == False) &
            (df['starmask'] == False)
        )
        extra_rec_masks = self._get_extra_rec_masks(df[self.data_ra_col].to_numpy(), df[self.data_dec_col].to_numpy())

        base_selection &= extra_rec_masks

        if self.additional_masking:
            base_selection &= additional_keep_mask

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
        if self.photom_type == 'total':
            if depth == 'Z<21.1':
                base_selection &= df['mag_Zt'] < 21.1
            elif depth == 'Z<21.25':
                base_selection &= df['mag_Zt'] < 21.25
            elif depth == 'Z<22':
                base_selection &= df['mag_Zt'] < 22
            elif depth == '16<Z<17':
                base_selection &= (df['mag_Zt'] > 16) & (df['mag_Zt'] < 17)
            elif depth == '17<Z<18':
                base_selection &= (df['mag_Zt'] > 17) & (df['mag_Zt'] < 18)
            elif depth == '18<Z<19':
                base_selection &= (df['mag_Zt'] > 18) & (df['mag_Zt'] < 19)
            elif depth == '19<Z<20':
                base_selection &= (df['mag_Zt'] > 19) & (df['mag_Zt'] < 20)
            elif depth == '20<Z<21':
                base_selection &= (df['mag_Zt'] > 20) & (df['mag_Zt'] < 21)
            elif depth == '21<Z<22':
                base_selection &= (df['mag_Zt'] > 21) & (df['mag_Zt'] < 22)

        elif self.photom_type == 'colour':
            print("using colour photometry for selection")
            # Convert colour-aperture fluxes to magnitudes (skip if already
            # computed above for additional masking).
            if 'mag_Zc' not in df.columns:
                df = self._add_colour_magnitudes(df)

            print("  Applying depth selection...")
            if depth == 'Z<21.1':
                base_selection &= df['mag_Zc'] < 21.1
            elif depth == 'Z<21.25':
                base_selection &= df['mag_Zc'] < 21.25
            elif depth == 'Z<22':
                base_selection &= df['mag_Zc'] < 22
            elif depth == '16<Z<17':
                base_selection &= (df['mag_Zc'] > 16) & (df['mag_Zc'] < 17)
            elif depth == '17<Z<18':
                base_selection &= (df['mag_Zc'] > 17) & (df['mag_Zc'] < 18)
            elif depth == '18<Z<19':
                base_selection &= (df['mag_Zc'] > 18) & (df['mag_Zc'] < 19)
            elif depth == '19<Z<20':
                base_selection &= (df['mag_Zc'] > 19) & (df['mag_Zc'] < 20)
            elif depth == '20<Z<21':
                base_selection &= (df['mag_Zc'] > 20) & (df['mag_Zc'] < 21)
            elif depth == '21<Z<22':
                base_selection &= (df['mag_Zc'] > 21) & (df['mag_Zc'] < 22)

            print(f"  Applied colour-based selection with photom_type='{self.photom_type}'.")
        print(f"  Number of objects after selection: {base_selection.sum()}")
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

        return ra_data, dec_data, ra_streak_cand, dec_streak_cand

    def _load_randoms(self, randoms_filepath, selection, ra_streak_cand=None, dec_streak_cand=None):
        print(f"  Loading randoms from {randoms_filepath} with selection {selection}...")
        df = pd.read_parquet(randoms_filepath, columns=self.columns_to_load_randoms)

        base_selection = (
            (df['starmask'] == False) &
            (df['polygon_mask'] == False) &
            (df['realisation'].isin(self.randoms_realisation_to_load))
        )

        extra_rec_masks = self._get_extra_rec_masks(df[self.randoms_ra_col].to_numpy(), df[self.randoms_dec_col].to_numpy())
        base_selection &= extra_rec_masks

        if selection['ghostmask_selection'] == 'with ghostmask':
            base_selection &= df['ghostmask'] == False

        if self.additional_masking:
            print("  Applying additional (streak) mask to randoms...")
            additional_keep_mask = self._apply_additional_mask_to_points(
                df[self.randoms_ra_col].to_numpy(),
                df[self.randoms_dec_col].to_numpy(),
                ra_streak_cand,
                dec_streak_cand,
            )
            n_flagged = (~additional_keep_mask).sum()
            print(f"  Additional masking flags {n_flagged} / {len(df)} random points as streak artefacts.")
            base_selection &= additional_keep_mask

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
        ra_n, dec_n, ra_streak_n, dec_streak_n = self._load_dataset(
            self.n_photom_filepath, self.n_stargal_filepath, selection
        )
        ra_s, dec_s, ra_streak_s, dec_streak_s = self._load_dataset(
            self.s_photom_filepath, self.s_stargal_filepath, selection
        )
        ra_streak_cand = np.concatenate([ra_streak_n, ra_streak_s])
        dec_streak_cand = np.concatenate([dec_streak_n, dec_streak_s])
        return (
            np.concatenate([ra_n, ra_s]),
            np.concatenate([dec_n, dec_s]),
            ra_streak_cand,
            dec_streak_cand,
        )

    def _load_WWC_randoms(self, selection, ra_streak_cand=None, dec_streak_cand=None):
        """Load and concatenate north + south randoms for the WW combined region."""
        ra_n, dec_n = self._load_randoms(self.n_randoms_filepath, selection, ra_streak_cand, dec_streak_cand)
        ra_s, dec_s = self._load_randoms(self.s_randoms_filepath, selection, ra_streak_cand, dec_streak_cand)
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
            ra_data, dec_data, ra_streak_cand, dec_streak_cand = self._load_WWC_data(selection)
            ra_rand, dec_rand = self._load_WWC_randoms(selection, ra_streak_cand, dec_streak_cand)
            print(f"  Loaded {len(ra_data)} data points and {len(ra_rand)} randoms for WW combined.")
        else:
            print("  Loading data and randoms...")
            photom_fp, stargal_fp, randoms_fp = self._get_filepaths_for_selection(selection)
            ra_data, dec_data, ra_streak_cand, dec_streak_cand = self._load_dataset(photom_fp, stargal_fp, selection)
            ra_rand, dec_rand = self._load_randoms(randoms_fp, selection, ra_streak_cand, dec_streak_cand)
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
        ac.do_correlations()
        print("  Clustering computation complete.")
        print("  Saving DD/DR/RR diagnostic plot and raw values...")   # new
        self.plot_dd_dr_rr(ac.dd, ac.dr, ac.rr, selection)            # new
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


    def _wrap_ra_for_region(self, ra, selection):
        """
        For WWS, wrap RA values > 180 deg into negative RA values.

        Example:
            359 -> -1
            270 -> -90
            181 -> -179
        """
        ra = np.asarray(ra).copy()

        if selection.get("region") == "WWS":
            ra[ra > 180] -= 360

        return ra


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

        ra_data_plot = self._wrap_ra_for_region(ra_data, selection)
        ra_rand_plot = self._wrap_ra_for_region(ra_rand, selection)

        fig, axes = plt.subplots(1, 2, figsize=(5, 5))

        density = normalise

        # --------------------------------------------------
        # RA histogram
        # --------------------------------------------------
        axes[0].hist(
            ra_data_plot,
            bins=bins,
            histtype='step',
            linewidth=2,
            density=density,
            label='Data'
        )

        axes[0].hist(
            ra_rand_plot,
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
        bins=500,
    ):
        """
        Save-only 2D density diagnostic.
        """
        save_path = self._get_diagnostic_plot_path(
            selection,
            "density2d"
        )
        ra_data_plot = self._wrap_ra_for_region(ra_data, selection)
        ra_rand_plot = self._wrap_ra_for_region(ra_rand, selection)

        # --------------------------------------------------
        # Derive true data extents from combined data + randoms
        # --------------------------------------------------
        ra_all  = np.concatenate([ra_data_plot, ra_rand_plot])
        dec_all = np.concatenate([dec_data,     dec_rand])

        ra_min,  ra_max  = ra_all.min(),  ra_all.max()
        dec_min, dec_max = dec_all.min(), dec_all.max()

        ra_range  = ra_max  - ra_min   # full RA  span in degrees
        dec_range = dec_max - dec_min  # full Dec span in degrees

        # Equal angular bin size: choose one bin width in degrees that applies
        # to both axes, then derive the number of bins on each axis.
        bin_deg   = ra_range / bins          # bin size driven by the longer axis
        n_ra_bins = bins                     # exactly `bins` cells along RA
        n_dec_bins = max(1, int(np.round(dec_range / bin_deg)))  # matched cell size

        x_bins = np.linspace(ra_min,  ra_max,  n_ra_bins  + 1)
        y_bins = np.linspace(dec_min, dec_max, n_dec_bins + 1)

        # --------------------------------------------------
        # Figure sizing: one pixel of figure space per bin cell,
        # scaled up so the shorter axis stays legible.
        # --------------------------------------------------
        aspect_ratio = n_ra_bins / n_dec_bins   # e.g. ~69 if RA≈8×Dec

        min_panel_height = 4.0          # inches — floor so Dec detail is visible
        panel_height = max(min_panel_height, 18.0 / aspect_ratio)
        panel_width  = panel_height * aspect_ratio

        fig, axes = plt.subplots(3, 1, figsize=(panel_width, panel_height * 3))

        # --------------------------------------------------
        # Data density
        # --------------------------------------------------
        h1 = axes[0].hist2d(
            ra_data_plot,
            dec_data,
            bins=[x_bins, y_bins],
            cmap='coolwarm',
        )
        axes[0].set_aspect('equal')
        axes[0].set_title('Data')
        axes[0].set_xlabel('RA [deg]')
        axes[0].set_ylabel('Dec [deg]')
        fig.colorbar(h1[3], ax=axes[0])

        # --------------------------------------------------
        # Random density
        # --------------------------------------------------
        h2 = axes[1].hist2d(
            ra_rand_plot,
            dec_rand,
            bins=[x_bins, y_bins],
            cmap='coolwarm',
        )
        axes[1].set_aspect('equal')
        axes[1].set_title('Randoms')
        axes[1].set_xlabel('RA [deg]')
        axes[1].set_ylabel('Dec [deg]')
        fig.colorbar(h2[3], ax=axes[1])

        # --------------------------------------------------
        # Difference map
        # --------------------------------------------------
        data_hist, xedges, yedges = np.histogram2d(
            ra_data_plot,
            dec_data,
            bins=[x_bins, y_bins]
        )
        rand_hist, _, _ = np.histogram2d(
            ra_rand_plot,
            dec_rand,
            bins=[xedges, yedges]
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
            extent=[
                xedges[0], xedges[-1],
                yedges[0], yedges[-1]
            ],
        )
        axes[2].set_title('Data - Randoms')
        axes[2].set_xlabel('RA [deg]')
        axes[2].set_ylabel('Dec [deg]')
        fig.colorbar(im, ax=axes[2])

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved density diagnostic to {save_path}")


    def plot_streak_mask(self, ra_streak_cand, dec_streak_cand, is_streak_point, selection):
        """
        Save-only diagnostic: shows the RA/Dec footprint of the d_magZ-selected
        'streak candidate' sources, with points flagged as being in a dense
        ('streak') region drawn as red exclusion boxes and kept points shown
        as grey scatter — matching the plot_mask_boxes() diagnostic.
        """
        if len(ra_streak_cand) == 0:
            print("  No streak-candidate sources found; skipping streak mask diagnostic plot.")
            return

        save_path = self._get_diagnostic_plot_path(selection, "streak_mask")

        ra_flagged = ra_streak_cand[is_streak_point]
        dec_flagged = dec_streak_cand[is_streak_point]
        ra_kept = ra_streak_cand[~is_streak_point]
        dec_kept = dec_streak_cand[~is_streak_point]

        fig, ax = plt.subplots(figsize=(30, 4))

        ax.scatter(ra_kept, dec_kept, s=1, alpha=0.2, color='gray', label='kept', zorder=1)

        if len(ra_flagged) > 0:
            boxes = [
                Rectangle(
                    (ra - self.streak_half_width_ra, dec - self.streak_half_width_dec),
                    2 * self.streak_half_width_ra, 2 * self.streak_half_width_dec
                )
                for ra, dec in zip(ra_flagged, dec_flagged)
            ]
            pc = PatchCollection(boxes, facecolor='red', edgecolor='none', alpha=0.3, zorder=2)
            ax.add_collection(pc)

            ax.set_xlim(ra_flagged.min() - 5 * self.streak_half_width_ra, ra_flagged.max() + 5 * self.streak_half_width_ra)
            ax.set_ylim(dec_flagged.min() - 5 * self.streak_half_width_dec, dec_flagged.max() + 5 * self.streak_half_width_dec)

        ax.set_aspect('equal')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.set_title('Additional (streak) masked footprint')
        ax.legend(markerscale=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved streak mask diagnostic to {save_path}")


    def plot_dd_dr_rr(self, dd, dr, rr, selection):
        """
        Plot and save DD, DR, RR pair counts (weight and npairs) as a function
        of mean separation, and save the raw values to a JSON file.
        Both are written to the diagnostics directory.
        """
        diag_dir = self._get_diagnostics_directory()
        base = self._selection_to_filename(selection).replace(".json", "")

        # ------------------------------------------------------------------ #
        # Save raw values
        # ------------------------------------------------------------------ #
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

        # ------------------------------------------------------------------ #
        # Plot
        # ------------------------------------------------------------------ #
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            "Pair counts: DD / DR / RR\n" +
            " | ".join(f"{k}={v}" for k, v in selection.items()),
            fontsize=9
        )

        colours = {'DD': 'steelblue', 'DR': 'darkorange', 'RR': 'seagreen'}

        for label, corr in [('DD', dd), ('DR', dr), ('RR', rr)]:
            sep = corr.meanr          # degrees
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