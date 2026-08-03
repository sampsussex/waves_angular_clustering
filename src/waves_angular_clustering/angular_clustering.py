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
                 results_directory=None, photom_type='total',
                 additional_masking=False, mask_rectangles_filepath=None):

        self.n_photom_filepath = n_photom_filepath
        self.s_photom_filepath = s_photom_filepath

        self.n_stargal_filepath = n_stargal_filepath
        self.s_stargal_filepath = s_stargal_filepath

        self.n_randoms_filepath = n_randoms_filepath
        self.s_randoms_filepath = s_randoms_filepath
        self.randoms_realisation_to_load = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # list of randoms realisations to load for each region

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
        # Parameters for the additional rectangle-based artefact masking
        # (only used when additional_masking=True). Rather than deriving
        # 'streak' candidate positions from the photometry itself, this now
        # reads a pre-computed JSON catalogue of flagged rectangular regions
        # (e.g. bad exposures / pointing footprints) and removes any source
        # or random point that falls inside one of those rectangles.
        #
        # The JSON file is expected to be a list of objects, each with at
        # least the keys 'ra_min', 'ra_max', 'dec_min', 'dec_max' (degrees),
        # e.g.:
        #   [{"dp_id": "...", "ra_min": 0.45, "ra_max": 0.65,
        #     "dec_min": -0.51, "dec_max": -0.32}, ...]
        # ------------------------------------------------------------------ #
        self.mask_rectangles_filepath = mask_rectangles_filepath
        # Fractional tolerance applied to each rectangle's width/height when
        # loaded, e.g. 0.1 grows each box by 10% of its own width and 10% of
        # its own height, split evenly on each side (so it stays centred on
        # the original box). Set to 0 for no padding.
        self.mask_rectangle_buffer_frac = 0.1
        # Dec value (degrees) used to split the mask rectangle catalogue into
        # 'north' / 'south' subsets so that, e.g., WWN only ever gets queried
        # against rectangles that could plausibly overlap it, rather than the
        # full (possibly much larger) combined catalogue.
        self.mask_hemisphere_dec_split = -15.0
        # Cached rectangle arrays — populated lazily on first use via
        # _load_mask_rectangles(), so the (potentially large) JSON file is
        # only read once regardless of how many selections are run.
        self._mask_rect_ra_min = None
        self._mask_rect_ra_max = None
        self._mask_rect_dec_min = None
        self._mask_rect_dec_max = None
        # Cache of boolean hemisphere-selection arrays, keyed by 'north' /
        # 'south', so the dec-split comparison is only computed once each.
        self._mask_rect_hemisphere_cache = {}

        self.data_ra_col = 'RAmax'
        self.data_dec_col = 'Decmax'
        self.randoms_ra_col = 'ra'
        self.randoms_dec_col = 'dec'

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
            'target_selection':   ['galaxy', 'galaxy/ambiguous', 'star', 'ambiguous'],
            'ghostmask_selection':['no ghostmask', 'with ghostmask'],
            'survey_depth':       ['Z<21.1', 'Z<21.25', 'Z<22',
                                   '16<Z<17', '17<Z<18', '18<Z<19', '19<Z<20', '20<Z<21', '21<Z<22',
                                   '16<Z<17.5', '17.5<Z<19', '19<Z<19.75', '19.75<Z<20.5',
                                   '20.5<Z<21.25', '21.25<Z<22', '17.5<Z<18.5', '18.5<Z<19.5', '19.5<Z<20.5'],
            'star_gal_method':    ['TOPZ/SFM/R50', 'baseline'],
            'region':             ['WWN', 'WWS', 'WW combined'],
        }

        #selections_to_run = {
        #    'target_selection':   ['galaxy', 'galaxy/ambiguous', 'star', 'ambiguous'],
        #    'ghostmask_selection':['with ghostmask'],
        #    'survey_depth':       ['Z<21.1'],
        #    'star_gal_method':    ['TOPZ/SFM/R50', 'baseline'],
        #    'region':             ['WWN', 'WWS'],
        #}
        # 17.5 18.5, 18.5, 19.5, 19.5 20.5 
        selections_to_run = {
            'target_selection':   ['galaxy'],
            'ghostmask_selection':['with ghostmask'],
            'survey_depth':       ['16<Z<17.5', '17.5<Z<18.5', '18.5<Z<19.5', '19.5<Z<20.5', '20.5<Z<21.25', '21.25<Z<22'],
            'star_gal_method':    ['TOPZ/SFM/R50'],
            'region':             ['WWN', 'WWS'],
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
            #[[157.25, 225], [-3.95, -3.5]], # the large slab at the bottom of Wwn
            [[201.8, 202], [-3.3, -3.1]],
            [[205.4, 205.5], [3.9, 3.95]],
            [[222, 222.2], [-2.6, -2.4]], 
            # Second round now
            [[193.83609, 194.07850], [3.07944, 3.72563]],
            [[206.39667, 206.4282], [-1.94328, -2.00215]],
            # second round south
            [[49.19606, 49.24780], [-35.56821, -35.52108]],
            [[39.33455, 39.35917], [-27.11029, -27.05638]],
            [[26.08654, 26.10927], [-34.87380, -34.71018]], 
            [[18.48791, 18.40237], [-31.78340, -31.71164]],
            [[27.27533, 27.36315], [-35.30838, -35.25087]], 
            [[40.14193, 40.17527], [-30.06847, -29.96956]], 
            [[18.144, 18.17667], [-33.68253, -33.64995]], 
            [[359.47490, 359.55225], [-32.64160, -32.55822]], 
            [[330.48250, 330.56051], [-32.01656, -31.93974]],
            [[351.59182, 351.64068], [-32.41250, -32.36937]]
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
        finite and positive. Used for the 'colour' depth selection.
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

    # ---------------------------------------------------------------------- #
    # Rectangle-based additional masking
    # ---------------------------------------------------------------------- #

    def _load_mask_rectangles(self):
        """
        Lazily load the JSON catalogue of rectangular mask regions from
        self.mask_rectangles_filepath, caching the ra_min/ra_max/dec_min/dec_max
        arrays on the instance so the file is only read once.
        """
        if self._mask_rect_ra_min is not None:
            return  # already loaded

        if self.mask_rectangles_filepath is None:
            raise ValueError(
                "additional_masking=True but mask_rectangles_filepath was not set."
            )

        print(f"  Loading mask rectangles from {self.mask_rectangles_filepath}...")
        with open(self.mask_rectangles_filepath, 'r') as f:
            rectangles = json.load(f)

        ra_min = np.array([r['ra_min'] for r in rectangles], dtype=float)
        ra_max = np.array([r['ra_max'] for r in rectangles], dtype=float)
        dec_min = np.array([r['dec_min'] for r in rectangles], dtype=float)
        dec_max = np.array([r['dec_max'] for r in rectangles], dtype=float)

        # Pad each rectangle by a fraction of its own width/height, split
        # evenly on each side, so the padded box grows by
        # `mask_rectangle_buffer_frac` in each dimension while staying
        # centred on the original box.
        buf = self.mask_rectangle_buffer_frac
        if buf:
            ra_pad = (ra_max - ra_min) * (buf / 2.0)
            dec_pad = (dec_max - dec_min) * (buf / 2.0)
            ra_min = ra_min - ra_pad
            ra_max = ra_max + ra_pad
            dec_min = dec_min - dec_pad
            dec_max = dec_max + dec_pad

        self._mask_rect_ra_min = ra_min
        self._mask_rect_ra_max = ra_max
        self._mask_rect_dec_min = dec_min
        self._mask_rect_dec_max = dec_max
        print(f"  Loaded {len(rectangles)} mask rectangles (buffer_frac={buf}).")

    def _get_rectangle_selection_for_hemisphere(self, hemisphere):
        """
        Return a boolean array (into the cached mask-rectangle arrays)
        selecting only the rectangles relevant to `hemisphere`. Cached per
        hemisphere so the comparison is only done once. `hemisphere=None`
        returns all rectangles (no filtering).
        """
        if hemisphere is None:
            return np.ones(len(self._mask_rect_ra_min), dtype=bool)

        if hemisphere in self._mask_rect_hemisphere_cache:
            return self._mask_rect_hemisphere_cache[hemisphere]

        split = self.mask_hemisphere_dec_split
        dec_min = self._mask_rect_dec_min
        dec_max = self._mask_rect_dec_max

        if hemisphere == 'north':
            # Keep rectangles that overlap the north region at all (i.e.
            # any part of the rectangle lies above the split).
            sel = dec_max > split
        elif hemisphere == 'south':
            sel = dec_min <= split
        else:
            raise ValueError(
                f"Unknown hemisphere: '{hemisphere}'. Expected 'north', 'south', or None."
            )

        self._mask_rect_hemisphere_cache[hemisphere] = sel
        print(
            f"  Filtered mask rectangles to hemisphere='{hemisphere}' "
            f"(dec split={split}): {sel.sum()} / {len(sel)} rectangles."
        )
        return sel

    def _apply_rectangle_mask(self, ra, dec, hemisphere=None):
        """
        Return a boolean keep-mask (True = keep) for the given RA/Dec points,
        flagging (and removing) any point that falls inside one of the
        rectangles loaded from self.mask_rectangles_filepath.

        `hemisphere` ('north', 'south', or None) restricts the rectangle
        catalogue to just the subset that could plausibly overlap this
        data (split on self.mask_hemisphere_dec_split), avoiding wasted
        query_ball_point calls against rectangles nowhere near this region.

        Rather than comparing every point against every rectangle (an
        N_points x N_rectangles brute force that becomes intractable once
        the rectangle catalogue has tens of thousands of entries — e.g.
        56M points x 24k rectangles is ~10^12 comparisons), this builds a
        single spatial index over the (large) point catalogue and then does
        one cheap radius query per rectangle (of which there are comparatively
        few). Each query returns only the points near that rectangle, which
        are then exactly tested against its true (rectangular, not circular)
        bounds.
        """
        self._load_mask_rectangles()

        ra = np.asarray(ra)
        dec = np.asarray(dec)

        rect_sel = self._get_rectangle_selection_for_hemisphere(hemisphere)
        ra_min = self._mask_rect_ra_min[rect_sel]
        ra_max = self._mask_rect_ra_max[rect_sel]
        dec_min = self._mask_rect_dec_min[rect_sel]
        dec_max = self._mask_rect_dec_max[rect_sel]

        n = len(ra)
        keep = np.ones(n, dtype=bool)

        if len(ra_min) == 0:
            return keep

        print(f"  Building spatial index over {n} points for rectangle masking ({hemisphere or 'all'})...")
        points = np.column_stack([ra, dec])
        tree = cKDTree(points)

        rect_centers = np.column_stack([
            (ra_min + ra_max) / 2.0,
            (dec_min + dec_max) / 2.0,
        ])
        # Enclosing-circle radius for each rectangle (half the diagonal) —
        # a cheap first-pass candidate filter ahead of the exact box test.
        rect_radii = 0.5 * np.sqrt((ra_max - ra_min) ** 2 + (dec_max - dec_min) ** 2)

        print(f"  Querying {len(ra_min)} rectangles against the point index...")
        for idx in range(len(ra_min)):
            candidate_idx = tree.query_ball_point(rect_centers[idx], r=rect_radii[idx])
            if not candidate_idx:
                continue
            candidate_idx = np.asarray(candidate_idx)
            inside = (
                (ra[candidate_idx] >= ra_min[idx]) & (ra[candidate_idx] <= ra_max[idx]) &
                (dec[candidate_idx] >= dec_min[idx]) & (dec[candidate_idx] <= dec_max[idx])
            )
            keep[candidate_idx[inside]] = False

        return keep

    def _load_dataset(self, photom_filepath, stargal_filepath, selection, hemisphere=None):
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

        # ensure artefacts are removed
        df = df[df['class'] != 'artefact']

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
        # Additional masking (rectangle-based artefact removal)
        # ------------------------------------------------------------------ #
        # Rectangles are static (loaded from JSON), so — unlike the old
        # streak-candidate approach — there is no need to derive anything
        # from this catalogue's photometry to also mask the randoms; the
        # randoms are masked independently with the same rectangle set in
        # _load_randoms.
        additional_keep_mask = None

        if self.additional_masking:
            print("  Applying rectangle-based additional mask...")
            additional_keep_mask = self._apply_rectangle_mask(
                df[self.data_ra_col].to_numpy(), df[self.data_dec_col].to_numpy(),
                hemisphere=hemisphere,
            )
            n_flagged = (~additional_keep_mask).sum()
            print(f"  Additional masking flags {n_flagged} / {len(df)} sources as inside a masked rectangle.")

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
            elif depth == '16<Z<17.5':
                base_selection &= (df['mag_Zt'] > 16) & (df['mag_Zt'] < 17.5)
            elif depth == '17.5<Z<19':
                base_selection &= (df['mag_Zt'] > 17.5) & (df['mag_Zt'] < 19)
            elif depth == '19<Z<19.75':
                base_selection &= (df['mag_Zt'] > 19) & (df['mag_Zt'] < 19.75)
            elif depth == '19.75<Z<20.5':
                base_selection &= (df['mag_Zt'] > 19.75) & (df['mag_Zt'] < 20.5)
            elif depth == '20.5<Z<21.25':
                base_selection &= (df['mag_Zt'] > 20.5) & (df['mag_Zt'] < 21.25)
            elif depth == '21.25<Z<22':
                base_selection &= (df['mag_Zt'] > 21.25) & (df['mag_Zt'] < 22)
            elif depth == '17.5<Z<18.5':
                base_selection &= (df['mag_Zt'] > 17.5) & (df['mag_Zt'] < 18.5)
            elif depth == '18.5<Z<19.5':
                base_selection &= (df['mag_Zt'] > 18.5) & (df['mag_Zt'] < 19.5)
            elif depth == '19.5<Z<20.5':
                base_selection &= (df['mag_Zt'] > 19.5) & (df['mag_Zt'] < 20.5)

        elif self.photom_type == 'colour':
            print("using colour photometry for selection")
            # Convert colour-aperture fluxes to magnitudes.
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
            elif depth == '16<Z<17.5':
                base_selection &= (df['mag_Zc'] > 16) & (df['mag_Zc'] < 17.5)
            elif depth == '17.5<Z<19':
                base_selection &= (df['mag_Zc'] > 17.5) & (df['mag_Zc'] < 19)
            elif depth == '19<Z<19.75':
                base_selection &= (df['mag_Zc'] > 19) & (df['mag_Zc'] < 19.75)
            elif depth == '19.75<Z<20.5':
                base_selection &= (df['mag_Zc'] > 19.75) & (df['mag_Zc'] < 20.5)
            elif depth == '20.5<Z<21.25':
                base_selection &= (df['mag_Zc'] > 20.5) & (df['mag_Zc'] < 21.25)
            elif depth == '21.25<Z<22':
                base_selection &= (df['mag_Zc'] > 21.25) & (df['mag_Zc'] < 22)
            elif depth == '17.5<Z<18.5':
                base_selection &= (df['mag_Zc'] > 17.5) & (df['mag_Zc'] < 18.5)
            elif depth == '18.5<Z<19.5':
                base_selection &= (df['mag_Zc'] > 18.5) & (df['mag_Zc'] < 19.5)
            elif depth == '19.5<Z<20.5':
                base_selection &= (df['mag_Zc'] > 19.5) & (df['mag_Zc'] < 20.5)

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

        return ra_data, dec_data

    def _load_randoms(self, randoms_filepath, selection, hemisphere=None):
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
            print("  Applying rectangle-based additional mask to randoms...")
            additional_keep_mask = self._apply_rectangle_mask(
                df[self.randoms_ra_col].to_numpy(),
                df[self.randoms_dec_col].to_numpy(),
                hemisphere=hemisphere,
            )
            n_flagged = (~additional_keep_mask).sum()
            print(f"  Additional masking flags {n_flagged} / {len(df)} random points as inside a masked rectangle.")
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
        ra_n, dec_n = self._load_dataset(
            self.n_photom_filepath, self.n_stargal_filepath, selection, hemisphere='north'
        )
        ra_s, dec_s = self._load_dataset(
            self.s_photom_filepath, self.s_stargal_filepath, selection, hemisphere='south'
        )
        return np.concatenate([ra_n, ra_s]), np.concatenate([dec_n, dec_s])

    def _load_WWC_randoms(self, selection):
        """Load and concatenate north + south randoms for the WW combined region."""
        ra_n, dec_n = self._load_randoms(self.n_randoms_filepath, selection, hemisphere='north')
        ra_s, dec_s = self._load_randoms(self.s_randoms_filepath, selection, hemisphere='south')
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
            hemisphere = 'north' if region == 'WWN' else 'south' if region == 'WWS' else None
            ra_data, dec_data = self._load_dataset(photom_fp, stargal_fp, selection, hemisphere=hemisphere)
            ra_rand, dec_rand = self._load_randoms(randoms_fp, selection, hemisphere=hemisphere)
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

    def plot_rectangle_mask(self, ra, dec, keep_mask, selection):
        """
        Save-only diagnostic: shows the RA/Dec footprint of points removed by
        the rectangle-based additional mask (red) versus points kept (grey),
        with the flagged rectangles themselves drawn as boxes. Call this with
        the pre-mask ra/dec arrays and the keep_mask returned from
        _apply_rectangle_mask.
        """
        removed = ~keep_mask
        if not np.any(removed):
            print("  No points removed by rectangle mask; skipping diagnostic plot.")
            return

        save_path = self._get_diagnostic_plot_path(selection, "rectangle_mask")

        ra_removed, dec_removed = ra[removed], dec[removed]
        ra_kept, dec_kept = ra[keep_mask], dec[keep_mask]

        self._load_mask_rectangles()

        # Only draw rectangles that overlap the plotted footprint, so the
        # figure stays legible even if the JSON contains many thousands of
        # entries covering a much wider area than this selection's data.
        pad = 0.5
        ra_lo, ra_hi = ra_removed.min() - pad, ra_removed.max() + pad
        dec_lo, dec_hi = dec_removed.min() - pad, dec_removed.max() + pad

        overlap = (
            (self._mask_rect_ra_max >= ra_lo) & (self._mask_rect_ra_min <= ra_hi) &
            (self._mask_rect_dec_max >= dec_lo) & (self._mask_rect_dec_min <= dec_hi)
        )

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.scatter(ra_kept, dec_kept, s=1, alpha=0.2, color='gray', label='kept', zorder=1)
        ax.scatter(ra_removed, dec_removed, s=2, alpha=0.4, color='red', label='removed', zorder=2)

        boxes = [
            Rectangle(
                (rmin, dmin), rmax - rmin, dmax - dmin
            )
            for rmin, rmax, dmin, dmax in zip(
                self._mask_rect_ra_min[overlap], self._mask_rect_ra_max[overlap],
                self._mask_rect_dec_min[overlap], self._mask_rect_dec_max[overlap],
            )
        ]
        if boxes:
            pc = PatchCollection(boxes, facecolor='none', edgecolor='red', alpha=0.6, zorder=3)
            ax.add_collection(pc)

        ax.set_xlim(ra_lo, ra_hi)
        ax.set_ylim(dec_lo, dec_hi)
        ax.set_aspect('equal')
        ax.set_xlabel('RA')
        ax.set_ylabel('Dec')
        ax.set_title('Rectangle-based additional masked footprint')
        ax.legend(markerscale=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved rectangle mask diagnostic to {save_path}")

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
        
import re
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import scipy.integrate
import scipy.special
from scipy.optimize import curve_fit

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

_STRIP_CMAP = cm.rainbow  # or cm.hsv


# ---------------------------------------------------------------------------
# Limber scaling test config / machinery
# ---------------------------------------------------------------------------

# TODO: replace with your actual BE (Baugh & Efstathiou 1993) N(z) fit
# parameters per survey_depth slice: (zc, alpha, beta, norm).
# `norm` cancels in the Limber amplitude ratio, so it need not be an accurate
# absolute normalisation -- keep it nonzero.
BE_PARAMS_BY_MAG_BIN = {
    '16<Z<17': (0.1325, 1.3553, 2.2421, 1.0),
    '17<Z<18': (0.1967, 1.3453, 2.3666, 1.0),
    '18<Z<19': (0.2852, 1.3387, 2.4943, 1.0),
    '19<Z<20': (0.4046,  1.3351, 2.6140, 1.0),
    '20<Z<21': (0.5637, 1.3327, 2.7176, 1.0),
    '21<Z<22': (0.7762,  1.3294,  2.8088, 1.0)
}

_BIN_EDGE_RE = re.compile(r'([\d.]+)\s*<\s*Z\s*<\s*([\d.]+)', re.IGNORECASE)


def _parse_bin_edges(bin_str):
    """'16<Z<17' -> (16.0, 17.0). Returns None if it doesn't match."""
    m = _BIN_EDGE_RE.search(str(bin_str))
    if not m:
        return None
    return float(m.group(1)), float(m.group(2))


def be_fit(z, zc, alpha, beta, norm):
    """Generalised Baugh & Efstathiou (1993, eqn 7) model for N(z)."""
    return norm * z ** alpha * np.exp(-(z / zc) ** beta)


class _DefaultFlatLCDM:
    """Minimal cosmology wrapper providing dc(z) and dxdz(z), used as the
    fallback when no `cosmo` object is supplied to AngularClusteringPlots.
    Requires astropy. Swap in your own object (with the same two methods)
    for anything more specific to your analysis."""

    def __init__(self, H0=100.0, Om0=0.3):
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        self._u = u
        self._cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
        self._c_km_s = 299792.458

    def dc(self, z):
        return self._cosmo.comoving_distance(z).to(self._u.Mpc).value

    def dxdz(self, z):
        Hz = self._cosmo.H(z).to(self._u.km / self._u.s / self._u.Mpc).value
        return self._c_km_s / Hz


def predict_limber_amplitude(cosmo, be_pars, gamma, eps=0.0,
                              zmin=1e-3, zmax=2.0):
    """Limber-predicted amplitude A for w(theta) = A * theta^(1-gamma)
    [theta in RADIANS], at r0 = 1, for a BE(z) redshift distribution."""
    def Nz(z):
        return be_fit(z, *be_pars)

    def denfun(z):
        return Nz(z)

    def xifun(z):
        x = cosmo.dc(z)
        return (x ** (1 - gamma) * Nz(z) ** 2 *
                (1 + z) ** (gamma - 3 - eps) / cosmo.dxdz(z))

    gfac = (math.pi ** 0.5 * scipy.special.gamma((gamma - 1) / 2) /
            scipy.special.gamma(gamma / 2))

    num = scipy.integrate.quad(xifun, zmin, zmax, epsabs=1e3, epsrel=1e-3)[0]
    den = scipy.integrate.quad(denfun, zmin, zmax, epsabs=1e3, epsrel=1e-3)[0] ** 2
    return gfac * num / den


def r0_from_fit(cosmo, be_pars, gamma, A_obs_rad, eps=0.0,
                 zmin=1e-3, zmax=2.0):
    """Infer r0 from a measured w(theta) = A_obs_rad * theta^(1-gamma)
    [theta in RADIANS] and a BE(z) fit."""
    A_unit = predict_limber_amplitude(cosmo, be_pars, gamma, eps=eps,
                                       zmin=zmin, zmax=zmax)
    if A_unit <= 0 or A_obs_rad <= 0:
        return np.nan
    return (A_obs_rad / A_unit) ** (1.0 / gamma)


def _amplitude_deg_to_rad(A_deg, gamma):
    """Convert amplitude A in w(theta)=A*theta^(1-gamma) from theta-in-degrees
    to theta-in-radians convention."""
    return A_deg * (180.0 / math.pi) ** (1 - gamma)


# ---------------------------------------------------------------------------

def _colour_for_strip(ra_strip) -> str:
    """Map strip 0-9 to a rainbow colour."""
    norm = mcolors.Normalize(vmin=0, vmax=9)
    return _STRIP_CMAP(norm(int(ra_strip)))


def _colour_for(selection: dict) -> str:
    target = selection.get('target_selection', '')
    return _COLOUR_BY_TARGET.get(target, _DEFAULT_COLOUR)


def _linestyle_for(selection: dict) -> str:
    method = selection.get('star_gal_method', '')
    return _LINESTYLE_BY_METHOD.get(method, _DEFAULT_LINESTYLE)


def _label_for(selection: dict, title_keys: set) -> str:
    parts = []
    for k, v in selection.items():
        if v is None or str(v) in title_keys:
            continue
        parts.append(f"strip {v}" if k == 'ra_strip' else str(v))
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


def _selection_sort_key(result: dict) -> tuple:
    """
    Deterministic sort key built purely from the *contents* of the selection
    dict (sorted alphabetically by key name, then stringified).  This makes
    the plotting order depend only on the selection values themselves, not on
    whatever order the results happened to arrive in / get matched in - which
    is what was causing the "WWN sometimes first, sometimes second" ordering
    problem.
    """
    sel = result.get('selection', {})
    return tuple(sorted((str(k), str(v)) for k, v in sel.items()))


def _line(x, m, c):
    return m * x + c


def _B_integral(cosmo, be_pars, gamma, eps=0.0, zmin=1e-3, zmax=2.0):
    """B_i(gamma): the Limber projection integral (pre-gfac, pre-r0) for a
    given BE(z) N(z) fit and power-law slope gamma. This is the 'B' in
    equation 16."""
    def Nz(z):
        return be_fit(z, *be_pars)

    def denfun(z):
        return Nz(z)

    def xifun(z):
        x = cosmo.dc(z)
        return (x ** (1 - gamma) * Nz(z) ** 2 *
                (1 + z) ** (gamma - 3 - eps) / cosmo.dxdz(z))

    num = scipy.integrate.quad(xifun, zmin, zmax, epsabs=1e3, epsrel=1e-3)[0]
    den = scipy.integrate.quad(denfun, zmin, zmax, epsabs=1e3, epsrel=1e-3)[0] ** 2
    return num / den


def compute_limber_shift_single_powerlaw(cosmo, be_pars_i, be_pars_ref, gamma,
                                          eps=0.0, zmin=1e-3, zmax=2.0):
    """Single power-law Limber shift: the vertical shift in log10(w) needed
    to move slice i's w(theta) onto the reference slice's depth, assuming
    one power law (index gamma, slice i's own fitted slope) dominates and
    only the amplitude differs due to N(z) dilution. No theta-axis shift.

    Returns dlog10_w such that shifted_w = w_i * 10**dlog10_w overlays the
    reference slice.
    """
    B_i   = _B_integral(cosmo, be_pars_i,   gamma, eps=eps, zmin=zmin, zmax=zmax)
    B_ref = _B_integral(cosmo, be_pars_ref, gamma, eps=eps, zmin=zmin, zmax=zmax)
    return math.log10(B_ref / B_i)


class AngularClusteringPlots:
    def __init__(self, clustering_results, num_panels, save_location=None,
                 log_scale=True, plot_fit=False, fit_max_theta=0.9,
                 limber_test=False, cosmo=None, limber_eps=0.0,
                 mag_bin_key='survey_depth', limber_zmin=1e-3, limber_zmax=2.0):
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
        plot_fit : bool
            If True, fit a straight line (power law) to log(xi) vs log(theta)
            for each curve, overplot the fit, and record the fitted slope /
            gamma.
        fit_max_theta : float
            Only points with theta < fit_max_theta [degrees] are used in the
            fit.
        limber_test : bool
            If True (and plot_fit is True), each fitted slice is compared
            against its Limber-predicted amplitude (using the BE(z) fit for
            that slice from BE_PARAMS_BY_MAG_BIN, keyed on the
            `mag_bin_key` selection value e.g. '16<Z<17') and an inferred r0
            is recorded per slice.
        cosmo : object or None
            Cosmology providing .dc(z) [comoving distance, Mpc] and
            .dxdz(z) [dx/dz, Mpc]. If None and limber_test=True, a default
            flat LCDM (H0=70, Om0=0.3, via astropy) is constructed
            automatically.
        limber_eps : float
            Clustering evolution parameter eps (0 = stable clustering in
            comoving coordinates).
        mag_bin_key : str
            Key in each result's `selection` dict used to look up the
            BE(z) params, e.g. selection['survey_depth'] == '16<Z<17'.
        limber_zmin, limber_zmax : float
            Integration limits in redshift for the Limber integral.
        """
        self.clustering_results = clustering_results
        self.save_location = save_location
        self.num_panels = num_panels
        self.log_scale = log_scale
        self.plot_fit = plot_fit
        self.fit_max_theta = fit_max_theta

        self.limber_test = limber_test
        self.limber_eps = limber_eps
        self.mag_bin_key = mag_bin_key
        self.limber_zmin = limber_zmin
        self.limber_zmax = limber_zmax

        self.cosmo = cosmo
        if self.limber_test and self.cosmo is None:
            print("limber_test=True with no cosmo supplied -- using default "
                  "flat LCDM (H0=70, Om0=0.3). Pass cosmo=... to override.")
            self.cosmo = _DefaultFlatLCDM()

        self.selections_per_panel = {panel: [] for panel in range(num_panels)}

        # Structure: {panel_index: [ {selection, m, m_err, c, gamma,
        #                              [r0, A_obs_rad, be_pars, mag_bin]} , ... ] }
        self.fit_results = {panel: [] for panel in range(num_panels)}

    def assign_results_to_panel(self, panel_index, selection_filters):
        """
        Assign clustering results matching *all* key/value pairs in
        selection_filters to a specific panel.

        Each value in selection_filters may be either:
          - a single value   e.g. 'no ghostmask'
          - a list of values e.g. ['galaxy', 'galaxy/ambiguous', 'star']

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

    # -----------------------------------------------------------------
    # Limber scaling helpers
    # -----------------------------------------------------------------

    def _run_limber_test(self, sel, gamma, c):
        """Compute Limber-predicted r0 for one fitted slice. Returns dict
        with r0, A_obs_rad, be_pars, mag_bin, or None if unavailable."""
        mag_bin = sel.get(self.mag_bin_key)
        if mag_bin is None:
            print(f"  Limber test skipped for {sel}: no '{self.mag_bin_key}' "
                  f"entry in selection.")
            return None

        be_pars = BE_PARAMS_BY_MAG_BIN.get(mag_bin)
        if be_pars is None:
            print(f"  Limber test skipped: no BE params for '{mag_bin}'. "
                  f"Add it to BE_PARAMS_BY_MAG_BIN.")
            return None

        A_obs_deg = math.exp(c)          # amplitude with theta in degrees
        A_obs_rad = _amplitude_deg_to_rad(A_obs_deg, gamma)

        r0 = r0_from_fit(
            self.cosmo, be_pars, gamma, A_obs_rad, eps=self.limber_eps,
            zmin=self.limber_zmin, zmax=self.limber_zmax,
        )
        return {'r0': r0, 'A_obs_rad': A_obs_rad, 'be_pars': be_pars,
                'mag_bin': mag_bin}

    def plot_limber_scaling(self, save_location=None, figsize=(6, 4)):
        """
        Plot inferred r0 vs. survey_depth-bin midpoint for every slice that
        had a successful Limber test, across all panels. Flat r0 => the
        amplitude differences between slices are explained by N(z) dilution
        alone; a trend => real clustering evolution (or eps is mis-set).
        """
        if not self.limber_test:
            raise RuntimeError(
                "plot_limber_scaling() requires limber_test=True and a "
                "prior call to plot_correlation_figure()."
            )

        fig, ax = plt.subplots(figsize=figsize)
        any_points = False
        for panel_idx, entries in self.fit_results.items():
            mids, r0s = [], []
            for e in entries:
                if e.get('r0') is None or not np.isfinite(e.get('r0', np.nan)):
                    continue
                edges = _parse_bin_edges(e['mag_bin'])
                if edges is None:
                    continue
                mlo, mhi = edges
                mids.append(0.5 * (mlo + mhi))
                r0s.append(e['r0'])
            if mids:
                any_points = True
                order = np.argsort(mids)
                mids_arr = np.array(mids)[order]
                r0_arr = np.array(r0s)[order]
                ax.plot(mids_arr, r0_arr, 'o-', label=f'panel {panel_idx}')

        if not any_points:
            print("No Limber results available to plot "
                  "(check mag_bin_key / BE_PARAMS_BY_MAG_BIN / cosmo).")
            plt.close(fig)
            return None

        ax.set_xlabel('mag / Z')
        ax.set_ylabel(r'inferred $r_0$ [Mpc]')
        ax.legend(fontsize=7)
        ax.grid()

        save_location = save_location or self.save_location
        if save_location:
            fig.savefig(save_location, dpi=150, bbox_inches='tight')
            print(f"Limber scaling figure saved to {save_location}")
        else:
            plt.show()
        return fig, ax

    def plot_limber_shift_test(self, reference_bin='19<Z<20', eps=None,
                                    ncols=None, figsize=None, save_location=None,
                                    residual_ylim=None):
            """
            For each panel, shift every survey_depth slice's w(theta) vertically
            (log10 w only) onto the reference_bin slice, using each slice's own
            fitted power-law slope gamma (from _fit_power_law) and the single
            power-law Limber amplitude ratio. Plots the aligned curves with a
            residual panel (with error bars) underneath.

            Parameters
            ----------
            reference_bin : str
                Key into BE_PARAMS_BY_MAG_BIN / survey_depth to use as reference
                (e.g. '19<Z<20').
            eps : float or None
                Clustering evolution parameter; defaults to self.limber_eps.
            residual_ylim : tuple(float, float) or None
                If given, sets the y-axis (delta log10 w) limits on every
                residual subplot to this (ymin, ymax). If None, matplotlib's
                default autoscaling is used.
            """
            if not self.limber_test:
                raise RuntimeError("plot_limber_shift_test() requires limber_test=True.")

            eps = self.limber_eps if eps is None else eps
            be_pars_ref = BE_PARAMS_BY_MAG_BIN.get(reference_bin)
            if be_pars_ref is None:
                raise ValueError(f"No BE params for reference_bin '{reference_bin}' "
                                f"in BE_PARAMS_BY_MAG_BIN.")

            panels_with_data = [p for p, res in self.selections_per_panel.items() if res]
            if not panels_with_data:
                print("No panel data to plot (call assign_results_to_panel first).")
                return None

            ncols = ncols or len(panels_with_data)
            nrows_pairs = int(np.ceil(len(panels_with_data) / ncols))
            figsize = figsize or (5 * ncols, 4 * 2 * nrows_pairs)

            fig, axes = plt.subplots(
                nrows_pairs * 2, ncols, figsize=figsize, squeeze=False,
                sharex='col', constrained_layout=True,
                gridspec_kw={'height_ratios': [3, 1] * nrows_pairs},
            )

            self.limber_shift_results = {p: [] for p in panels_with_data}

            for idx, panel_idx in enumerate(panels_with_data):
                row_pair = idx // ncols
                col = idx % ncols
                ax_main = axes[row_pair * 2, col]
                ax_res  = axes[row_pair * 2 + 1, col]

                results = self.selections_per_panel[panel_idx]

                ref_results = [r for r in results
                            if r.get('selection', {}).get(self.mag_bin_key) == reference_bin]
                if not ref_results:
                    print(f"Panel {panel_idx}: no result matching reference_bin "
                        f"'{reference_bin}', skipping.")
                    ax_main.set_visible(False)
                    ax_res.set_visible(False)
                    continue
                ref_result = ref_results[0]

                ref_meanlogr = np.array(ref_result['columns']['meanlogr'])
                ref_xi       = np.array(ref_result['columns']['xi'])
                ref_varxi    = np.array(ref_result['columns']['varxi'])
                ref_pos      = ref_xi > 0

                ref_theta = np.exp(ref_meanlogr[ref_pos])
                ref_log_theta = np.log10(ref_theta)
                ref_log_xi = np.log10(ref_xi[ref_pos])
                # Var[log10(xi)] ~= Var[xi] / (xi * ln10)^2
                ref_sigma_logxi = np.sqrt(ref_varxi[ref_pos]) / (ref_xi[ref_pos] * math.log(10))

                order = np.argsort(ref_log_theta)
                ref_log_theta = ref_log_theta[order]
                ref_log_xi = ref_log_xi[order]
                ref_sigma_logxi = ref_sigma_logxi[order]

                ax_main.plot(10**ref_log_theta, 10**ref_log_xi, 'k-',
                            lw=2, label=f'{reference_bin} (ref)', zorder=5)

                for result in sorted(results, key=_selection_sort_key):
                    sel = result.get('selection', {})
                    mag_bin = sel.get(self.mag_bin_key)
                    if mag_bin is None:
                        continue
                    be_pars_i = BE_PARAMS_BY_MAG_BIN.get(mag_bin)
                    if be_pars_i is None:
                        print(f"  Shift test skipped for '{mag_bin}': no BE params.")
                        continue

                    meanlogr = np.array(result['columns']['meanlogr'])
                    xi       = np.array(result['columns']['xi'])
                    varxi    = np.array(result['columns']['varxi'])

                    fit = self._fit_power_law(meanlogr, xi, varxi)
                    if fit is None:
                        print(f"  Shift test skipped for '{mag_bin}': power-law fit failed.")
                        continue
                    gamma_i = fit['gamma']

                    dlgw = compute_limber_shift_single_powerlaw(
                        self.cosmo, be_pars_i, be_pars_ref, gamma_i,
                        eps=eps, zmin=self.limber_zmin, zmax=self.limber_zmax,
                    )
                    self.limber_shift_results[panel_idx].append({
                        'selection': sel, 'mag_bin': mag_bin,
                        'gamma': gamma_i, 'dlog10_w': dlgw,
                    })

                    pos = xi > 0
                    if not np.any(pos):
                        continue

                    theta = np.exp(meanlogr[pos])
                    log_xi = np.log10(xi[pos])
                    # Var[log10(xi)] ~= Var[xi] / (xi * ln10)^2
                    sigma_logxi_i = np.sqrt(varxi[pos]) / (xi[pos] * math.log(10))

                    shifted_log_xi = log_xi + dlgw
                    shifted_xi = 10**shifted_log_xi
                    shifted_err = shifted_xi * math.log(10) * sigma_logxi_i  # linear-space error

                    label = f"{mag_bin}" + ('' if mag_bin == reference_bin
                                            else f", $\\gamma$={gamma_i:.2f}, $\\Delta\\log w$={dlgw:.2f}")
                    line, = ax_main.plot(theta, shifted_xi, 'o-', ms=3, label=label)
                    colour = line.get_color()
                    ax_main.errorbar(theta, shifted_xi, yerr=shifted_err,
                                    lw=1, alpha=0.3, ls='', color=colour)

                    if mag_bin == reference_bin:
                        continue

                    log_theta = np.log10(theta)
                    in_range = ((log_theta >= ref_log_theta.min()) &
                                (log_theta <= ref_log_theta.max()))
                    if not np.any(in_range):
                        continue

                    ref_interp = np.interp(log_theta[in_range], ref_log_theta, ref_log_xi)
                    ref_sigma_interp = np.interp(log_theta[in_range], ref_log_theta, ref_sigma_logxi)

                    residual = shifted_log_xi[in_range] - ref_interp
                    residual_err = np.sqrt(sigma_logxi_i[in_range]**2 + ref_sigma_interp**2)

                    ax_res.errorbar(theta[in_range], residual, yerr=residual_err,
                                    fmt='o', ms=3, color=colour, alpha=0.7, capsize=2)

                ax_res.axhline(0, color='k', lw=1, ls='--')
                ax_main.set_xscale('log')
                ax_main.set_yscale('log')
                ax_res.set_xscale('log')
                if residual_ylim is not None:
                    ax_res.set_ylim(*residual_ylim)
                ax_main.set_ylabel(r'$w(\theta)$ (shifted)')
                ax_res.set_ylabel(r'$\Delta\log_{10}w$')
                ax_res.set_xlabel(r'$\theta$ [degrees]')
                ax_main.legend(fontsize=6)
                ax_main.grid()
                ax_res.grid()

            save_location = save_location or self.save_location
            if save_location:
                fig.savefig(save_location, dpi=150, bbox_inches='tight')
                print(f"Limber shift-test figure saved to {save_location}")
            else:
                plt.show()

            return fig, axes, self.limber_shift_results

    def plot_correlation_figure(self, ncols=None, figsize=None):
        """
        Draw all panels in a single figure.

        Returns
        -------
        fig, axes, fit_results
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

        self.fit_results = {panel: [] for panel in range(self.num_panels)}

        for panel_idx in range(self.num_panels):
            ax = axes_flat[panel_idx]
            results_for_panel = self.selections_per_panel[panel_idx]
            if results_for_panel:
                self._plot_correlation_function_subplot(ax, results_for_panel, panel_idx)
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

        return fig, axes, self.fit_results

    def _fit_power_law(self, meanlogr, xi_raw, varxi_raw):
        meanlogr = np.asarray(meanlogr)
        xi_raw = np.asarray(xi_raw)
        varxi_raw = np.asarray(varxi_raw)

        sel_on_r = (
            (meanlogr < np.log(self.fit_max_theta))
            & (xi_raw > 0)
            & (varxi_raw > 0)
        )

        if np.count_nonzero(sel_on_r) < 2:
            return None

        r = meanlogr[sel_on_r]
        xi_raw_sel = xi_raw[sel_on_r]
        varxi_sel = varxi_raw[sel_on_r]

        log_xi = np.log(xi_raw_sel)
        var_logxi = varxi_sel / xi_raw_sel ** 2
        sigma_logxi = np.sqrt(var_logxi)

        try:
            popt, pcov = curve_fit(
                _line, r, log_xi,
                sigma=sigma_logxi,
                absolute_sigma=True,
            )
        except Exception as exc:
            print(f"  Fit failed: {exc}")
            return None

        m, c = popt
        m_err = np.sqrt(pcov[0, 0])
        gamma = -(m - 1)

        return {
            'm': m,
            'm_err': m_err,
            'c': c,
            'gamma': gamma,
            'log_r': r,
            'log_xi': log_xi,
            'sigma_logxi': sigma_logxi,
        }

    def _plot_correlation_function_subplot(self, ax, clustering_result_per_plot, panel_idx=None):
        panel_title, title_value_set = _build_panel_title(clustering_result_per_plot)

        clustering_result_per_plot = sorted(
            clustering_result_per_plot,
            key=_selection_sort_key,
        )

        for result in clustering_result_per_plot:
            columns  = result['columns']
            xi       = np.array(columns['xi'])
            varxi    = np.array(columns['varxi'])
            meanlogr = np.array(columns['meanlogr'])

            r   = np.exp(meanlogr)
            sel = result.get('selection', {})

            linestyle = _linestyle_for(sel)
            label     = _label_for(sel, title_value_set)

            if self.log_scale:
                pos_mask = xi > 0
                if not np.any(pos_mask):
                    print(f"  Warning: no positive xi values for selection {sel}. Skipping.")
                    continue
            else:
                pos_mask = np.ones_like(xi, dtype=bool)

            ra_strip = sel.get('ra_strip')
            colour = _colour_for_strip(ra_strip) if ra_strip is not None else None

            fit = None
            if self.plot_fit:
                fit = self._fit_power_law(meanlogr, xi, varxi)
                if fit is not None:
                    label = f"{label}, " r"$\gamma$" f" = {fit['gamma']:.3f}"

                    limber_info = None
                    if self.limber_test:
                        limber_info = self._run_limber_test(sel, fit['gamma'], fit['c'])
                        if limber_info is not None and np.isfinite(limber_info['r0']):
                            label = f"{label}, " r"$r_0$" f" = {limber_info['r0']:.2f}"

                    if panel_idx is not None:
                        entry = {
                            'selection': sel,
                            'm': fit['m'],
                            'm_err': fit['m_err'],
                            'c': fit['c'],
                            'gamma': fit['gamma'],
                        }
                        if limber_info is not None:
                            entry.update(limber_info)
                        self.fit_results[panel_idx].append(entry)

                        print(f"{sel}")
                        print(f"  slope = {fit['m']} +- {fit['m_err']}")
                        print(f"  gamma = {fit['gamma']}")
                        if limber_info is not None:
                            print(f"  Limber r0 = {limber_info['r0']}")

            line, = ax.plot(
                r[pos_mask], xi[pos_mask],
                label=label,
                linestyle=linestyle,
                color=colour,
            )
            colour = line.get_color()
            ax.errorbar(
                r[pos_mask], xi[pos_mask],
                yerr=np.sqrt(varxi[pos_mask]),
                lw=1.5, alpha=0.25, ls='', color=colour,
            )

            if fit is not None:
                log_r_min = fit['log_r'].min()
                log_r_max = fit['log_r'].max()
                fit_r  = np.exp([log_r_min, log_r_max])
                fit_xi = np.exp(_line(np.array([log_r_min, log_r_max]), fit['m'], fit['c']))
                ax.plot(
                    fit_r, fit_xi,
                    linestyle='-', linewidth=1.2, color=colour,
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