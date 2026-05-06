import numpy as np
import matplotlib.pyplot as plt
import treecorr
import pandas as pd

class AngularClustering:
    def __init__(self, ra_cat, dec_cat, ra_rand, dec_rand, selection_name,
                 min_sep = 0.01, max_sep = 10, nbins = 100, sep_units = 'degrees',
                 cat_units = 'degrees', rand_units = 'degrees',
                 n_patch= 10, var_method = 'jackknife'):
        self.ra_cat = ra_cat
        self.dec_cat = dec_cat
        self.ra_rand = ra_rand
        self.dec_rand = dec_rand
        self.selection_name = selection_name
        self.n_patch = n_patch
        self.var_method = var_method
        self.min_sep = min_sep
        self.max_sep = max_sep
        self.sep_units = sep_units
        self.cat_units = cat_units
        self.rand_units = rand_units
        self.nbins = nbins

        self._make_catalogs()

    def _make_catalogs(self):

        self.data_cat = treecorr.Catalog(ra=self.ra_cat, dec=self.dec_cat, ra_units=self.cat_units, dec_units=self.cat_units,
                                         npatch=self.n_patch, var_method=self.var_method)
        
        self.rand_cat = treecorr.Catalog(ra=self.ra_rand, dec=self.dec_rand, ra_units=self.rand_units, dec_units=self.rand_units,
                                         patch_centers=self.data_cat.patch_centers)

    def do_correlations(self):

        dd = treecorr.NNCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins, sep_units=self.sep_units)

        dr = treecorr.NNCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins, sep_units=self.sep_units)

        rr = treecorr.NNCorrelation(min_sep=self.min_sep, max_sep=self.max_sep, nbins=self.nbins, sep_units=self.sep_units)

        dd.process(self.data_cat)

        dr.process(self.data_cat, self.rand_cat)

        rr.process(self.rand_cat)

        self.xi, self.varxi = dd.calculateXi(rr=rr, dr=dr)

        self.meanlogr = dd.meanlogr
    

    def save_results(self, save_location):
        save_df = pd.DataFrame({'meanlogr': self.meanlogr, 'xi': self.xi, 'varxi': self.varxi})
        save_df.to_csv(save_location, index=False)
    

    def clean_up(self):
        del self.data_cat
        del self.rand_cat
        del self.dd
        del self.dr
        del self.rr

    # Need to work out how to save catalogs correlation res, and errors.


class WavesWideClustering:
    def __init__(self, n_photom_filepath, s_photom_filepath, 
                 n_stargal_filepath, s_stargal_filepath,
                 n_randoms_filepath, s_randoms_filepath,
                 n_photoz_filepath = None, s_photoz_filepath = None):

        self.n_photom_filepath = n_photom_filepath
        self.s_photom_filepath = s_photom_filepath

        self.n_stargal_filepath = n_stargal_filepath
        self.s_stargal_filepath = s_stargal_filepath

        self.n_photoz_filepath = n_photoz_filepath
        self.s_photoz_filepath = s_photoz_filepath

        self.n_randoms_filepath = n_randoms_filepath
        self.s_randoms_filepath = s_randoms_filepath

        self.data_ra_col = 'RAGaia'
        self.data_dec_col = 'DecGaia'
        self.randoms_ra_col = 'RA'
        self.randoms_dec_col = 'Dec'

        self.columns_to_load_photom = ['uberID', self.data_ra_col, self.data_dec_col, 'class', 'mag_Zt', 'mask', 'starmask', 'ghostmask', 'duplicate']
        self.columns_to_load_stargal = ['uberID', 'stargal']
        self.columns_to_load_randoms = [self.randoms_ra_col, self.randoms_dec_col, 'mask', 'starmask', 'ghostmask']

        target_selection = ['galaxy', 'galaxy/ambiguous', 'star'] # defo same plot
        ghostmask_selection = ['no ghostmask', 'with ghostmask'] # defo same plot
        survey_depth = ['Z<21.1', 'Z<21.25', 'Z<22'] # Defo not same plot
        star_gal_method = ['TOPZ+SFM', 'Baseline', 'UMAP']
        
        mag_slice_cuts = np.array([17, 18, 19, 20, 21, 22])

        possible_selections = {
            'target_selection': target_selection,
            'ghostmask_selection': ghostmask_selection,
            'survey_depth': survey_depth,
            'star_gal_method': star_gal_method,
        }

        selections_to_run = {
            'target_selection': ['galaxy', 'galaxy/ambiguous', 'star'],
            'ghostmask_selection': ['no ghostmask', 'with ghostmask'],
            'survey_depth': ['Z<21.25'],
            'star_gal_method': ['TOPZ+SFM'],
        }



    def _load_data(self, photom_filepath, stargal_filepath, selection):
        df = pd.read_parquet(photom_filepath, columns=self.columns_to_load_photom)
        df_stargal = pd.read_csv(stargal_filepath, usecols=self.columns_to_load_stargal)
        # ensure uberID has type int64 for both dfs
        df['uberID'] = df['uberID'].astype(np.int64)
        df_stargal['uberID'] = df_stargal['uberID'].astype(np.int64)
        
        df = df.merge(df_stargal, on='uberID', how='left')

        base_selection = (df['duplicate'] == False) & (df['mask'] == False) & (df['starmask'] == False)


        if selection['target_selection'] == 'galaxy':
            base_selection &= df['stargal'] == 'galaxy'
        elif selection['target_selection'] == 'galaxy/ambiguous':
            base_selection &= (df['stargal'] == 'galaxy') | (df['stargal'] == 'ambiguous')
        elif selection['target_selection'] == 'star':
            base_selection &= df['stargal'] == 'star'

        if selection['ghostmask_selection'] == 'with ghostmask':
            base_selection &= df['ghostmask'] == False

        if selection['survey_depth'] == 'Z<21.1':
            base_selection &= df['mag_Zt'] < 21.1
        elif selection['survey_depth'] == 'Z<21.25':
            base_selection &= df['mag_Zt'] < 21.25
        elif selection['survey_depth'] == 'Z<22':
            base_selection &= df['mag_Zt'] < 22

        ra_data = df.loc[base_selection, self.data_ra_col].values
        dec_data = df.loc[base_selection, self.data_dec_col].values
        return ra_data, dec_data
    

    def _load_randoms(self, randoms_filepath, selection):
        df = pd.read_parquet(randoms_filepath, columns=self.columns_to_load_randoms)

        base_selection = (df['mask'] == False) & (df['starmask'] == False)

        if selection['ghostmask_selection'] == 'with ghostmask':
            base_selection &= df['ghostmask'] == False

        ra_randoms = df.loc[base_selection, self.randoms_ra_col].values
        dec_randoms = df.loc[base_selection, self.randoms_dec_col].values
        return ra_randoms, dec_randoms



class AngularClusteringPlots:
    def __init__(self, clustering_results, save_location = None):
        self.clustering_results = clustering_results
        self.save_location = save_location


    def plot_correlation_figure(self):
        pass


    def _plot_correlation_function_subplot(self, ax, clustering_result_per_plot):
        for result in clustering_result_per_plot:

            r1 = np.exp(result['meanlogr'])
            r1_pos_mask = result['xi'] > 0
            r1 = r1[r1_pos_mask]
            xi = result['xi'][r1_pos_mask]
            varxi = result['varxi'][r1_pos_mask]

            ax.plot(r1, xi, label="LABEL PLACEHOLDER", color = colour)
            ax.errorbar(r1, xi, yerr=np.sqrt(varxi), lw=0.1, ls = '', color = colour)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\theta$ (degrees)')
        ax.set_ylabel(r'$w(\theta)$')
        ax.legend()
        ax.set_xlim(0.01, 10)
        ax.grid()