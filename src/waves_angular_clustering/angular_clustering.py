import numpy as np
import matplotlib.pyplot as plt
import treecorr
import pandas as pd
import json

class AngularClustering:
    def __init__(self, ra_cat, dec_cat, ra_rand, dec_rand, selection_dic,
                 min_sep = 0.01, max_sep = 10, nbins = 100, sep_units = 'degrees',
                 cat_units = 'degrees', rand_units = 'degrees',
                 n_patch= 10, var_method = 'jackknife'):
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

        self.results = {    # dictionary to hold results
                        'selection': selection_dic,
                        'columns':{
                        'xi': None,
                        'varxi': None,
                        'meanlogr': None
                        }
                        }

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

        self.results['columns']['xi'] = self.xi
        self.results['columns']['varxi'] = self.varxi
        self.results['columns']['meanlogr'] = self.meanlogr


    def save_results(self, save_location):
        with open(save_location, 'w') as f:
            json.dump(self.results, f)
        

    def clean_up(self):
        del self.data_cat
        del self.rand_cat
        del self.dd
        del self.dr
        del self.rr


class WavesWideClustering:
    def __init__(self, n_photom_filepath = None, s_photom_filepath = None,
                 n_stargal_filepath= None, s_stargal_filepath= None,
                 n_randoms_filepath=None, s_randoms_filepath= None,
                 results_directory = None):

        self.n_photom_filepath = n_photom_filepath
        self.s_photom_filepath = s_photom_filepath

        self.n_stargal_filepath = n_stargal_filepath
        self.s_stargal_filepath = s_stargal_filepath

        self.n_randoms_filepath = n_randoms_filepath
        self.s_randoms_filepath = s_randoms_filepath
        self.randoms_realisation_to_load = [0, 1, 2, 3, 4]

        self.results_directory = results_directory

        self.data_ra_col = 'RAGAIA'
        self.data_dec_col = 'DecGAIA'
        self.randoms_ra_col = 'ra'
        self.randoms_dec_col = 'dec'

        self.columns_to_load_photom = ['uberID', self.data_ra_col, self.data_dec_col, 'class', 'mag_Zt', 'mask', 'starmask', 'ghostmask', 'duplicate']
        self.columns_to_load_stargal = ['uberID', 'stargal']
        self.columns_to_load_randoms = [self.randoms_ra_col, self.randoms_dec_col, 'in_region', 'starmask', 'ghostmask', 'polygon_mask', 'realisation']

        target_selection = ['galaxy', 'galaxy/ambiguous', 'star'] 
        ghostmask_selection = ['no ghostmask', 'with ghostmask']
        survey_depth = ['Z<21.1', 'Z<21.25', 'Z<22']
        star_gal_method = ['TOPZ+SFM', 'baseline'] 
        region = ['WWN', 'WWS', 'WW combined'] # north, south, and deep (deep is subset of south)

        possible_selections = {
            'target_selection': target_selection,
            'ghostmask_selection': ghostmask_selection,
            'survey_depth': survey_depth,
            'star_gal_method': star_gal_method,
            'region': region
        }

        selections_to_run = {
            'target_selection': ['galaxy', 'galaxy/ambiguous', 'star'],
            'ghostmask_selection': ['no ghostmask', 'with ghostmask'],
            'survey_depth': ['Z<21.1'],
            'star_gal_method': ['TOPZ/SFM/R50'],
            'region': ['WWN']
        }

        # check selection_to_run is valid.

        # recompose selections to run into a list of selection dictionaries, where each dictionary is a single selection to run in angular clustering.


    def _load_dataset(self, photom_filepath, stargal_filepath, selection):
        # check this section if the indexes get messed up and need to be reset.

        df = pd.read_parquet(photom_filepath, columns=self.columns_to_load_photom)
        df['uberID'] = df['uberID'].astype(np.int64)

        if selection['star_gal_method'] == 'TOPZ/SFM/R50':
            df_stargal = pd.read_csv(stargal_filepath, usecols=self.columns_to_load_stargal)
            df_stargal['uberID'] = df_stargal['uberID'].astype(np.int64)
            
            df = df.merge(df_stargal, on='uberID', how='left')
            if selection['target_selection'] == 'galaxy':
                mask = df['stargal'] == 'galaxy'
                df.loc[mask, 'stargal'] = 'galaxy'
            elif selection['target_selection'] == 'galaxy/ambiguous':
                mask = (df['stargal'] == 'galaxy') | (df['stargal'] == 'ambiguous')
                df.loc[mask, 'stargal'] = 'galaxy/ambiguous'
            elif selection['target_selection'] == 'star':
                mask = df['stargal'] == 'star'
                df.loc[mask, 'stargal'] = 'star'
        
        if selection['star_gal_method'] == 'baseline':
            if selection['target_selection'] == 'galaxy':
                mask = df['class'] == 'galaxy'
                df.loc[mask, 'stargal'] = 'galaxy'
            elif selection['target_selection'] == 'galaxy/ambiguous':
                mask = (df['class'] == 'galaxy') | (df['class'] == 'ambiguous')
                df.loc[mask, 'stargal'] = 'galaxy/ambiguous'
            elif selection['target_selection'] == 'star':
                mask = df['class'] == 'star'
                df.loc[mask, 'stargal'] = 'star'
        
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

        # inset check for if dataset is empty, and if dataset contains nans. 
        return ra_data, dec_data
    

    def _load_randoms(self, randoms_filepath, selection):
        df = pd.read_parquet(randoms_filepath, columns=self.columns_to_load_randoms)

        base_selection = (df['in_region'] == True) & (df['starmask'] == False) & (df['polygon_mask'] == False) & (df['realisation'].isin(self.randoms_realisation_to_load))


        if selection['ghostmask_selection'] == 'with ghostmask':
            base_selection &= df['ghostmask'] == False

        ra_randoms = df.loc[base_selection, self.randoms_ra_col].values
        dec_randoms = df.loc[base_selection, self.randoms_dec_col].values
        return ra_randoms, dec_randoms
    

    def _load_WWC_data(self, selection):
        # need to write wrapper when loading combined datasets
        pass

    def _load_WWC_randoms(self, selection):
        # needs to write wrapper when loading combined datasets
        pass


    def load_results(self, results_filepath):
        with open(results_filepath, 'r') as f:
            results = json.load(f)
        return results
    
    
    def _check_if_results_exist(self, selection):
        # check if results exist for selection. Need to finish.
        pass

    def get_previously_run_results(self, selection):
        # get previously run results for selection. Need to finish. 
        # remove selections that have been run already from selections to run.
        pass

    def get_clustering_for_selection(self, selection):
        # run on data for selection. Need to finish. 
        # ALWAYS SAVE!
        pass
    
    def get_clustering_for_all_selections_to_run(self):
        # run on all selections to run. Need to finish.
        # ALWAYS SAVE!
        pass


class AngularClusteringPlots:
    def __init__(self, clustering_results, num_panels, save_location = None):
        self.clustering_results = clustering_results # it may be easier to define the dataset some other way
        self.save_location = save_location
        self.num_panels = num_panels
        panel_dic = {'selection': None, 'colour': None, 'columns': {'xi': None, 'varxi': None, 'meanlogr': None}} # this is ugly and doenst really work
        # I basically want a common sense way of chosing which single selections i want in each panel

        self.selections_per_panel = {panel: panel_dic for panel in range(num_panels)}


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