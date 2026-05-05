import numpy as np
import matplotlib.pyplot as plt
import treecorr

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

        #self.xi, self.rr = dd.calculateXi(rr=rr, dr=dr)

    def clean_up(self):
        del self.data_cat
        del self.rand_cat
        del self.dd
        del self.dr
        del self.rr

    # Need to work out how to save catalogs and correlation results.


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

        columns_to_load = ['uberID', 'RAGaia', 'DecGaia', 'class', 'mag_Zt']


        ghostmask_selection = ['No ghostmask', 'with ghostmask']
        target_selection = ['galaxy-galaxy', 'star-star']
        survey_depth = ['Z < 21.1', 'Z < 21.25', 'Z < 22']
        photo_z_selection = ['Wide Photo-z', 'Deep Photo-z']
        star_gal_method = ['TOPZ+SFM', 'Baseline', 'UMAP']


        selections = [
            'galaxy-galaxy WavesWide', 
            'star-star WavesWide',
            'galaxy-galaxy WavesWide with ghostmask',
            'star-star WavesWide with ghostmask'
            ]
        
        mag_slice_cuts = np.array([17, 18, 19, 20, 21, 22])

        selection_mag_slices = []
        for mag_slice in range(len(mag_slice_cuts)-1):
            selection_mag_slices.append(f'galaxy-galaxy WavesWide with ghostmask and {mag_slice_cuts[mag_slice]} < mag_Zt < {mag_slice_cuts[mag_slice+1]}')

    def _load_data(self):
        return None


class AngularClusteringPlots:
    def __init__(self, clustering_results):
        self.clustering_results = clustering_results


    def plot_correlation_figure(self):
        pass


    def _plot_correlation_function(self, ax, clustering_result):

        pass