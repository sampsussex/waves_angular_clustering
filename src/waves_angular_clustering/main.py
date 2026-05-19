from angular_clustering import WavesWideClustering, AngularClusteringPlots


if __name__ == '__main__':
    ww = WavesWideClustering(
        n_photom_filepath  = '/mnt/lustre/projects/astro/general/sp624/waves-catas/d1m3p1f1/WAVES-N_d1m3p1f1.parquet',
        s_photom_filepath  = '/mnt/lustre/projects/astro/general/sp624/waves-catas/d1m3p1f1/WAVES-S_d1m3p1f1.parquet',
        n_stargal_filepath = '/research/astrodata/4most/WAVES/target_catalogues/star_gal_sep/WAVES-N_d1m3p1f1_Z22_stargal.parquet',
        s_stargal_filepath = '/research/astrodata/4most/WAVES/target_catalogues/star_gal_sep/WAVES-S_d1m3p1f1_Z22_stargal.parquet',
        n_randoms_filepath = '/mnt/lustre/projects/astro/general/sp624/waves_randoms/waves-wide_n_randoms.parquet',
        s_randoms_filepath = '/mnt/lustre/projects/astro/general/sp624/waves_randoms/waves-wide_s_randoms.parquet',
        results_directory  = '/mnt/lustre/projects/astro/general/sp624/angular_clustering_waves/results_20260519_mock/',
        use_mock=True,
        mock_data_filepath='/its/home/sp624/sharks_sim/fibre_incomplete_mocks.parquet',
    )

    all_results = ww.get_clustering_for_all_selections_to_run()