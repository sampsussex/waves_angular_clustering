from angular_clustering import WavesWideClustering, AngularClusteringPlots

ww = WavesWideClustering(
    n_photom_filepath  = '/data/waves/WWN_photom.parquet',
    s_photom_filepath  = '/data/waves/WWS_photom.parquet',
    n_stargal_filepath = '/data/waves/WWN_stargal.csv',
    s_stargal_filepath = '/data/waves/WWS_stargal.csv',
    n_randoms_filepath = '/data/waves/WWN_randoms.parquet',
    s_randoms_filepath = '/data/waves/WWS_randoms.parquet',
    results_directory  = '/results/clustering',
)

# Runs all 6 combinations (3 target_selections × 2 ghostmask_selections).
# On the first run, RR is computed once per unique (region, ghostmask) pair
# and saved to /results/clustering/rr_cache/.
# On subsequent runs, the cached RR files are reused automatically.
all_results = ww.get_clustering_for_all_selections_to_run()