# whitenoise.radiojove — SWNA pipeline for RadioJOVE radio burst observations
#
# This submodule provides RadioJOVE-specific I/O, preprocessing, and analysis
# functions that bridge raw RadioJOVE CSV data to the core whitenoise package.
#
# Typical workflow:
#   import whitenoise as wn
#
#   # Read a single burst CSV
#   time, intensity, meta = wn.radiojove.read_radiojove_csv('burst.csv')
#
#   # Analyze and plot
#   out = wn.radiojove.analyze_burst('burst.csv', model='exponential')
#   out['figure'].show()
#
#   # Batch-analyze a folder of CSVs
#   df = wn.radiojove.batch_analyze_bursts('Solar/Type 3 Bursts/first trials/')
#   print(df[['dataset', 'mu']])
#
#   # Group files by cadence and compare
#   paths = wn.radiojove.list_burst_csvs('Solar/Type 3 Bursts/first trials/')
#   groups = wn.radiojove.group_by_cadence(paths)
#   cr = wn.radiojove.compare_bursts(groups[0.1], model='exponential')

from .io import read_radiojove_csv, parse_filename_metadata, list_burst_csvs
from .preprocess import zscore_normalize, resample, group_by_cadence
from .pipeline import analyze_burst, batch_analyze_bursts, plot_burst_msd, compare_bursts

__all__ = [
    # I/O
    'read_radiojove_csv',
    'parse_filename_metadata',
    'list_burst_csvs',
    # Preprocessing
    'zscore_normalize',
    'resample',
    'group_by_cadence',
    # Pipeline
    'analyze_burst',
    'batch_analyze_bursts',
    'plot_burst_msd',
    'compare_bursts',
]
