"""
whitenoise.exoplanet — SWNA for TESS transit light curves.

Preprocessing pipeline that turns full-sector TESS PDC-SAP light curves
into gap-filled transit windows ready for ``wn.analyze()``, plus one-shot
and batch analysis drivers.

See ``whitenoise/exoplanet/EXOPLANET_GUIDE.html`` and
``whitenoise/exoplanet/exoplanet_demo.ipynb`` for a full walkthrough.
"""

from .preprocess import (
    estimate_T14,
    window_has_gap,
    fill_gaps_nn,
    find_empirical_midpoint,
    extract_transit_window,
)
from .io import (
    clean_lc,
    download_transit_lc,
    write_pipeline_csv,
    read_pipeline_csv,
    list_transit_csvs,
)
from .pipeline import (
    analyze_transit,
    batch_analyze_transits,
)

__all__ = [
    'estimate_T14',
    'window_has_gap',
    'fill_gaps_nn',
    'find_empirical_midpoint',
    'extract_transit_window',
    'clean_lc',
    'download_transit_lc',
    'write_pipeline_csv',
    'read_pipeline_csv',
    'list_transit_csvs',
    'analyze_transit',
    'batch_analyze_transits',
]
