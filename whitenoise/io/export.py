"""
io/export.py — Export analysis results to CSV files.

export_csv(result, path)               Save lags + empirical/fitted MSD for an AnalysisResult.
export_summary(cr, path)               Save ComparisonResult.summary_df to CSV.
export_series(values, path, ...)       Save a 1-D observable/preprocessed array to CSV.
export_msd(lags, msd, path, ...)       Save MSD vs lags arrays to CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def export_csv(result, path: str) -> None:
    """
    Save lag, empirical MSD, and fitted MSD from an :class:`AnalysisResult`.

    The output is a 3-column CSV::

        lag,msd_empirical,msd_fitted
        1.0,0.12345,0.11980
        2.0,0.24100,0.23750
        ...

    The ``msd_fitted`` column contains the theoretical curve evaluated at each
    lag.  Values outside the fitting window (set by *max_lag_fraction*) and rows
    where the model returns non-finite values are written as ``NaN``.

    Parameters
    ----------
    result : AnalysisResult
        Output of :func:`~whitenoise.analysis.pipeline.analyze`.
    path : str
        Destination file path (e.g. ``'results/msd.csv'``).
    """
    if result.fit is not None:
        # Trim both empirical and fitted to the fitting window so every row
        # has a value in both columns — no NaN padding needed.
        n_use      = len(result.fit.lags_used)
        lags       = np.asarray(result.lags[:n_use], dtype=float)
        msd_emp    = np.asarray(result.msd_empirical[:n_use], dtype=float)
        msd_fitted = np.asarray(result.fit.msd_fitted, dtype=float)
    else:
        # No fit available — export empirical only, fitted column is all NaN.
        lags       = np.asarray(result.lags, dtype=float)
        msd_emp    = np.asarray(result.msd_empirical, dtype=float)
        msd_fitted = np.full(len(lags), np.nan, dtype=float)

    df = pd.DataFrame({
        'lag':           lags,
        'msd_empirical': msd_emp,
        'msd_fitted':    msd_fitted,
    })
    df.to_csv(path, index=False)
    print(f'\U0001f4be Saved to {path}')


def export_series(
    values,
    path: str,
    label: str = 'value',
    unit: str = '',
    time=None,
    time_label: str = 'index',
    time_unit: str = 'step',
) -> None:
    """
    Save a 1-D observable or preprocessed data array to a whitenoise-format CSV.

    The output follows the standard whitenoise 2-column header format::

        index [step], value []
        0,12.345
        1,13.102
        ...

    This CSV can be reloaded directly with :func:`~whitenoise.io.reader.read_csv`
    to re-enter the pipeline at any point without recomputing.

    Use this to checkpoint data **before** calling :func:`~whitenoise.core.msd.compute_msd`,
    for example after extracting nucleotide transition distances or after detrending.

    .. note::
        For very large arrays (e.g. tens of millions of genomic distances) the
        resulting CSV will be large. Consider saving only a representative subset
        or using :func:`export_msd` to checkpoint after MSD computation instead.

    Parameters
    ----------
    values : array-like (1-D)
        The observable values to save.
    path : str
        Destination file path (e.g. ``'distances_A_A.csv'``).
    label : str, default ``'value'``
        Column name for the values. Used as the header name.
    unit : str, default ``''``
        Unit string for the values column. Empty string → unitless.
    time : array-like (1-D), optional
        Index or time axis. Defaults to ``[0, 1, 2, …, N-1]`` if not provided.
    time_label : str, default ``'index'``
        Column name for the time/index column.
    time_unit : str, default ``'step'``
        Unit string for the time/index column.

    Examples
    --------
    >>> distances = wn.genomics.get_transition_distances(seq, 'A', 'A')
    >>> wn.export_series(distances, 'distances_A_A.csv', label='distance', unit='bp')

    >>> # Reload later and skip straight to compute_msd:
    >>> _, distances, _ = wn.read_csv('distances_A_A.csv')
    >>> lags, msd = wn.compute_msd(distances)
    """
    values = np.asarray(values, dtype=float).ravel()
    if time is None:
        time = np.arange(len(values), dtype=float)
    else:
        time = np.asarray(time, dtype=float).ravel()

    t_header = f'{time_label} [{time_unit}]'
    v_header = f'{label} [{unit}]'

    df = pd.DataFrame({t_header: time, v_header: values})
    df.to_csv(path, index=False)
    print(f'\U0001f4be Saved {len(values):,} rows to {path}')


def export_msd(
    lags,
    msd: 'np.ndarray',
    path: str,
    label: str = 'msd',
) -> None:
    """
    Save MSD vs lags arrays to a whitenoise-format CSV.

    The output is a 2-column CSV::

        lag [step], msd []
        1,0.12345
        2,0.24100
        ...

    This CSV can be reloaded with :func:`~whitenoise.io.reader.read_csv` and
    passed directly to :func:`~whitenoise.core.fitting.fit_msd` to try different
    models without rerunning :func:`~whitenoise.core.msd.compute_msd`.

    Use this to checkpoint **after** :func:`~whitenoise.core.msd.compute_msd`
    and **before** :func:`~whitenoise.core.fitting.fit_msd`.

    Parameters
    ----------
    lags : array-like (1-D)
        Lag values from :func:`~whitenoise.core.msd.compute_msd`.
    msd : array-like (1-D)
        Empirical MSD values from :func:`~whitenoise.core.msd.compute_msd`.
    path : str
        Destination file path (e.g. ``'msd_A_A.csv'``).
    label : str, default ``'msd'``
        Column name for the MSD values.

    Examples
    --------
    >>> lags, msd = wn.compute_msd(distances, max_lag=1500)
    >>> wn.export_msd(lags, msd, 'msd_A_A.csv')

    >>> # Reload later and fit a different model:
    >>> lags, msd, _ = wn.read_csv('msd_A_A.csv')
    >>> fit = wn.fit_msd(lags, msd, model='exponential')
    """
    lags = np.asarray(lags, dtype=float).ravel()
    msd  = np.asarray(msd,  dtype=float).ravel()

    df = pd.DataFrame({
        'lag [step]': lags,
        f'{label} []': msd,
    })
    df.to_csv(path, index=False)
    print(f'\U0001f4be Saved {len(lags):,} MSD points to {path}')


def export_summary(cr, path: str) -> None:
    """
    Save the comparison summary table from a :class:`ComparisonResult` to CSV.

    Writes ``cr.summary_df`` using :meth:`pandas.DataFrame.to_csv` (no index).

    Parameters
    ----------
    cr : ComparisonResult
        Output of :func:`~whitenoise.analysis.compare.compare` or
        :func:`~whitenoise.analysis.batch.batch_model_search`.
    path : str
        Destination file path.
    """
    cr.summary_df.to_csv(path, index=False)
    print(f'\U0001f4be Saved to {path}')
