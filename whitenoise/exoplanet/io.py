"""
whitenoise.exoplanet.io
========================
I/O utilities for TESS transit light curves.

Two boundaries are handled here:

1. **lightkurve → numpy** — :func:`clean_lc` turns a ``lightkurve``
   LightCurve object into plain ``(time, flux)`` arrays (NaN-removal,
   sigma-clipping, median-normalization). :func:`download_transit_lc`
   wraps a full TESS SPOC search + download for one target.

2. **numpy → whitenoise CSV** — :func:`write_pipeline_csv` writes the
   2-column ``time_min_from_mid,flux`` format that ``wn.analyze()``
   consumes; :func:`read_pipeline_csv` reads it back;
   :func:`list_transit_csvs` lists a folder of them.

``lightkurve`` is an optional dependency — only imported inside the
functions that need it (:func:`download_transit_lc`, :func:`clean_lc`).
Install with ``pip install whitenoise-swna[exoplanet]``.

Typical workflow::

    import whitenoise as wn

    # Download + clean a TESS light curve
    time, flux, meta = wn.exoplanet.download_transit_lc('WASP-078', cadence=120)

    # ... extract windows with wn.exoplanet.extract_transit_window ...

    wn.exoplanet.write_pipeline_csv(
        window['time'], window['flux'], t_mid,
        'pipeline/WASP-078/WASP-078_T01.csv',
    )

    # Later, batch-read a folder of pipeline CSVs
    paths = wn.exoplanet.list_transit_csvs('pipeline/WASP-078/')
"""

from __future__ import annotations

import glob
import os

import numpy as np


# ── clean_lc ──────────────────────────────────────────────────────────────────

def clean_lc(lc, sigma: float = 3.0) -> tuple:
    """
    Convert a lightkurve LightCurve object into clean, normalized numpy arrays.

    Removes NaNs and sigma-clipped outliers, then normalizes flux by its
    median. This is the standard boundary between a ``lightkurve`` object
    and the plain ``(time, flux)`` arrays used by the rest of
    ``whitenoise.exoplanet`` (and by ``wn.analyze()``).

    Parameters
    ----------
    lc : lightkurve.LightCurve
        A light curve object, e.g. one entry of a
        ``lightkurve.LightCurveCollection`` from
        ``lightkurve.search_lightcurve(...).download_all()``.
    sigma : float, default 3.0
        Outlier-clipping threshold passed to ``lc.remove_outliers()``.

    Returns
    -------
    time : np.ndarray
        Time values (BTJD days), NaNs and outliers removed.
    flux : np.ndarray
        Flux values, median-normalized to 1.0.

    Example
    -------
    >>> import lightkurve as lk
    >>> sr = lk.search_lightcurve('WASP-078', mission='TESS', cadence=120, author='SPOC')
    >>> lc_collection = sr.download_all(quality_bitmask='default')
    >>> time, flux = wn.exoplanet.clean_lc(lc_collection[0])
    """
    lc = lc.remove_nans().remove_outliers(sigma=sigma)
    time = np.asarray(lc.time.value, dtype=float)
    flux = np.asarray(lc.flux.value, dtype=float)
    flux = flux / np.nanmedian(flux)
    return time, flux


# ── download_transit_lc ───────────────────────────────────────────────────────

def download_transit_lc(
    target: str,
    cadence: int = 120,
    author: str = 'SPOC',
    quality_bitmask: str = 'default',
    sigma: float = 3.0,
    pick: str = 'most_transits',
    period_days: 'float | None' = None,
) -> tuple:
    """
    Download all available TESS sectors for *target* and return the best one.

    Wraps ``lightkurve.search_lightcurve(...).download_all()``, cleans each
    sector with :func:`clean_lc`, and selects a single "best" sector.

    Requires ``lightkurve`` — install with ``pip install whitenoise-swna[exoplanet]``.

    Parameters
    ----------
    target : str
        Target name resolvable by the MAST portal (e.g. ``'WASP-078'``).
    cadence : int, default 120
        Cadence in seconds. ``120`` selects TESS's standard 2-minute
        cadence SPOC light curves.
    author : str, default ``'SPOC'``
        Pipeline author passed to ``lightkurve.search_lightcurve()``.
    quality_bitmask : str, default ``'default'``
        Quality-flag mask passed to ``download_all()``.
    sigma : float, default 3.0
        Outlier-clipping threshold passed to :func:`clean_lc`.
    pick : str, default ``'most_transits'``
        Sector-selection strategy. ``'most_transits'`` picks the sector
        with the most estimated transits, i.e. the longest baseline
        divided by *period_days* (falls back to longest baseline if
        *period_days* is not given).
    period_days : float, optional
        Orbital period (days), used to score sectors when
        ``pick='most_transits'``.

    Returns
    -------
    time : np.ndarray
        Time values (BTJD days) for the selected sector.
    flux : np.ndarray
        Normalized flux values for the selected sector.
    metadata : dict
        ``'sector'`` (int), ``'n_transits'`` (float, if *period_days* given),
        ``'all_sectors'`` — list of ``(sector, n_points)`` for every
        downloaded sector, for diagnostics.

    Raises
    ------
    RuntimeError
        If no light curves are found or none are usable after cleaning.

    Example
    -------
    >>> time, flux, meta = wn.exoplanet.download_transit_lc('WASP-078', period_days=2.175)
    >>> print(meta['sector'], meta['n_transits'])
    """
    import lightkurve as lk

    sr = lk.search_lightcurve(target, mission='TESS', cadence=cadence, author=author)
    if len(sr) == 0:
        raise RuntimeError(f"✗ No TESS light curves found for '{target}'.")

    lc_collection = sr.download_all(quality_bitmask=quality_bitmask)

    best_sector = None
    best_score = -np.inf
    best_time = None
    best_flux = None
    all_sectors = []

    for lc_raw in lc_collection:
        sector = lc_raw.meta.get('SECTOR', None)
        try:
            t, f = clean_lc(lc_raw, sigma=sigma)
        except Exception:
            continue
        if len(t) == 0:
            continue

        span = t[-1] - t[0]
        score = span / period_days if period_days else span
        all_sectors.append((sector, len(t)))

        if score > best_score:
            best_score = score
            best_sector = sector
            best_time = t
            best_flux = f

    if best_time is None:
        raise RuntimeError(f"✗ No usable TESS sectors for '{target}' after cleaning.")

    metadata = {
        'sector': best_sector,
        'n_transits': best_score if period_days else None,
        'all_sectors': all_sectors,
    }
    return best_time, best_flux, metadata


# ── write_pipeline_csv ────────────────────────────────────────────────────────

def write_pipeline_csv(
    time: 'np.ndarray',
    flux: 'np.ndarray',
    t_mid: float,
    out_path: str,
) -> str:
    """
    Write a transit window to the 2-column whitenoise pipeline CSV format.

    Output format (no units, no comments — read directly by ``wn.analyze()``
    / ``wn.read_csv()``, neither of which support commented header lines)::

        time_min_from_mid,flux
        -45.2,0.9998
        ...

    To keep provenance metadata (planet, sector, midpoint-finding method,
    SNR, ...) alongside the data, save it separately — e.g. one row per
    transit in a companion DataFrame/CSV — rather than embedding it in this
    file.

    Parameters
    ----------
    time : array-like
        Time values (days), e.g. ``window['time']`` from
        :func:`~whitenoise.exoplanet.preprocess.extract_transit_window`.
    flux : array-like
        Flux values corresponding to *time*.
    t_mid : float
        Transit midpoint (days) — *time* is re-centered on this value and
        converted to minutes.
    out_path : str
        Destination CSV path. Parent directories are created if needed.

    Returns
    -------
    str
        *out_path*, unchanged, for chaining.

    Example
    -------
    >>> wn.exoplanet.write_pipeline_csv(
    ...     window['time'], window['flux'], t_mid,
    ...     'pipeline/WASP-078/WASP-078_T01.csv',
    ... )
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    time_min_from_mid = (time - t_mid) * 1440.0

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('time_min_from_mid,flux\n')
        for t_val, f_val in zip(time_min_from_mid, flux):
            fh.write(f'{t_val:.6f},{f_val:.8f}\n')

    return out_path


# ── read_pipeline_csv ─────────────────────────────────────────────────────────

def read_pipeline_csv(path: str) -> tuple:
    """
    Read a whitenoise pipeline CSV (``time_min_from_mid,flux``) back into arrays.

    Parameters
    ----------
    path : str
        Path to a pipeline CSV.

    Returns
    -------
    time_min_from_mid : np.ndarray
    flux : np.ndarray

    Example
    -------
    >>> t_min, flux = wn.exoplanet.read_pipeline_csv('pipeline/WASP-078/WASP-078_T01.csv')
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f'✗ File not found: {path}')

    time_vals, flux_vals = [], []
    with open(path, 'r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('time_min_from_mid'):
                continue
            t_str, f_str = line.split(',')
            time_vals.append(float(t_str))
            flux_vals.append(float(f_str))

    return np.asarray(time_vals, dtype=float), np.asarray(flux_vals, dtype=float)


# ── list_transit_csvs ─────────────────────────────────────────────────────────

def list_transit_csvs(folder: str, pattern: str = '*.csv') -> list:
    """
    Return a sorted list of pipeline CSV paths in *folder*.

    Parameters
    ----------
    folder : str
        Directory containing pipeline transit CSVs (e.g.
        ``pipeline/<Planet>/``).
    pattern : str, default ``'*.csv'``
        Glob pattern to filter files.

    Returns
    -------
    list of str
        Sorted file paths.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.

    Example
    -------
    >>> paths = wn.exoplanet.list_transit_csvs('pipeline/WASP-078/')
    >>> print(len(paths), 'transit CSVs found')
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'✗ Directory not found: {folder}')

    return sorted(glob.glob(os.path.join(folder, pattern)))
