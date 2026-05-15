"""
whitenoise.radiojove.preprocess
================================
RadioJOVE-specific preprocessing utilities.

These functions operate on arrays already loaded from CSV. They are all
optional — apply them before calling ``wn.analyze()`` or
``wn.radiojove.analyze_burst()`` if your data needs additional processing.

The standard RadioJOVE pipeline (GUI → CSV) already produces Z-scored
intensity values, so ``zscore_normalize()`` is only needed if you have
raw ADC counts or linear intensity data.

Typical use::

    time, intensity, meta = wn.radiojove.read_radiojove_csv('burst.csv')

    # If intensity is NOT already Z-scored:
    intensity = wn.radiojove.zscore_normalize(intensity)

    # Downsample from 0.1s to 1.0s cadence:
    time_1s, int_1s = wn.radiojove.resample(time, intensity, target_cadence=1.0)

    # Group a folder of CSVs by cadence:
    paths = wn.radiojove.list_burst_csvs('Solar/Type 3 Bursts/first trials/')
    groups = wn.radiojove.group_by_cadence(paths)
    paths_01s = groups[0.1]
    paths_10s = groups[1.0]
"""

from __future__ import annotations

import os

import numpy as np


# ── zscore_normalize ──────────────────────────────────────────────────────────

def zscore_normalize(intensity: 'np.ndarray') -> 'np.ndarray':
    """
    Z-score normalize an intensity array.

    Subtracts the mean and divides by the standard deviation.
    RadioJOVE CSVs produced by the GUI pipeline are already Z-scored —
    call this only if working with raw ADC counts or linear intensity.

    Parameters
    ----------
    intensity : array-like
        1D array of intensity values.

    Returns
    -------
    np.ndarray
        Z-score normalized array with mean ≈ 0 and std ≈ 1.
        Same length as input. If std is zero, returns mean-subtracted array.

    Example
    -------
    >>> intensity_z = wn.radiojove.zscore_normalize(raw_intensity)
    >>> print(intensity_z.mean(), intensity_z.std())  # ≈ 0.0, ≈ 1.0
    """
    x = np.asarray(intensity, dtype=float)
    mu = np.mean(x)
    sigma = np.std(x)
    if sigma == 0.0:
        return x - mu
    return (x - mu) / sigma


# ── resample ──────────────────────────────────────────────────────────────────

def resample(
    time: 'np.ndarray',
    intensity: 'np.ndarray',
    target_cadence: float,
) -> 'tuple[np.ndarray, np.ndarray]':
    """
    Resample a time series to a target cadence using linear interpolation.

    Use this to convert 0.1s-cadence data to 1.0s cadence (or vice versa)
    for comparing analyses across resolutions. Both upsampling and
    downsampling are supported.

    Parameters
    ----------
    time : array-like
        Original time axis in seconds. Must be strictly increasing.
    intensity : array-like
        Intensity values corresponding to *time*.
    target_cadence : float
        Desired time step in seconds (e.g. ``1.0`` for 1-second cadence,
        ``0.1`` for 100-millisecond cadence).

    Returns
    -------
    time_new : np.ndarray
        Resampled time axis from ``time[0]`` to ``time[-1]`` with step
        *target_cadence*. May be slightly shorter than the original if the
        total duration is not an exact multiple.
    intensity_new : np.ndarray
        Linearly interpolated intensity values at *time_new*.

    Examples
    --------
    >>> # Downsample from 0.1s to 1.0s cadence
    >>> time_1s, int_1s = wn.radiojove.resample(time_01s, int_01s, target_cadence=1.0)
    >>> print(len(time_01s), '->', len(time_1s))

    >>> # Upsample from 1.0s to 0.5s cadence
    >>> time_05s, int_05s = wn.radiojove.resample(time_1s, int_1s, target_cadence=0.5)
    """
    t = np.asarray(time, dtype=float)
    x = np.asarray(intensity, dtype=float)

    t_new = np.arange(t[0], t[-1], target_cadence)
    x_new = np.interp(t_new, t, x)

    return t_new, x_new


# ── group_by_cadence ──────────────────────────────────────────────────────────

def group_by_cadence(paths: list) -> dict:
    """
    Group a list of RadioJOVE CSV paths by their time cadence.

    Reads the cadence from each filename using
    :func:`~whitenoise.radiojove.io.parse_filename_metadata`.
    Files whose cadence cannot be determined are grouped under key ``None``.

    Parameters
    ----------
    paths : list of str
        File paths to RadioJOVE CSVs (full paths or basenames).

    Returns
    -------
    dict
        Maps cadence_s (float or ``None``) to a list of file paths.

    Example
    -------
    >>> paths = wn.radiojove.list_burst_csvs('Solar/Type 3 Bursts/first trials/')
    >>> groups = wn.radiojove.group_by_cadence(paths)
    >>> paths_01s = groups[0.1]    # all 0.1-second cadence files
    >>> paths_10s = groups[1.0]    # all 1.0-second cadence files
    >>> print(list(groups.keys()))
    [0.1, 1.0]
    """
    from .io import parse_filename_metadata

    groups: dict = {}
    for path in paths:
        meta = parse_filename_metadata(os.path.basename(path))
        key = meta['cadence_s']
        groups.setdefault(key, [])
        groups[key].append(path)

    return groups
