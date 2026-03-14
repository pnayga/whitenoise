"""
utils/preprocess.py — Optional preprocessing helpers for whitenoise.

These functions are called manually by the researcher BEFORE analyze().
The pipeline never invokes them automatically.

Typical workflow::

    time, values, meta = wn.read_csv('co2.csv')
    fluct = wn.detrend(values, method='polynomial', poly_order=2)
    result = wn.analyze(fluct, model='cosine', label='CO2 fluctuations')
"""

from __future__ import annotations

import numpy as np


# ── Internal helper ───────────────────────────────────────────────────────────

def _as_1d(values) -> np.ndarray:
    """Convert array-like to 1-D float ndarray."""
    return np.asarray(values, dtype=float).ravel()


# ── Public API ────────────────────────────────────────────────────────────────

def detrend(
    values,
    method: str = 'linear',
    poly_order: int = 1,
    window: int = 7,
) -> np.ndarray:
    """
    Remove a trend from a time series to extract fluctuations.

    Parameters
    ----------
    values : array-like (1D)
        Input time series.
    method : str, default ``'linear'``
        Detrending method:

        * ``'linear'``          — subtract a degree-1 (straight-line) fit.
          (Bernido et al. §3.1)
        * ``'polynomial'``      — subtract a degree-``poly_order`` polynomial fit.
          (Bernido et al. §3.1)
        * ``'mean'``            — subtract the global mean only.
        * ``'moving_average'``  — subtract a uniform moving average of width
          ``window``. (Bernido et al. §3.2, sunspot example)

    poly_order : int, default 1
        Polynomial degree.  Only used when ``method='polynomial'``.
    window : int, default 7
        Moving-average window size.  Only used when ``method='moving_average'``.
        Odd values recommended; even values are silently incremented by 1.

    Returns
    -------
    np.ndarray
        Detrended fluctuations, same length as ``values``.

    Raises
    ------
    ValueError
        ``'✗ Unknown detrend method …'`` for unrecognised ``method``.

    Examples
    --------
    >>> time, values, meta = wn.read_csv('co2.csv')
    >>> fluct = wn.detrend(values, method='polynomial', poly_order=2)
    >>> result = wn.analyze(fluct, model='cosine', label='CO2')

    >>> # Sunspot detrending as in Bernido et al. §3.2
    >>> fluct = wn.detrend(sunspot_values, method='moving_average', window=7)
    """
    arr = _as_1d(values)
    idx = np.arange(len(arr), dtype=float)

    if method == 'linear':
        # Step 1: Fit a straight line to the data (index as x-axis)
        # Step 2: Subtract the line — residuals are the fluctuations
        coeffs = np.polyfit(idx, arr, 1)
        return arr - np.polyval(coeffs, idx)

    elif method == 'polynomial':
        # Step 1: Fit a polynomial of degree poly_order
        # Step 2: Subtract the polynomial trend — residuals are the fluctuations
        coeffs = np.polyfit(idx, arr, int(poly_order))
        return arr - np.polyval(coeffs, idx)

    elif method == 'mean':
        # Subtract the global mean — simplest detrending, removes DC offset
        return arr - np.mean(arr)

    elif method == 'moving_average':
        # Step 1: Force odd window (symmetric kernel)
        w = int(window)
        if w % 2 == 0:
            w += 1
        # Step 2: Compute uniform moving average (box filter)
        #         mode='same' keeps output the same length as input
        kernel = np.ones(w) / w
        ma = np.convolve(arr, kernel, mode='same')
        # Step 3: Subtract moving average — residuals are the fluctuations
        #         (Bernido et al. §3.2: "subtracting the raw data from the moving average")
        return arr - ma

    else:
        raise ValueError(
            f"✗ Unknown detrend method '{method}'. "
            f"Choose: 'linear', 'polynomial', 'mean', 'moving_average'."
        )


def normalize(
    values,
    method: str = 'zscore',
) -> np.ndarray:
    """
    Normalize a time series.

    Parameters
    ----------
    values : array-like (1D)
        Input time series.
    method : str, default ``'zscore'``
        Normalization method:

        * ``'zscore'`` — subtract mean, divide by standard deviation.
        * ``'minmax'`` — scale to the interval ``[0, 1]``.
        * ``'maxabs'`` — divide by the maximum absolute value → range ``[-1, 1]``.
                         Preserves zero and sign; best for oscillating or
                         detrended data where zero has physical meaning.
        * ``'mean'``   — divide by the mean only (preserves shape).

    Returns
    -------
    np.ndarray
        Normalized series, same length as ``values``.

    Raises
    ------
    ValueError
        ``'✗ Unknown normalize method …'`` for unrecognised ``method``.
    """
    arr = _as_1d(values)

    if method == 'zscore':
        return (arr - np.mean(arr)) / np.std(arr)
    elif method == 'minmax':
        lo, hi = np.min(arr), np.max(arr)
        return (arr - lo) / (hi - lo)
    elif method == 'maxabs':
        m = np.max(np.abs(arr))
        if m == 0:
            raise ValueError("✗ Cannot apply 'maxabs' normalization: all values are zero.")
        return arr / m
    elif method == 'mean':
        return arr / np.mean(arr)
    else:
        raise ValueError(
            f"✗ Unknown normalize method '{method}'. "
            f"Choose: 'zscore', 'minmax', 'maxabs', 'mean'."
        )


def smooth(
    values,
    window: int = 5,
    method: str = 'moving_average',
) -> np.ndarray:
    """
    Smooth a time series.  Output is always the same length as the input.

    Parameters
    ----------
    values : array-like (1D)
        Input time series.
    window : int, default 5
        Number of points in the smoothing kernel.  Must be a positive odd
        integer.  If an even value is given it is incremented to the next
        odd integer and a warning is printed:
        ``"⚠ Window size must be odd. Using {window+1} instead."``
    method : str, default ``'moving_average'``
        Smoothing method:

        * ``'moving_average'`` — uniform (box) kernel via ``np.convolve``.
        * ``'gaussian'``       — Gaussian kernel with
          ``sigma = window / 4`` via ``scipy.ndimage.gaussian_filter1d``.

    Returns
    -------
    np.ndarray
        Smoothed series, same length as ``values``.

    Raises
    ------
    ValueError
        ``'✗ Unknown smooth method …'`` for unrecognised ``method``.
    """
    arr = _as_1d(values)

    window = int(window)
    if window % 2 == 0:
        print(f'⚠ Window size must be odd. Using {window + 1} instead.')
        window += 1

    if method == 'moving_average':
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='same')
    elif method == 'gaussian':
        from scipy.ndimage import gaussian_filter1d
        sigma = window / 4.0
        return gaussian_filter1d(arr, sigma=sigma)
    else:
        raise ValueError(
            f"✗ Unknown smooth method '{method}'. "
            f"Choose: 'moving_average', 'gaussian'."
        )


# ── Power Spectral Density detrending ─────────────────────────────────────────
# Bernido et al. §3.3 — used for predator-prey algae population data (Blasius et al. 2020)

def detrend_psd(
    values,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Remove dominant periodic components via Power Spectral Density (PSD) analysis.

    Method (Bernido et al. §3.3):

    1. Compute the FFT of the signal.
    2. Compute the normalized PSD = |FFT|² / max(|FFT|²).
    3. Zero out frequency components where normalized PSD > ``threshold``
       (these are the dominant/deterministic peaks).
    4. Apply inverse FFT — only the residual fluctuations remain.

    Parameters
    ----------
    values : array-like (1D)
        Input time series.
    threshold : float, default 0.5
        PSD cutoff (0–1 scale).  Components with normalized power above this
        value are removed.  Lower values filter more aggressively.
        Typical choices: 0.2, 0.3, 0.5, 1.0 (Bernido et al. Fig. 3.5).

    Returns
    -------
    np.ndarray
        Fluctuations with dominant periodic components removed, same length
        as ``values``.

    Examples
    --------
    >>> # Algae predator-prey data — remove dominant cycle
    >>> fluct = wn.detrend_psd(algae_values, threshold=0.3)
    >>> result = wn.analyze(fluct, model='cosine', label='Algae fluctuations')
    """
    arr = _as_1d(values)
    n   = len(arr)

    # Step 1: Forward FFT — convert time domain → frequency domain
    fft_vals = np.fft.rfft(arr)

    # Step 2: Normalized PSD = |FFT|² / max(|FFT|²)
    #         This scales the dominant peak to 1.0, making threshold intuitive
    psd      = np.abs(fft_vals) ** 2
    psd_norm = psd / np.max(psd)

    # Step 3: Zero out frequency bins whose normalized power exceeds threshold
    #         These are the deterministic/periodic components to be removed
    fft_filtered = fft_vals.copy()
    fft_filtered[psd_norm > threshold] = 0.0

    # Step 4: Inverse FFT — return to time domain; only fluctuations remain
    return np.fft.irfft(fft_filtered, n=n)


# ── Cycle averaging / Fingerprinting ──────────────────────────────────────────
# Bernido et al. §3.4 — used for algae population data (Blasius et al. 2020)
# References: Jutras & Howard (2015), Cole et al. (2009)

def detrend_fingerprint(
    values,
    cycle_length: int = None,
    trough_indices=None,
) -> np.ndarray:
    """
    Cycle averaging (fingerprinting) detrending.

    Method (Bernido et al. §3.4):

    1. Segment the signal at cycle troughs.
    2. Align all segments at their maxima by zero-padding the start.
    3. Equalize segment lengths by zero-padding the end.
    4. Compute the fingerprint = mean of all padded segments.
    5. Subtract the fingerprint from each original segment (padding excluded).
    6. Concatenate residuals — output has same length as input.

    Parameters
    ----------
    values : array-like (1D)
        Input time series with a dominant repeating cycle.
    cycle_length : int, optional
        If provided, splits the signal into equal-length chunks of this size.
        Simpler alternative when cycles are uniform.
    trough_indices : list of int, optional
        Indices of cycle boundaries (trough positions).
        If neither ``cycle_length`` nor ``trough_indices`` is given, troughs
        are detected automatically using ``scipy.signal.find_peaks``.

    Returns
    -------
    np.ndarray
        Fluctuations with the repeating cycle pattern removed, same length
        as ``values``.

    Raises
    ------
    ValueError
        If no valid cycles can be identified.

    Examples
    --------
    >>> # Algae predator-prey — repeating predator-prey cycle
    >>> fluct = wn.detrend_fingerprint(algae_values, cycle_length=20)
    >>> result = wn.analyze(fluct, model='cosine', label='Algae residuals')
    """
    from scipy.signal import find_peaks

    arr = _as_1d(values)
    n   = len(arr)

    # Step 1: Determine cycle boundaries (trough positions)
    if trough_indices is not None:
        # User-provided trough indices
        boundaries = sorted(int(i) for i in trough_indices)
    elif cycle_length is not None:
        # Equal-length cycles
        boundaries = list(range(0, n, int(cycle_length)))
    else:
        # Auto-detect troughs: find local minima of the signal
        troughs, _ = find_peaks(-arr)
        boundaries  = [0] + list(troughs)

    # Ensure the final boundary reaches the end of the array
    if not boundaries or boundaries[-1] < n - 1:
        boundaries.append(n)

    # Step 2: Extract individual cycle segments (skip segments too short to use)
    segments   = []   # list of 1-D arrays, one per cycle
    seg_starts = []   # original start index of each segment in arr
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg = arr[start:end].copy()
        if len(seg) > 1:            # ignore degenerate 0- or 1-point segments
            segments.append(seg)
            seg_starts.append(start)

    if not segments:
        raise ValueError(
            "✗ Could not identify valid cycles. "
            "Try providing trough_indices= or cycle_length=."
        )

    # Step 3: Align segments at their maxima by zero-padding the start
    #         max_positions[i] = index of the peak within segment i
    max_positions = [int(np.argmax(seg)) for seg in segments]
    max_max_pos   = max(max_positions)   # largest peak offset across all segments

    padded = []
    for seg, mp in zip(segments, max_positions):
        pad_before = max_max_pos - mp                        # zeros to prepend
        padded.append(np.concatenate([np.zeros(pad_before), seg]))

    # Step 4: Equalize lengths by zero-padding the end of shorter segments
    max_len    = max(len(p) for p in padded)
    padded_eq  = np.array([
        np.concatenate([p, np.zeros(max_len - len(p))]) for p in padded
    ])

    # Step 5: Fingerprint = mean of all length-equalized, max-aligned segments
    fingerprint = np.mean(padded_eq, axis=0)

    # Step 6: Subtract fingerprint from each original segment
    #         Use only the fingerprint slice that corresponds to actual data
    #         (skip the zero-padding portion) — no new points are added
    fluct = arr.copy()
    for seg, mp, start in zip(segments, max_positions, seg_starts):
        end      = start + len(seg)
        fp_start = max_max_pos - mp           # fingerprint offset for this segment
        fp_slice = fingerprint[fp_start: fp_start + len(seg)]
        fluct[start:end] = seg - fp_slice

    return fluct
