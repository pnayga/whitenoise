"""
core/msd.py — Empirical Mean Square Displacement computation.

MSD(Δ) = (1 / (N - Δ)) · Σᵢ [x(i + Δ) - x(i)]²
"""

from __future__ import annotations

import numpy as np


# ── Internal helper ───────────────────────────────────────────────────────────

def _to_1d_array(x) -> np.ndarray:
    """
    Convert any array-like to a 1-D float ``np.ndarray``.

    Accepted inputs: ``np.ndarray``, ``pd.Series``, ``list``, ``tuple``.
    Isolated NaN values are replaced by linear interpolation via
    ``np.interp`` before the array is returned.

    Parameters
    ----------
    x : array-like
        Input data.

    Returns
    -------
    np.ndarray
        1-D float array, NaN-free.

    Raises
    ------
    ValueError
        * ``'✗ Input must be 1D. Got shape {shape}.'``
          if the result is not 1-dimensional.
        * ``'✗ Need at least 10 data points. Got {n}.'``
          if fewer than 10 values are present.
        * ``'✗ Too many missing values ({pct:.0f}% NaN). Check your data.'``
          if more than 50 % of values are NaN.
    """
    # Try pandas first so we don't lose the underlying dtype
    try:
        import pandas as pd
        if isinstance(x, pd.Series):
            arr = x.to_numpy(dtype=float)
        else:
            arr = np.asarray(x, dtype=float)
    except Exception:
        arr = np.asarray(x, dtype=float)

    # 1. Dimensionality check
    if arr.ndim != 1:
        raise ValueError(f'✗ Input must be 1D. Got shape {arr.shape}.')

    n = len(arr)

    # 2. Minimum length check
    if n < 10:
        raise ValueError(f'✗ Need at least 10 data points. Got {n}.')

    # 3. NaN fraction check
    n_nan = int(np.sum(np.isnan(arr)))
    if n_nan > 0:
        pct = 100.0 * n_nan / n
        if pct > 50.0:
            raise ValueError(
                f'✗ Too many missing values ({pct:.0f}% NaN). Check your data.'
            )
        # 4. Interpolate remaining NaN
        valid_idx = np.where(~np.isnan(arr))[0]
        all_idx = np.arange(n)
        arr = np.interp(all_idx, valid_idx, arr[valid_idx])

    return arr


# ── Public API ────────────────────────────────────────────────────────────────

def compute_msd(
    x,
    max_lag: int | None = None,  # defaults to N//2; hard-capped at N//2 (beyond N/2, estimates use too few pairs)
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical Mean Square Displacement (MSD) of a 1-D time series.

    The MSD at lag Δ is defined as:

    .. math::

        \\text{MSD}(\\Delta) = \\frac{1}{N - \\Delta}
            \\sum_{i=0}^{N-\\Delta-1} \\bigl[x(i+\\Delta) - x(i)\\bigr]^2

    .. note::

        The sum uses 0-based indexing (i = 0 … N−Δ−1) to match Python/NumPy
        conventions. In textbook formulations the sum starts at i = 1 with
        upper limit N−Δ — both yield the same N−Δ terms and identical results.

    Parameters
    ----------
    x : array-like (1D)
        Sequential values to analyze.  This can be any 1-D sequence of
        equally spaced measurements — a physical observable, a time series,
        a spatial profile, or any other uniformly sampled quantity.
        Accepts ``np.ndarray``, ``pd.Series``, ``list``, or ``tuple``.
        Converted internally to a 1-D float array.
        Must contain at least 10 finite points.

        .. note::

            The values are assumed to be **equally spaced** (uniform step
            spacing). If the original data has gaps, represent missing
            positions as ``NaN`` before passing — isolated NaN values are
            filled by linear interpolation internally. If more than 50 % of
            the values are NaN the function raises a ``ValueError``.

            For 2-column CSV data (index + observable), use
            ``wn.read_csv()`` first to extract the two columns, then pass
            the observable column here.
    max_lag : int, optional
        Maximum lag to compute.  Defaults to ``len(x) // 2``.
        Capped at ``len(x) // 2`` — beyond N/2 the estimates use fewer
        than N/2 samples and become statistically unreliable.
    normalize : bool, default False
        If ``True``, divide every MSD value by ``MSD[1]`` (lag = 1)
        so that the first returned value equals 1.0.

    Returns
    -------
    lags : np.ndarray of int
        Integer lag values ``[1, 2, …, max_lag]``,  shape ``(max_lag,)``.
    msd : np.ndarray of float
        Corresponding MSD values,  shape ``(max_lag,)``.

    Raises
    ------
    ValueError
        Propagated from :func:`_to_1d_array` for bad input.

    Examples
    --------
    >>> time, values, meta = wn.read_csv('sunspot.csv')
    >>> lags, msd = wn.compute_msd(values)
    >>> lags, msd = wn.compute_msd(values, max_lag=100, normalize=True)
    """
    # Convert any array-like input to a clean 1-D float array.
    # NaN interpolation and dimension/length checks happen inside here.
    arr = _to_1d_array(x)
    n = len(arr)

    # Cap max_lag at N//2 (mentor rule):
    # beyond N/2, each lag has fewer than N/2 pairs → statistically unreliable.
    if max_lag is None:
        max_lag = n // 2
    max_lag = min(int(max_lag), n // 2)

    # Lag values: Δ = 1, 2, 3, …, max_lag (never 0 — since zero lag gives zero displacement).
    lags = np.arange(1, max_lag + 1, dtype=int)

    # Pre-allocate the output array (one MSD value per lag).
    msd = np.empty(max_lag, dtype=float)

    for i, lag in enumerate(lags):
        # Compute all displacements x(j+Δ) - x(j) for this lag Δ in one shot.
        # arr[lag:]  = [x(Δ), x(Δ+1), …, x(N-1)]   ← "right" points
        # arr[:-lag] = [x(0), x(1),   …, x(N-Δ-1)] ← "left"  points
        # The subtraction pairs every j with j+Δ simultaneously (NumPy vectorization).
        # Result: an array of (N-Δ) displacement values.
        diff = arr[lag:] - arr[:-lag]

        # MSD(Δ) = (1 / (N-Δ)) · Σ [x(j+Δ) - x(j)]²
        # np.mean automatically divides by len(diff) = N-Δ,
        # so the unbiased denominator is handled without extra code.
        msd[i] = np.mean(diff * diff)

    if normalize:
        # Divide all MSD values by MSD(Δ=1) so the first point equals 1.0.
        # This removes amplitude differences between datasets while preserving
        # the growth shape (scaling exponent μ is unchanged).
        msd = msd / msd[0]

    return lags, msd
