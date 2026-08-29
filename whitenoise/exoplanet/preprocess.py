"""
whitenoise.exoplanet.preprocess
================================
Exoplanet-specific preprocessing utilities for TESS transit light curves.

These functions operate on plain (time, flux) arrays already loaded from a
light curve (see :mod:`whitenoise.exoplanet.io`). They convert a full-sector
light curve into a clean, gap-filled transit window ready for
``wn.analyze()``.

Typical use::

    import whitenoise as wn

    T14_d = wn.exoplanet.estimate_T14(P_days=1.509, Rs=1.131, Rp=1.551, a_AU=0.02536)

    t_mid, depth, snr, method = wn.exoplanet.find_empirical_midpoint(
        time, flux, t_lo=1234.5, t_hi=1234.7, T14_d=T14_d,
    )

    window = wn.exoplanet.extract_transit_window(time, flux, t_mid, T14_d)
    if window is not None:
        print(window['time'].shape, window['is_filled'].sum())
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

# Physical constants (SI, meters)
R_SUN = 6.957e8
R_JUP = 7.149e7
AU = 1.496e11


# ── estimate_T14 ─────────────────────────────────────────────────────────────

def estimate_T14(
    P_days: float,
    Rs: float,
    Rp: float,
    a_AU: float,
    inc_deg: float = 90.0,
) -> float:
    """
    Estimate total transit duration T14 (days).

    Uses the standard circular-orbit transit-duration formula::

        T14 = (P / pi) * arcsin( sqrt((Rs + Rp)^2 - b^2) / a )

    where ``b = a * cos(inc)`` is the impact parameter. With the default
    ``inc_deg=90.0`` (central transit, b=0) this reduces to::

        T14 = (P / pi) * arcsin((Rs + Rp) / a)

    Parameters
    ----------
    P_days : float
        Orbital period in days.
    Rs : float
        Stellar radius in solar radii.
    Rp : float
        Planet radius in Jupiter radii.
    a_AU : float
        Orbital semi-major axis in AU.
    inc_deg : float, default 90.0
        Orbital inclination in degrees. ``90.0`` gives a central transit
        (b=0), matching the simplified formula used elsewhere in this
        package's history. Pass the catalog inclination for a
        non-central-transit estimate.

    Returns
    -------
    float
        T14 in days.

    Examples
    --------
    >>> import whitenoise as wn
    >>> wn.exoplanet.estimate_T14(P_days=1.509, Rs=1.131, Rp=1.551, a_AU=0.02536)
    0.0623...

    >>> # With a non-central impact parameter
    >>> wn.exoplanet.estimate_T14(P_days=1.509, Rs=1.131, Rp=1.551, a_AU=0.02536, inc_deg=85.0)
    """
    a_m = a_AU * AU
    b = a_m * np.cos(np.radians(inc_deg))
    numerator = (Rs * R_SUN + Rp * R_JUP) ** 2 - b ** 2
    arg = np.clip(np.sqrt(np.clip(numerator, 0, None)) / a_m, 0, 1)
    return float((P_days / np.pi) * np.arcsin(arg))


# ── window_has_gap ────────────────────────────────────────────────────────────

def window_has_gap(
    time: 'np.ndarray',
    t_lo: float,
    t_hi: float,
    max_gap_min: float = 5.0,
) -> bool:
    """
    Check whether any gap wider than *max_gap_min* exists within [t_lo, t_hi].

    Parameters
    ----------
    time : array-like
        Observed time values (days).
    t_lo, t_hi : float
        Window bounds (days), same units as *time*.
    max_gap_min : float, default 5.0
        Gap threshold in minutes.

    Returns
    -------
    bool
        ``True`` if a gap wider than *max_gap_min* is found, or if fewer
        than 2 points fall in the window.

    Example
    -------
    >>> wn.exoplanet.window_has_gap(time, 1234.5, 1234.7, max_gap_min=5.0)
    False
    """
    time = np.asarray(time, dtype=float)
    mask = (time >= t_lo) & (time <= t_hi)
    t_w = time[mask]
    if len(t_w) < 2:
        return True
    return bool(np.any(np.diff(t_w) > max_gap_min / 1440.0))


# ── fill_gaps_nn ──────────────────────────────────────────────────────────────

def fill_gaps_nn(
    time: 'np.ndarray',
    flux: 'np.ndarray',
    t_lo: float,
    t_hi: float,
    cadence_min: float = 2.0,
    max_fill_gap_min: float = 60.0,
) -> tuple:
    """
    Fill gaps within [t_lo, t_hi] on a uniform-cadence grid using nearest-neighbour flux.

    Builds a uniform-cadence output grid spanning the observed points inside
    the window. For each grid point, the nearest observed point is used
    directly if within ``0.6 * cadence``; otherwise, if the grid point falls
    inside a real observational gap no wider than *max_fill_gap_min*, it is
    filled with the nearest observed flux value (not interpolated). Grid
    points inside gaps wider than *max_fill_gap_min* are dropped.

    Parameters
    ----------
    time, flux : array-like
        Full light-curve time (days) and flux arrays.
    t_lo, t_hi : float
        Window bounds (days).
    cadence_min : float, default 2.0
        Expected observing cadence in minutes (e.g. TESS SPOC 2-min).
    max_fill_gap_min : float, default 60.0
        Only gaps narrower than this (minutes) are filled.

    Returns
    -------
    t_out : np.ndarray
        Output time grid (days), observed + filled points, gaps wider than
        *max_fill_gap_min* excluded.
    f_out : np.ndarray
        Corresponding flux values.
    is_filled : np.ndarray of bool
        ``True`` for nearest-neighbour-filled points, ``False`` for real
        observed points.
    gap_info : list of (t_start, t_end, gap_minutes)
        Real observational gaps found in the window (regardless of whether
        they were filled).

    Example
    -------
    >>> t_w, f_w, is_filled, gap_info = wn.exoplanet.fill_gaps_nn(
    ...     time, flux, t_lo=1234.5, t_hi=1234.7, cadence_min=2.0, max_fill_gap_min=60.0,
    ... )
    >>> print(f'{is_filled.sum()} of {len(t_w)} points NN-filled')
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    cadence_d = cadence_min / 1440.0
    max_fill_d = max_fill_gap_min / 1440.0

    mask = (time >= t_lo) & (time <= t_hi)
    t_obs = time[mask]
    f_obs = flux[mask]

    if len(t_obs) < 2:
        return t_obs, f_obs, np.zeros(len(t_obs), dtype=bool), []

    t_out = np.arange(t_obs[0], t_obs[-1] + cadence_d * 0.5, cadence_d)
    f_out = np.full(len(t_out), np.nan)
    is_filled = np.ones(len(t_out), dtype=bool)

    for i, tg in enumerate(t_out):
        idx = int(np.argmin(np.abs(t_obs - tg)))
        dt = abs(t_obs[idx] - tg)
        if dt <= cadence_d * 0.6:
            f_out[i] = f_obs[idx]
            is_filled[i] = False
        else:
            before = t_obs[t_obs < tg]
            after = t_obs[t_obs > tg]
            if len(before) > 0 and len(after) > 0:
                if after[0] - before[-1] <= max_fill_d:
                    f_out[i] = f_obs[idx]

    gap_info = []
    for gi in np.where(np.diff(t_obs) > cadence_d * 1.5)[0]:
        gap_info.append((t_obs[gi], t_obs[gi + 1], (t_obs[gi + 1] - t_obs[gi]) * 1440.0))

    valid = ~np.isnan(f_out)
    return t_out[valid], f_out[valid], is_filled[valid], gap_info


# ── find_empirical_midpoint ───────────────────────────────────────────────────

def find_empirical_midpoint(
    t: 'np.ndarray',
    f: 'np.ndarray',
    t_lo: float,
    t_hi: float,
    T14_d: float,
) -> tuple:
    """
    Find the empirical transit midpoint within [t_lo, t_hi] without a forward model.

    Three-step hybrid approach:

    1. **Sliding scan** — 500 candidate midpoints spaced across the window;
       score each by the median flux in a ``±T14/2`` window; pick the
       minimum (deepest) as a robust coarse midpoint. Works even at SNR~1.
    2. **Parabolic refinement** — smooth the flux around the scan winner and
       fit a local quadratic for sub-cadence precision.
    3. **T14-informed flux-weighted centroid** — compute depth, noise, and
       SNR from the in-/out-of-window flux, and a flux-weighted centroid of
       the transit.

    The three estimates are blended based on SNR: high-SNR detections blend
    the parabolic and centroid estimates; low-SNR detections fall back to
    the parabolic estimate alone.

    Parameters
    ----------
    t, f : array-like
        Full light-curve time (days) and flux arrays.
    t_lo, t_hi : float
        Search window bounds (days) — typically a user- or
        catalog-selected region wide enough to contain one transit.
    T14_d : float
        Total transit duration (days), e.g. from :func:`estimate_T14`.

    Returns
    -------
    t_mid : float or None
        Estimated transit midpoint (days). ``None`` if fewer than 5 points
        fall in the window or the scan fails.
    depth : float
        Estimated transit depth (baseline − in-transit median flux).
    snr : float
        Depth / out-of-transit noise.
    method : str
        Which blend was used: ``'SCAN+BLEND(SNR=..)'``, ``'SCAN+PAR+nudge(SNR=..)'``,
        ``'SCAN+PARAB'``, ``'TOO_FEW'``, or ``'SCAN_FAILED'``.

    Example
    -------
    >>> t_mid, depth, snr, method = wn.exoplanet.find_empirical_midpoint(
    ...     time, flux, t_lo=1234.5, t_hi=1234.7, T14_d=0.0623,
    ... )
    >>> print(f't_mid={t_mid:.6f}  depth={depth:.5f}  SNR={snr:.1f}  [{method}]')
    """
    t = np.asarray(t, dtype=float)
    f = np.asarray(f, dtype=float)

    mask = (t >= t_lo) & (t <= t_hi)
    t_w = t[mask]
    f_w = f[mask]
    if len(t_w) < 5:
        return None, 0.0, 0.0, 'TOO_FEW'

    half = T14_d / 2.0

    # Step 1: sliding scan (500 candidates)
    candidates = np.linspace(t_lo + half, t_hi - half, 500)
    scores = np.full(len(candidates), np.nan)
    for ci, tc in enumerate(candidates):
        in_m = (t_w >= tc - half) & (t_w <= tc + half)
        if in_m.sum() >= 3:
            scores[ci] = np.median(f_w[in_m])
    valid_c = ~np.isnan(scores)
    if not valid_c.any():
        return None, 0.0, 0.0, 'SCAN_FAILED'
    t_scan = float(candidates[valid_c][np.argmin(scores[valid_c])])

    # Step 2: parabolic refinement around scan winner
    refine_hw = T14_d
    r_mask = (t_w >= t_scan - refine_hw) & (t_w <= t_scan + refine_hw)
    t_r = t_w[r_mask]
    f_r = f_w[r_mask]
    t_parab = t_scan
    parab_conf = 0.0
    if len(t_r) >= 5:
        smooth = uniform_filter1d(f_r, size=max(3, len(t_r) // 6))
        idx_min = int(np.argmin(smooth))
        hw2 = max(4, len(t_r) // 5)
        lo2 = max(0, idx_min - hw2)
        hi2 = min(len(t_r), idx_min + hw2 + 1)
        try:
            c = np.polyfit(t_r[lo2:hi2], f_r[lo2:hi2], 2)
            if c[0] > 0:
                t_v = -c[1] / (2 * c[0])
                if t_r.min() <= t_v <= t_r.max():
                    t_parab = float(t_v)
                    parab_conf = float(c[0])
        except Exception:
            pass

    # Step 3: T14 centroid around parabolic peak
    in_mask = (t_w >= t_parab - half) & (t_w <= t_parab + half)
    out_mask = ~in_mask
    t_centroid = None
    snr = 0.0
    depth = 0.0
    if in_mask.sum() >= 3 and out_mask.sum() >= 3:
        baseline = float(np.nanmedian(f_w[out_mask]))
        noise = float(np.nanstd(f_w[out_mask]))
        depth = baseline - float(np.nanmedian(f_w[in_mask]))
        snr = depth / noise if noise > 0 else 0.0
        weights = np.clip(baseline - f_w[in_mask], 0, None)
        if weights.sum() > 0 and snr >= 1.0:
            t_centroid = float(np.sum(weights * t_w[in_mask]) / weights.sum())

    # Blend
    if t_centroid is not None and snr >= 2.0:
        w_p = min(parab_conf * 1e6, 5.0)
        w_c = snr
        t_mid = (w_p * t_parab + w_c * t_centroid) / (w_p + w_c)
        method = f'SCAN+BLEND(SNR={snr:.1f})'
    elif t_centroid is not None and snr >= 1.0:
        t_mid = 0.75 * t_parab + 0.25 * t_centroid
        method = f'SCAN+PAR+nudge(SNR={snr:.1f})'
    else:
        t_mid = t_parab
        method = 'SCAN+PARAB'

    return float(t_mid), float(depth), float(snr), method


# ── extract_transit_window ────────────────────────────────────────────────────

def extract_transit_window(
    time: 'np.ndarray',
    flux: 'np.ndarray',
    t_mid: float,
    T14_d: float,
    window_periods: float = 1.5,
    cadence_min: float = 2.0,
    max_fill_gap_min: float = 60.0,
    min_points: int = 5,
    min_coverage: float = 0.5,
) -> 'dict | None':
    """
    Extract and gap-fill a "3T"-style transit window around a known midpoint.

    Combines :func:`fill_gaps_nn` with the windowing and quality-gating
    convention used for TESS transit extraction: a window of
    ``±window_periods * T14_d`` around *t_mid*, rejected if too few points
    remain after gap-filling, or if the recovered coverage of the expected
    uniform-cadence grid is too low.

    Parameters
    ----------
    time, flux : array-like
        Full light-curve time (days) and flux arrays.
    t_mid : float
        Transit midpoint (days), e.g. from :func:`find_empirical_midpoint`.
    T14_d : float
        Total transit duration (days), e.g. from :func:`estimate_T14`.
    window_periods : float, default 1.5
        Half-window width in units of T14 on each side of *t_mid*
        (i.e. the window is ``2 * window_periods * T14_d`` wide — the
        default of 1.5 gives the "3T" window: ``t_mid ± 1.5*T14``).
    cadence_min : float, default 2.0
        Expected observing cadence in minutes.
    max_fill_gap_min : float, default 60.0
        Only gaps narrower than this (minutes) are filled.
    min_points : int, default 5
        Minimum number of points required after gap-filling.
    min_coverage : float, default 0.5
        Minimum fraction of the expected uniform-cadence grid that must be
        present after gap-filling.

    Returns
    -------
    dict or None
        ``None`` if the window fails a quality gate (out of bounds, too few
        points, or insufficient coverage). Otherwise a dict with keys:

        ``'t_lo'``, ``'t_hi'`` — window bounds (days)
        ``'time'``, ``'flux'`` — gap-filled arrays within the window
        ``'is_filled'`` — bool array, ``True`` for NN-filled points
        ``'gap_info'`` — list of ``(t_start, t_end, gap_minutes)``
        ``'coverage'`` — fraction of expected uniform-cadence points present
        ``'n_filled'`` — number of NN-filled points

    Example
    -------
    >>> window = wn.exoplanet.extract_transit_window(time, flux, t_mid=1234.6, T14_d=0.0623)
    >>> if window is None:
    ...     print('window rejected by quality gates')
    ... else:
    ...     print(len(window['time']), 'points,', window['n_filled'], 'NN-filled')
    """
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    t_lo = t_mid - window_periods * T14_d
    t_hi = t_mid + window_periods * T14_d

    if t_lo < time[0] or t_hi > time[-1]:
        return None

    t_w, f_w, is_filled, gap_info = fill_gaps_nn(
        time, flux, t_lo, t_hi,
        cadence_min=cadence_min,
        max_fill_gap_min=max_fill_gap_min,
    )

    if len(t_w) < min_points:
        return None

    coverage = len(t_w) / max(1, (t_hi - t_lo) / (cadence_min / 1440.0))
    if coverage < min_coverage:
        return None

    return {
        't_lo': t_lo,
        't_hi': t_hi,
        'time': t_w,
        'flux': f_w,
        'is_filled': is_filled,
        'gap_info': gap_info,
        'coverage': coverage,
        'n_filled': int(is_filled.sum()),
    }
