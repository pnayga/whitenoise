"""
core/fitting.py — Parameter extraction with 95% confidence intervals.

Two fitting modes are always tried and compared:
  (1) Pure fit:   msd_theory(T, *params)        — N fixed at 1
  (2) Scaled fit: N · msd_theory(T, *params)    — N is a free parameter

The mode with the higher R² is selected as the winner. Both R² values are
always stored in FitResult and reported in summaries so the user can see
which formulation fits better for their data.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from scipy.optimize import curve_fit

from .models import MODELS


# ── Greek symbol display ───────────────────────────────────────────────────────

_GREEK: dict[str, str] = {
    'mu':   'μ',   # μ
    'nu':   'ν',   # ν
    'beta': 'β',   # β
    'H':    'H',
    'N':    'N',
    'a':    'a',
    'b':    'b',
    'c':    'c',
}


def _sym(name: str) -> str:
    return _GREEK.get(name, name)


# ── Default initial parameters and bounds ─────────────────────────────────────

DEFAULTS: dict[str, dict] = {
    'cosine': {
        'p0':     [1.2, 0.01],
        'bounds': ([0.01, 1e-9], [5.0, 10.0]),
    },
    'exponential': {
        'p0':     [1.2, 0.1],
        'bounds': ([0.01, 1e-9], [5.0, 100.0]),
    },
    'sine': {
        'p0':     [1.2, 0.01],
        'bounds': ([0.01, 1e-9], [5.0, 10.0]),
    },
    'fbm': {
        'p0':     [0.6],
        'bounds': ([0.01], [2.0]),
    },
    'exp_plateau': {
        'p0':     [5.0, 0.01, 3.0],
        'bounds': ([0.01, 1e-9, 0.01], [1000.0, 100.0, 1000.0]),
    },
    # inc_gamma: nu = shape (gamma order), mu = rate (saturation scale)
    'inc_gamma': {
        'p0':     [1.5, 0.1],
        'bounds': ([0.01, 1e-9], [10.0, 100.0]),
    },
    # bessel_jmu_nu: mu = memory param (denominator), nu = secondary order
    'bessel_jmu_nu': {
        'p0':     [1.0, 0.5],
        'bounds': ([0.01, -0.99], [10.0, 10.0]),
    },
}
# Backward-compat alias
DEFAULTS['dna'] = DEFAULTS['exp_plateau']


# ── FitResult dataclass ────────────────────────────────────────────────────────

@dataclass
class FitResult:
    """
    Results from a single MSD fit.

    Attributes
    ----------
    params : dict
        Fitted parameter values from the winning mode.
        Includes 'N' when fit_mode == 'scaled'.
    std_errors : dict
        Standard errors, same keys as *params*.
    confidence_intervals : dict
        95% confidence intervals, same keys as *params*.
    r_squared : float
        R² of the winning fit (whichever of pure/scaled was higher).
    r_squared_pure : float
        R² when fitting msd_theory(T, *params) directly (N = 1 fixed).
        nan if that fit failed.
    r_squared_scaled : float
        R² when fitting N · msd_theory(T, *params) (N free).
        nan if that fit failed.
    fit_mode : str
        ``'pure'`` or ``'scaled'`` — which mode won (higher R²).
    model : str
        Name of the fitted model.
    lags_used : np.ndarray
        Lag values supplied to ``fit_msd`` after applying *max_lag_fraction*.
    msd_fitted : np.ndarray
        Theoretical curve from the winning fit evaluated on *lags_used*.
    """

    params:               dict
    std_errors:           dict
    confidence_intervals: dict
    r_squared:            float
    model:                str
    lags_used:            np.ndarray
    msd_fitted:           np.ndarray
    r_squared_pure:       float = float('nan')
    r_squared_scaled:     float = float('nan')
    fit_mode:             str   = 'scaled'

    def summary(self) -> str:
        """
        Return a formatted box-drawing string showing fit results.

        Includes R² for both fitting modes so the user can compare.
        """
        W = 45

        def _line(text: str) -> str:
            return '│  ' + text.ljust(W - 2) + '│'

        mode_label = 'N·MSD' if self.fit_mode == 'scaled' else 'pure MSD'
        header = f'Model  : {self.model:<12}  [{mode_label} selected]'
        lines = [
            '┌' + '─' * W + '┐',
            _line('Fit Summary'),
            _line(header),
            '├' + '─' * W + '┤',
        ]

        for pname, pval in self.params.items():
            se       = self.std_errors.get(pname, float('nan'))
            lo, hi   = self.confidence_intervals.get(pname, (float('nan'), float('nan')))
            sym      = _sym(pname)
            lines.append(_line(f'  {sym:<4} = {pval:.4f} ± {se:.4f}'))
            lines.append(_line(f'         95% CI: ({lo:.3f}, {hi:.3f})'))

        lines.append('├' + '─' * W + '┤')

        def _r2_str(v: float) -> str:
            return f'{v:.4f}' if np.isfinite(v) else 'failed'

        pure_tag   = ' ← selected' if self.fit_mode == 'pure'   else ''
        scaled_tag = ' ← selected' if self.fit_mode == 'scaled' else ''
        lines.append(_line(f'  R² (pure MSD):   {_r2_str(self.r_squared_pure)}{pure_tag}'))
        lines.append(_line(f'  R² (N·MSD):      {_r2_str(self.r_squared_scaled)}{scaled_tag}'))
        lines.append('└' + '─' * W + '┘')
        return '\n'.join(lines)


# ── Internal fit helpers ───────────────────────────────────────────────────────

def _r2(msd_fit: np.ndarray, fitted: np.ndarray) -> float:
    """Compute R² between empirical and fitted arrays (finite values only)."""
    mask = np.isfinite(msd_fit) & np.isfinite(fitted) & (fitted < 1e29)
    if not np.any(mask):
        return float('nan')
    y    = msd_fit[mask]
    yhat = fitted[mask]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def _fit_pure(
    msd_fn,
    lags_fit: np.ndarray,
    msd_fit:  np.ndarray,
    param_names: list[str],
    p0:   list,
    bounds: tuple,
) -> 'tuple | None':
    """
    Fit msd_theory(T, *params) directly — N = 1 fixed.

    Returns (popt, perr, r2) where popt/perr are indexed to param_names.
    Returns None if fitting fails.
    """
    if len(param_names) == 0:
        # No-param model: evaluate directly, no optimization
        try:
            fitted = np.asarray(msd_fn(lags_fit), dtype=float)
            fitted = np.where(np.isfinite(fitted) & (fitted > 0), fitted, np.nan)
        except Exception:
            return None
        return np.array([]), np.array([]), _r2(msd_fit, fitted)

    def wrapper(T, *args):
        out = msd_fn(T, *args)
        if np.ndim(out) == 0:
            v = float(out)
            # Use 0.0 for undefined/negative model values instead of 1e30.
            # A large sentinel (1e30) creates residuals of order 1e60 when the
            # cosine model is undefined (cos(νT/2) ≤ 0), which makes the
            # optimizer unable to converge.  Returning 0.0 is physically
            # reasonable: the model simply predicts no displacement at those
            # lags, and the resulting moderate residual lets the optimizer
            # find good μ and ν on the valid rising portion.
            return 0.0 if (not np.isfinite(v) or v <= 0) else v
        out_arr = np.asarray(out, dtype=float)
        return np.where(np.isfinite(out_arr) & (out_arr > 0), out_arr, 0.0)

    try:
        popt, pcov = curve_fit(
            wrapper, lags_fit, msd_fit,
            p0=p0, bounds=bounds, maxfev=10_000,
        )
    except Exception:
        return None

    perr   = np.sqrt(np.diag(np.clip(pcov, 0.0, None)))
    fitted = np.asarray(wrapper(lags_fit, *popt), dtype=float)
    return popt, perr, _r2(msd_fit, fitted)


def _fit_scaled(
    msd_fn,
    lags_fit: np.ndarray,
    msd_fit:  np.ndarray,
    p0:   list,
    bounds: tuple,
) -> 'tuple | None':
    """
    Fit N · msd_theory(T, *params) — N is a free scaling parameter.

    Returns (popt_full, perr_full, r2) where popt_full is indexed to
    param_names + ['N'].
    Returns None if fitting fails.
    """
    p0_full     = list(p0) + [1.0]
    bounds_full = (list(bounds[0]) + [0.0], list(bounds[1]) + [np.inf])

    def wrapper(T, *args):
        phys = args[:-1]
        N    = args[-1]
        out  = msd_fn(T, *phys)
        if np.ndim(out) == 0:
            v = float(out)
            return 0.0 if (not np.isfinite(v) or v <= 0) else N * v
        out_arr = np.asarray(out, dtype=float) * N
        return np.where(np.isfinite(out_arr) & (out_arr > 0), out_arr, 0.0)

    try:
        popt, pcov = curve_fit(
            wrapper, lags_fit, msd_fit,
            p0=p0_full, bounds=bounds_full, maxfev=10_000,
        )
    except Exception:
        return None

    perr   = np.sqrt(np.diag(np.clip(pcov, 0.0, None)))
    fitted = np.asarray(wrapper(lags_fit, *popt), dtype=float)
    return popt, perr, _r2(msd_fit, fitted)


# ── fit_msd ────────────────────────────────────────────────────────────────────

def fit_msd(
    lags: np.ndarray,
    msd_empirical: np.ndarray,
    model: str = 'cosine',
    p0: list | None = None,
    bounds: tuple | None = None,
    max_lag_fraction: float = 1.0,
) -> 'FitResult | None':
    """
    Fit MSD theory to an empirical MSD curve using dual-mode comparison.

    Both fitting modes are always attempted:

    * **Pure fit** — ``msd_theory(T, *params)`` with N = 1 (fixed).
    * **Scaled fit** — ``N · msd_theory(T, *params)`` with N as a free parameter.

    The mode with the higher R² is selected. Both R² values are stored in
    :class:`FitResult` and shown in summaries for user comparison.

    Parameters
    ----------
    lags : np.ndarray
        Lag values from :func:`~whitenoise.core.msd.compute_msd`.
    msd_empirical : np.ndarray
        Empirical MSD values, same shape as *lags*.
    model : str, default ``'cosine'``
        Model name.  Must be a registered, available model.
    p0 : list, optional
        Initial guess for physical parameters only.
        Defaults to ``DEFAULTS[model]['p0']``.
    bounds : tuple, optional
        ``(lower, upper)`` for physical parameters only.
        Defaults to ``DEFAULTS[model]['bounds']``.
    max_lag_fraction : float, default 1.0
        Fraction of the lag range to use for fitting.

    Returns
    -------
    FitResult or None
        ``None`` if both fits fail.

    Notes
    -----
    * ``r_squared_pure`` = R² of the pure fit (no N scaling).
    * ``r_squared_scaled`` = R² of the N·MSD fit.
    * ``r_squared`` = best of the two (the winner).
    * ``fit_mode`` = ``'pure'`` or ``'scaled'`` — which won.
    * N is included in ``params`` only when ``fit_mode == 'scaled'``.
    """
    # ── Validate model ─────────────────────────────────────────────────────────
    if model not in MODELS:
        raise ValueError(
            f"✗ Unknown model '{model}'.\n"
            f"Run wn.list_models() to see all available models."
        )
    info = MODELS[model]
    if info['status'] != 'available':
        raise ValueError(
            f"✗ Model '{model}' is not yet implemented.\n"
            f"Run wn.list_models() to see available models."
        )

    param_names: list[str] = info['params']
    msd_fn = info['msd']

    # ── Prepare data ───────────────────────────────────────────────────────────
    lags_arr = np.asarray(lags, dtype=float)
    msd_arr  = np.asarray(msd_empirical, dtype=float)

    n_use    = max(1, int(len(lags_arr) * max_lag_fraction))
    lags_use = lags_arr[:n_use]
    msd_use  = msd_arr[:n_use]

    finite_vals = msd_use[np.isfinite(msd_use)]
    max_abs = float(np.max(np.abs(finite_vals))) if len(finite_vals) > 0 else 0.0
    if max_abs < 1e-20 or np.all(np.isnan(msd_use)):
        print(f"✗ Fitting failed for model '{model}': MSD is all-zero or all-nan.")
        return None

    valid    = np.isfinite(msd_use) & np.isfinite(lags_use)
    if not np.any(valid):
        print(f"✗ Fitting failed for model '{model}': no finite MSD values.")
        return None
    lags_fit = lags_use[valid]
    msd_fit  = msd_use[valid]

    # ── Defaults ───────────────────────────────────────────────────────────────
    # Resolve model key for DEFAULTS (handles 'dna' alias)
    defaults_key = model if model in DEFAULTS else 'exp_plateau' if model == 'dna' else model
    if p0 is None:
        p0 = list(DEFAULTS.get(defaults_key, {}).get('p0', [1.0] * len(param_names)))
    if bounds is None:
        d  = DEFAULTS.get(defaults_key, {})
        lb = list(d.get('bounds', ([1e-9] * len(param_names), [1e9] * len(param_names)))[0])
        ub = list(d.get('bounds', ([1e-9] * len(param_names), [1e9] * len(param_names)))[1])
        bounds = (lb, ub)

    # ── Cosine/sine: adaptive ν guess + dynamic ν upper bound ──────────────────
    #
    # The cosine model is only valid (positive MSD) where cos(νT/2) > 0,
    # i.e. T < π/ν.  When ν is too large the optimizer can settle in a
    # degenerate local minimum where most model predictions collapse to 0.
    #
    # Two fixes applied together:
    #
    # 1. Initial ν seed — start ν so its first cosine zero sits at 2×T_max,
    #    well outside the data: ν_est = π/(2×T_max).
    #    This places the optimizer in the basin where the cosine stays near 1
    #    and the model behaves as a smooth power law over the full lag range.
    #
    # 2. Dynamic ν upper bound — cap ν at π/T_max so the cosine first zero
    #    cannot move inside the lag window during optimisation.
    #    This turns the physical validity constraint into a hard bound.
    if model in ('cosine', 'sine') and len(lags_fit) > 4:
        T_max   = float(lags_fit[-1])
        nu_est  = float(np.pi / (2.0 * T_max))          # first zero at 2×T_max
        nu_max  = float(np.pi / T_max)                   # first zero exactly at T_max
        nu_est  = float(np.clip(nu_est, bounds[0][1], nu_max))
        nu_max  = float(np.clip(nu_max, bounds[0][1], bounds[1][1]))
        bounds  = (list(bounds[0]), list(bounds[1]))
        bounds[1][1] = nu_max                            # tighten upper bound for ν
        bounds  = (bounds[0], bounds[1])
        p0 = [p0[0], nu_est]

    # ── Dual fit ───────────────────────────────────────────────────────────────
    res_pure   = _fit_pure(msd_fn, lags_fit, msd_fit, param_names, p0, bounds)
    res_scaled = _fit_scaled(msd_fn, lags_fit, msd_fit, p0, bounds)

    r2_pure   = res_pure[2]   if res_pure   is not None else float('nan')
    r2_scaled = res_scaled[2] if res_scaled is not None else float('nan')

    if res_pure is None and res_scaled is None:
        print(f"✗ Fitting failed for model '{model}'")
        return None

    # Select winner: higher R² wins; scaled preferred on exact tie
    if np.isnan(r2_pure) or (np.isfinite(r2_scaled) and r2_scaled >= r2_pure):
        fit_mode  = 'scaled'
        popt, perr, r2 = res_scaled
        all_names = param_names + ['N']
    else:
        fit_mode  = 'pure'
        popt, perr, r2 = res_pure
        all_names = param_names

    # ── Build param dicts ──────────────────────────────────────────────────────
    params_out = {n: float(popt[i]) for i, n in enumerate(all_names)}
    se_out     = {n: float(perr[i]) for i, n in enumerate(all_names)}
    ci_out     = {
        n: (float(popt[i] - 1.96 * perr[i]), float(popt[i] + 1.96 * perr[i]))
        for i, n in enumerate(all_names)
    }

    # ── Evaluate winning curve on full lags_use window ─────────────────────────
    if fit_mode == 'scaled':
        def _final(T, *args):
            phys = args[:-1]
            N    = args[-1]
            out  = msd_fn(T, *phys)
            if np.ndim(out) == 0:
                v = float(out)
                return 1e30 if (not np.isfinite(v) or v <= 0) else N * v
            out_arr = np.asarray(out, dtype=float) * N
            return np.where(np.isfinite(out_arr), out_arr, 1e30)
    else:
        def _final(T, *args):
            if len(args) == 0:
                out = msd_fn(T)
            else:
                out = msd_fn(T, *args)
            if np.ndim(out) == 0:
                v = float(out)
                return 1e30 if (not np.isfinite(v) or v <= 0) else v
            out_arr = np.asarray(out, dtype=float)
            return np.where(np.isfinite(out_arr) & (out_arr > 0), out_arr, 1e30)

    msd_out = np.asarray(_final(lags_use, *popt), dtype=float)

    # ── Quality feedback ───────────────────────────────────────────────────────
    def _r2s(v: float) -> str:
        return f'{v:.4f}' if np.isfinite(v) else 'failed'

    mode_label  = 'N·MSD' if fit_mode == 'scaled' else 'pure MSD'
    r2_line     = (
        f"R²(pure)={_r2s(r2_pure)}  "
        f"R²(N·MSD)={_r2s(r2_scaled)}  "
        f"→ {mode_label} selected"
    )

    alternatives = [n for n in MODELS if MODELS[n]['status'] == 'available' and n != model and n != 'dna']
    if r2 >= 0.8:
        print(f"✓ Good fit  | {r2_line}")
    elif r2 >= 0.5:
        print(f"⚠ Moderate fit  | {r2_line}")
    else:
        print(f"⚠ Low R²  | {r2_line}  | Consider: {alternatives}")

    return FitResult(
        params=params_out,
        std_errors=se_out,
        confidence_intervals=ci_out,
        r_squared=r2,
        r_squared_pure=r2_pure,
        r_squared_scaled=r2_scaled,
        fit_mode=fit_mode,
        model=model,
        lags_used=lags_use,
        msd_fitted=msd_out,
    )
