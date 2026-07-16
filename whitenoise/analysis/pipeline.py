"""
analysis/pipeline.py — High-level SWNA analysis pipeline.

Orchestrates the full analysis sequence:
  read_csv → detrend → normalize → compute_msd → fit_msd → AnalysisResult
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass

from ..io.reader import read_csv
from ..core.msd import compute_msd
from ..core.fitting import fit_msd, FitResult, _sym
from ..utils.preprocess import detrend, normalize as _normalize_fn


# ── Classification (model-family specific — do not merge these schemes) ────────
#
# Two distinct, independently-sourced classification schemes apply to two
# distinct model families. A single generic μ-only classification across all
# models is NOT scientifically justified and must not be reintroduced.
#
# 1. cosine / sine — diffusive-regime classification via α = 2μ − 1.
#    These models admit a short-time (T ≪ 1) reduction of the MSD to a power
#    law, MSD ≈ c·T^α. Classification is on α, not μ directly.
#    Source: Elnar et al. (2021, Climate Dynamics).
#
# 2. exponential — memory-persistence classification via μ directly.
#    MSD = Γ(μ)·β^(−μ)·T^(μ−1)·e^(−β/T) does not reduce to a clean power law
#    over the fitting range, so the α-based diffusive labels above do not
#    apply. Instead, classification comes from the power-law scaling
#    exponent (μ−1)/2 in the memory function
#    f(t−τ)h(τ) = (t−τ)^((μ−1)/2) · exp(−β/2τ) / τ:
#
#      μ = 1  -> exponent (μ−1)/2 = 0, so the power-law term (t−τ)^((μ−1)/2)
#                disappears; f(t−τ)h(τ) = exp(−β/2τ)/τ. Fluctuation behavior
#                is dominated purely by the exponential damping term:
#                a memoryless process.
#      μ > 1  -> exponent (μ−1)/2 > 0, so (t−τ)^((μ−1)/2) increases as
#                t−τ increases: the memory effect of an earlier fluctuation
#                at τ ≪ t is stronger. Non-Markovian, long memory.
#      μ < 1  -> exponent (μ−1)/2 < 0, so (t−τ)^((μ−1)/2) decreases as
#                t−τ increases: the memory effect of an earlier fluctuation
#                at τ ≪ t is weaker (though it strengthens again as τ → t).
#                Non-Markovian, short memory.
#
#    Source: Sithi et al. (2025, Physica Scripta 100, 015243).
#
# All other models (fbm, and the stub models) have no automatic
# classification until their own reduction/source is confirmed.

def _classify_cosine_sine(mu: float) -> str | None:
    """α = 2μ−1 diffusive-regime classification (Elnar et al. 2021)."""
    if mu is None or mu != mu:  # None or nan
        return None
    alpha = 2.0 * mu - 1.0
    if alpha < 1:
        return 'subdiffusive'
    elif alpha == 1:
        return 'brownian'
    elif alpha < 2:
        return 'superdiffusive'
    else:
        return 'hyperballistic'


def _classify_exponential(mu: float) -> str | None:
    """
    μ-based memory-persistence classification, via the (μ−1)/2 power-law
    scaling exponent in the memory function (Sithi et al. 2025).
    """
    if mu is None or mu != mu:  # None or nan
        return None
    if mu == 1:
        return 'memoryless'
    elif mu > 1:
        return 'non-Markovian, long memory'
    else:
        return 'non-Markovian, short memory'


def _alpha(fit: FitResult | None, model: str) -> float | None:
    """Return α = 2μ−1 for cosine/sine models, else None."""
    if fit is None or model not in ('cosine', 'sine'):
        return None
    return 2.0 * fit.params.get('mu', float('nan')) - 1.0


def _memory_exponent(fit: FitResult | None, model: str) -> float | None:
    """Return the (μ−1)/2 power-law scaling exponent for the exponential
    model's memory function, else None."""
    if fit is None or model != 'exponential':
        return None
    return (fit.params.get('mu', float('nan')) - 1.0) / 2.0


def _classify(fit: FitResult | None, model: str) -> str | None:
    """
    Dispatch to the correct model-family classification scheme.

    cosine/sine  -> α-based diffusive regime (Elnar et al. 2021)
    exponential  -> μ-based memory persistence (Sithi et al. 2025)
    other models -> None (no classification without a confirmed source)
    """
    if fit is None:
        return None
    if model in ('cosine', 'sine'):
        return _classify_cosine_sine(fit.params.get('mu'))
    elif model == 'exponential':
        return _classify_exponential(fit.params.get('mu'))
    return None


# ── AnalysisResult ─────────────────────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """
    Full output of a single SWNA analysis run.

    Attributes
    ----------
    dataset_name : str
        CSV filename without extension (or a user-supplied label).
    model : str
        SWNA model name used for fitting.
    fit : FitResult or None
        Fitting results.  ``None`` if fitting failed.
    lags : np.ndarray
        Lag array from :func:`~whitenoise.core.msd.compute_msd`.
    msd_empirical : np.ndarray
        Empirical MSD values.
    values : np.ndarray
        Preprocessed observable values (after detrend / normalize).
    time : np.ndarray
        Time array from the CSV.
    metadata : dict
        Column names, units, and source info from the reader.
    """

    dataset_name:  str
    model:         str
    fit:           FitResult | None
    lags:          np.ndarray
    msd_empirical: np.ndarray
    values:        np.ndarray
    time:          np.ndarray
    metadata:      dict

    @property
    def alpha(self) -> float | None:
        """α = 2μ−1, defined only for cosine/sine models (Elnar et al. 2021)."""
        return _alpha(self.fit, self.model)

    @property
    def memory_exponent(self) -> float | None:
        """
        (μ−1)/2, the power-law scaling exponent in the exponential model's
        memory function f(t−τ)h(τ) = (t−τ)^((μ−1)/2)·exp(−β/2τ)/τ.
        Defined only for the exponential model (Sithi et al. 2025); ``None``
        otherwise.
        """
        return _memory_exponent(self.fit, self.model)

    @property
    def regime(self) -> str | None:
        """
        Model-family-specific classification label, or ``None`` if the
        current model has no confirmed classification scheme.

        cosine/sine  : diffusive regime via α = 2μ−1 (Elnar et al. 2021)
                       -> 'subdiffusive' | 'brownian' | 'superdiffusive' | 'hyperballistic'
        exponential  : memory persistence via μ (Sithi et al. 2025)
                       -> 'memoryless' | 'non-Markovian, long memory' |
                          'non-Markovian, short memory'
        other models : None
        """
        return _classify(self.fit, self.model)

    def summary(self) -> None:
        """
        Print a formatted analysis summary block.

        Example output::

            ══════════════════════════════════════════
             SWNA Analysis Summary
            ══════════════════════════════════════════
             Dataset   : sunspot_data
             Model     : cosine
             Points    : 300
             Lags used : 150
            ──────────────────────────────────────────
             Parameters:
               μ      = 1.2341  ±  0.0082
               ν      = 0.0082  ±  0.0003
               N      = 2.4312  ±  0.0441
             R²        = 0.9823
            ──────────────────────────────────────────
             Units     : x=time (months), y=sunspot_number (count)
            ══════════════════════════════════════════
        """
        SEP_DOUBLE = '\u2550' * 42
        SEP_SINGLE = '\u2500' * 42

        print(SEP_DOUBLE)
        print(' SWNA Analysis Summary')
        print(SEP_DOUBLE)
        print(f' Dataset   : {self.dataset_name}')
        print(f' Model     : {self.model}')
        print(f' Points    : {len(self.values)}')
        print(f' Lags used : {len(self.lags)}')
        print(SEP_SINGLE)

        if self.fit is None:
            print(' Parameters: N/A (fitting failed)')
            print(' R\u00b2        : N/A')
        else:
            print(' Parameters:')
            for pname, pval in self.fit.params.items():
                se = self.fit.std_errors.get(pname, float('nan'))
                sym = _sym(pname)
                print(f'   {sym:<6} = {pval:.4f}  \u00b1  {se:.4f}')

            def _r2s(v: float) -> str:
                return f'{v:.4f}' if v == v else 'failed'  # nan check

            mode_lbl = 'N\u00b7MSD' if getattr(self.fit, 'fit_mode', 'scaled') == 'scaled' else 'pure MSD'
            r2_pure   = getattr(self.fit, 'r_squared_pure',   float('nan'))
            r2_scaled = getattr(self.fit, 'r_squared_scaled', float('nan'))
            pure_tag   = ' \u2190 selected' if self.fit.fit_mode == 'pure'   else ''
            scaled_tag = ' \u2190 selected' if self.fit.fit_mode == 'scaled' else ''
            print(f' R\u00b2 (pure MSD)  = {_r2s(r2_pure)}{pure_tag}')
            print(f' R\u00b2 (N\u00b7MSD)     = {_r2s(r2_scaled)}{scaled_tag}')

            regime_label = self.regime
            if regime_label is not None:
                if self.model in ('cosine', 'sine'):
                    print(f' \u03b1 = 2\u03bc\u22121  = {self.alpha:.4f}')
                    print(f' Regime    : {regime_label}  (Elnar et al. 2021)')
                elif self.model == 'exponential':
                    print(f' (\u03bc\u22121)/2   = {self.memory_exponent:.4f}')
                    print(f' Memory    : {regime_label}  (Sithi et al. 2025)')

        print(SEP_SINGLE)
        t_label = self.metadata.get('x_label', 'x')
        v_label = self.metadata.get('y_label', 'y')
        print(f' Units     : x={t_label}, y={v_label}')
        print(SEP_DOUBLE)


# ── analyze ────────────────────────────────────────────────────────────────────

def analyze(
    source,
    model: str = 'cosine',
    label: str = '',
    time: np.ndarray | None = None,
    detrend_method: str | None = None,
    normalize: bool = False,
    max_lag_fraction: float = 1.0,
    fit_kwargs: dict | None = None,
    verbose: bool = True,
) -> AnalysisResult:
    """
    Run the full SWNA pipeline on a CSV file or a data array.

    Steps
    -----
    1. Load data — from CSV path or array input.
    2. Detrend — if *detrend_method* is not ``None``.
    3. Normalize — z-score, if *normalize* is ``True``.
    4. Compute empirical MSD.
    5. Fit the chosen SWNA model.
    6. Return :class:`AnalysisResult`.

    Parameters
    ----------
    source : str or array-like
        * ``str`` — path to a whitenoise-format CSV.  Labels and units are
          read automatically from the header.
        * array-like (1-D) — pre-processed data array.  Must supply *label*.
    model : str, default ``'cosine'``
        SWNA model name.  Run ``wn.list_models()`` for options.
    label : str, optional
        Human-readable name for the dataset.  Auto-set from the CSV value
        column name when *source* is a CSV.  Required when *source* is an
        array.
    time : array-like, optional
        Time axis.  Only used when *source* is an array; ignored for CSV input
        (time comes from the file).
    detrend_method : str or None, default ``None``
        Passed to :func:`~whitenoise.utils.preprocess.detrend`.
        ``None`` (default) skips detrending — the raw values are used as-is.
        Choices: ``'linear'``, ``'polynomial'``, ``'mean'``,
        ``'moving_average'``.
    normalize : bool, default ``False``
        If ``True``, apply z-score normalization after detrending.
    max_lag_fraction : float, default 1.0
        Fraction of lags to use in fitting.  Default 1.0 means all N//2 lags
        are used, so empirical and fitted MSD always cover the same range.
    fit_kwargs : dict, optional
        Extra keyword arguments forwarded to
        :func:`~whitenoise.core.fitting.fit_msd` (e.g. ``p0``, ``bounds``).
    verbose : bool, default ``True``
        If ``True``, print ✓ progress lines and the final R² summary.
        If ``False``, suppress all output from the pipeline (note: fitting
        quality warnings from fit_msd itself are still printed).

    Returns
    -------
    AnalysisResult

    Raises
    ------
    ValueError
        If *source* is an array but *label* is not provided.

    Examples
    --------
    >>> # From CSV (recommended for research)
    >>> result = wn.analyze('sunspot.csv', model='exponential')
    >>> result.summary()

    >>> # From array (after manual detrending)
    >>> fluct = wn.detrend(values, method='moving_average', window=7)
    >>> result = wn.analyze(fluct, model='cosine', label='Sunspot residuals')
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    # ── Step 1: Load data ──────────────────────────────────────────────────────
    if isinstance(source, str):
        # CSV path — read file, extract labels and units automatically
        if verbose:
            print(f'\u2713 Loading: {source}')
        time_arr, values, metadata = read_csv(source)
        # dataset_name = filename stem (identifies the dataset, not the variable)
        dataset_name = os.path.splitext(os.path.basename(source))[0]
        # label for plot titles: caller-supplied > y_name > filename stem
        if not label:
            label = metadata.get('y_name', dataset_name)

    else:
        # Array input — require an explicit label so results are identifiable
        if not label:
            raise ValueError(
                "\u2717 Please provide label= when passing an array.\n"
                "  Example: wn.analyze(data, model='cosine', label='My System')"
            )
        values   = np.asarray(source, dtype=float).ravel()
        # Build a minimal time axis if none supplied
        time_arr = np.asarray(time, dtype=float).ravel() if time is not None \
                   else np.arange(len(values), dtype=float)
        metadata = {
            'source_file': 'array input',
            'x_label':     'index',
            'y_label':     label,
            'x_name':      'index',
            'y_name':      label,
            'x_unit':      '',
            'y_unit':      '',
            'n_points':    len(values),
        }
        dataset_name = label
        if verbose:
            print(f'\u2713 Array input: {len(values)} points  label="{label}"')

    # ── Steps 2 & 3: Optional preprocessing ───────────────────────────────────
    if detrend_method is not None:
        if verbose:
            print(f'\u2713 Detrending: method={detrend_method}')
        values = detrend(values, method=detrend_method)

    if normalize:
        if verbose:
            print('\u2713 Normalizing: z-score')
        values = _normalize_fn(values)

    # ── Step 4: Empirical MSD ──────────────────────────────────────────────────
    max_lag = len(values) // 2
    if verbose:
        print(f'\u2713 Computing MSD  ({len(values)} points, max_lag={max_lag})...')
    lags, msd_emp = compute_msd(values)

    # ── Step 5: Fit ────────────────────────────────────────────────────────────
    if verbose:
        print(f'\u2713 Fitting {model} model...')
    fit_result = fit_msd(
        lags, msd_emp,
        model=model,
        max_lag_fraction=max_lag_fraction,
        **fit_kwargs,
    )

    # ── Step 6: Report ─────────────────────────────────────────────────────────
    if verbose:
        if fit_result is None:
            print('\u2717 Fitting failed — check data or try a different model.')
        else:
            regime_label = _classify(fit_result, model)
            if regime_label is not None:
                print(f'✓ Done.  R² = {fit_result.r_squared:.4f}  |  {regime_label}')
            else:
                print(f'✓ Done.  R² = {fit_result.r_squared:.4f}')

    return AnalysisResult(
        dataset_name=dataset_name,
        model=model,
        fit=fit_result,
        lags=lags,
        msd_empirical=msd_emp,
        values=values,
        time=time_arr,
        metadata=metadata,
    )
