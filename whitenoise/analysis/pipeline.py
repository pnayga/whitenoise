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


# ── Regime label ───────────────────────────────────────────────────────────────

def _regime(fit: FitResult | None) -> str:
    """
    Return a plain-English diffusion regime label from a FitResult.

    For μ-based models (cosine, sine, exponential, etc.):
      μ < 0.95          → 'subdiffusive'
      0.95 ≤ μ ≤ 1.05   → 'near-Brownian'
      1.05 < μ ≤ 2.0    → 'superdiffusive'
      μ > 2.0           → 'hyperballistic'

    For fBm (H parameter):
      H < 0.475         → 'subdiffusive'
      0.475 ≤ H ≤ 0.525 → 'near-Brownian'
      H > 0.525         → 'superdiffusive'

    For DNA model (plateau — a, b, c parameters): 'plateau'
    """
    if fit is None:
        return 'N/A'

    params = fit.params

    if 'mu' in params:
        # μ-based models: cosine, sine, exponential, and most others
        mu = params['mu']
        if mu < 0.95:
            return 'subdiffusive'
        elif mu <= 1.05:
            return 'near-Brownian'
        elif mu <= 2.0:
            return 'superdiffusive'
        else:
            return 'hyperballistic'

    elif 'H' in params:
        # fBm model — classify by Hurst exponent
        H = params['H']
        if H < 0.475:
            return 'subdiffusive'
        elif H <= 0.525:
            return 'near-Brownian'
        else:
            return 'superdiffusive'

    elif 'a' in params:
        # DNA model — saturating MSD, not a power-law diffusion regime
        return 'plateau (DNA)'

    return 'unknown'


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
    def regime(self) -> str:
        """Diffusion regime label derived from the fitted parameters."""
        return _regime(self.fit)

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
             Regime    : superdiffusive
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
            print(' Regime    : N/A')
        else:
            print(' Parameters:')
            for pname, pval in self.fit.params.items():
                se = self.fit.std_errors.get(pname, float('nan'))
                sym = _sym(pname)
                print(f'   {sym:<6} = {pval:.4f}  \u00b1  {se:.4f}')
            print(f' R\u00b2        = {self.fit.r_squared:.4f}')
            print(f' Regime    : {self.regime}')

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
        If ``True``, print ✓ progress lines and the final regime/R² summary.
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
            regime_str = _regime(fit_result)
            print(
                f'\u2713 Done.  R\u00b2 = {fit_result.r_squared:.4f}  '
                f'|  regime: {regime_str}'
            )

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
