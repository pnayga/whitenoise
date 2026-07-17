"""
whitenoise.radiojove.pipeline
==============================
Step-by-step SWNA pipeline for RadioJOVE radio burst CSV data.

Each function corresponds to one stage of the analysis. Use them
individually for full control or call :func:`analyze_burst` for the
complete pipeline in a single call.

Typical workflow::

    import whitenoise as wn

    # Single burst
    out = wn.radiojove.analyze_burst(
        '230728154000Higgins_Home_region_1_0.1s.csv',
        model='exponential',
    )
    out['figure'].show()
    print(out['result'].fit.params.get('mu'))

    # Batch — whole folder
    df = wn.radiojove.batch_analyze_bursts(
        'Solar/Type 3 Bursts/first trials/',
        model='exponential',
        output_dir='swna_results/',
    )
    print(df[['dataset', 'mu', 'r_squared']])

    # Compare bursts at the same cadence
    paths = wn.radiojove.list_burst_csvs('Solar/Type 3 Bursts/first trials/')
    groups = wn.radiojove.group_by_cadence(paths)
    cr = wn.radiojove.compare_bursts(groups[0.1], model='exponential')
    wn.print_comparison(cr)

Model guidance
--------------
- **exponential** (recommended for Type III solar bursts):
  MSD grows monotonically — matches fast-rise, slow-decay burst profile.
  μ > 1 indicates faster-than-linear MSD growth (typical for Type III bursts).
  Classification for this model is μ-based memory persistence — see
  "Classification" below.

- **cosine**: MSD oscillates — use if you observe periodic fluctuations.
  (Rarely appropriate for radio bursts; fails on monotonic MSD.)

- **fbm**: Fractional Brownian motion. Hurst exponent H instead of μ.
  H = 0.5 → Brownian; H > 0.5 → faster-than-linear MSD growth.
  No automatic classification is attached to this model.

Run ``wn.list_models()`` to see all available models.

Classification
--------------
``result.regime`` is model-family-specific — the two schemes are not
interchangeable and must not be conflated:

- cosine/sine: diffusive regime via α = 2μ−1
  (Elnar et al. 2021, Climate Dynamics).
- exponential: memory persistence via μ directly
  (Tey et al. 2024, Physica Scripta 100, 015243).
- All other models: no classification (``result.regime`` is ``None``).
"""

from __future__ import annotations

import contextlib
import io
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Colour palette consistent with whitenoise publish.py
_EMPIRICAL   = '#2C3E50'
_THEORETICAL = '#E74C3C'
_SUBTLE      = '#BDC3C7'


# ── analyze_burst ─────────────────────────────────────────────────────────────

def analyze_burst(
    path: str,
    model: str = 'exponential',
    detrend_method: 'str | None' = 'linear',
    normalize: bool = False,
    output_dir: 'str | None' = None,
    verbose: bool = True,
) -> dict:
    """
    Full SWNA pipeline for a single RadioJOVE burst CSV.

    Reads the CSV, runs ``wn.analyze()``, generates a dual-panel MSD plot,
    and optionally saves the plot and a result row summary.

    Parameters
    ----------
    path : str
        Path to a RadioJOVE whitenoise-format CSV.
    model : str, default ``'exponential'``
        SWNA model.  ``'exponential'`` is recommended for Type III solar
        bursts and most Jovian S-bursts.  Run ``wn.list_models()`` to see
        all options.
    detrend_method : str or None, default ``'linear'``
        Detrending applied before MSD computation.  ``'linear'`` removes a
        linear trend within the burst window.  Pass ``None`` to skip.
    normalize : bool, default ``False``
        Z-score normalize values before fitting. Set ``True`` if the CSV
        contains raw intensity rather than Z-scored values.
    output_dir : str, optional
        If provided, the MSD plot PNG is saved here as
        ``<dataset>_msd.png``.  Directory is created if it does not exist.
    verbose : bool, default True
        Print μ and R² after fitting.

    Returns
    -------
    dict with keys:

        ``'path'``    — the input *path* argument
        ``'dataset'`` — filename stem (no extension)
        ``'result'``  — :class:`~whitenoise.analysis.pipeline.AnalysisResult`
                        (``None`` if analysis failed)
        ``'figure'``  — :class:`matplotlib.figure.Figure` with dual-panel plot
        ``'row'``     — dict summary row for building DataFrames

    Examples
    --------
    >>> out = wn.radiojove.analyze_burst('burst_region_1_0.1s.csv')
    >>> print(out['result'].fit.params.get('mu'))
    1.34

    >>> # Save the plot automatically
    >>> out = wn.radiojove.analyze_burst(
    ...     'burst_region_1_0.1s.csv',
    ...     output_dir='swna_results/',
    ... )
    >>> out['figure'].show()
    """
    import whitenoise as wn

    stem = os.path.splitext(os.path.basename(path))[0]

    if verbose:
        print(f'\n  {stem}')
        print(f'  {"-" * 56}')

    result = None
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = wn.analyze(
                path,
                model=model,
                detrend_method=detrend_method,
                normalize=normalize,
                verbose=False,
            )
    except Exception as exc:
        if verbose:
            print(f'  Analysis failed: {exc}')

    if verbose and result is not None and result.fit is not None:
        mu = result.fit.params.get('mu', float('nan'))
        lo, hi = result.fit.confidence_intervals.get('mu', (float('nan'), float('nan')))
        r2_pure   = getattr(result.fit, 'r_squared_pure',   float('nan'))
        r2_scaled = getattr(result.fit, 'r_squared_scaled', float('nan'))
        fit_mode  = getattr(result.fit, 'fit_mode', 'scaled')
        mode_lbl  = 'N·MSD' if fit_mode == 'scaled' else 'pure MSD'

        def _r2s(v): return f'{v:.4f}' if v == v else 'failed'
        mu_str = f'mu={mu:.4f} (95% CI: {lo:.4f}-{hi:.4f})' if mu == mu else ''
        regime_label = result.regime
        regime_str = f'  |  {regime_label}' if regime_label is not None else ''
        print(
            f'  {mu_str}'
            f'  |  R2(pure)={_r2s(r2_pure)}  R2(N·MSD)={_r2s(r2_scaled)}'
            f'  [{mode_lbl} selected]'
            f'{regime_str}'
        )

    fig = plot_burst_msd(
        result,
        title=stem,
        save_path=os.path.join(output_dir, f'{stem}_msd.png') if output_dir else None,
    )
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    row = _build_row(result, stem, model, path)

    return {
        'path':    path,
        'dataset': stem,
        'result':  result,
        'figure':  fig,
        'row':     row,
    }


# ── batch_analyze_bursts ──────────────────────────────────────────────────────

def batch_analyze_bursts(
    source: 'str | list',
    model: str = 'exponential',
    detrend_method: 'str | None' = 'linear',
    normalize: bool = False,
    output_dir: 'str | None' = None,
    n_jobs: int = 1,
    verbose: bool = True,
) -> 'pd.DataFrame':
    """
    Batch SWNA analysis on multiple RadioJOVE burst CSVs.

    Parameters
    ----------
    source : str or list of str
        Folder path — all ``*.csv`` files in the folder are analyzed.
        List of file paths — analyzed in the given order.
    model : str, default ``'exponential'``
        SWNA model applied to all files.
    detrend_method : str or None, default ``'linear'``
        Detrending applied before MSD computation.
    normalize : bool, default ``False``
        Z-score normalize values before fitting.
    output_dir : str, optional
        Directory for per-file MSD plots and a ``swna_summary.csv``.
        Created automatically if it does not exist.
    n_jobs : int, default 1
        Number of parallel workers (uses ``ThreadPoolExecutor`` when > 1).
    verbose : bool, default True
        Print per-file results as they complete.

    Returns
    -------
    pd.DataFrame
        One row per analyzed file with columns:
        ``dataset``, ``filename``, ``model``, ``n_points``,
        ``mu``, ``mu_ci_low``, ``mu_ci_high``, ``beta`` (or ``nu``/``H``),
        ``r_squared``.

    Examples
    --------
    >>> # Analyze all CSVs in a folder
    >>> df = wn.radiojove.batch_analyze_bursts(
    ...     'Solar/Type 3 Bursts/first trials/',
    ...     model='exponential',
    ...     output_dir='swna_results/',
    ... )
    >>> print(df[['dataset', 'mu']])

    >>> # Analyze a specific subset
    >>> selected = ['region_1_0.1s.csv', 'region_2_0.1s.csv']
    >>> df = wn.radiojove.batch_analyze_bursts(selected, model='exponential')
    """
    import pandas as pd
    from .io import list_burst_csvs

    if isinstance(source, str):
        paths = list_burst_csvs(source)
    else:
        paths = list(source)

    n = len(paths)
    print(f'\n{"=" * 60}')
    print(f'  RadioJOVE — Batch SWNA  ({model} model,  {n} file{"s" if n != 1 else ""})')
    print(f'{"=" * 60}')

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    def _run_one(p):
        return analyze_burst(
            p,
            model=model,
            detrend_method=detrend_method,
            normalize=normalize,
            output_dir=output_dir,
            verbose=verbose,
        )

    if n_jobs == 1:
        outputs = [_run_one(p) for p in paths]
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            outputs = list(pool.map(_run_one, paths))

    rows = [out['row'] for out in outputs if out['row'] is not None]
    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    succeeded = sum(1 for o in outputs if o['result'] is not None)
    print(f'\n  Done. {succeeded}/{n} files analyzed successfully.')

    if output_dir and not df.empty:
        summary_path = os.path.join(output_dir, 'swna_summary.csv')
        df.to_csv(summary_path, index=False)
        print(f'  Summary saved: {summary_path}')

    print(f'{"=" * 60}')

    return df


# ── plot_burst_msd ────────────────────────────────────────────────────────────

def plot_burst_msd(
    result,
    title: str = '',
    save_path: 'str | None' = None,
) -> 'matplotlib.figure.Figure':
    """
    Dual-panel MSD plot (linear scale + log-log scale) for a burst result.

    Parameters
    ----------
    result : AnalysisResult or None
        Output from ``wn.analyze()`` or ``wn.radiojove.analyze_burst()``.
        If ``None``, an empty figure with an error label is returned.
    title : str, optional
        Figure suptitle. Defaults to dataset name from *result*.
    save_path : str, optional
        If provided, saves the figure as PNG at this path (dpi=150).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> out = wn.radiojove.analyze_burst('burst.csv', verbose=False)
    >>> fig = wn.radiojove.plot_burst_msd(out['result'], title='Region 1 — 0.1s')
    >>> fig.show()

    >>> # Save to file
    >>> wn.radiojove.plot_burst_msd(out['result'], save_path='region1_msd.png')
    """
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4))

    if result is None or result.fit is None:
        for ax in (ax_lin, ax_log):
            ax.text(0.5, 0.5, 'Analysis failed', ha='center', va='center',
                    transform=ax.transAxes, color='grey', fontsize=11)
            ax.spines[['top', 'right']].set_visible(False)
        if title:
            fig.suptitle(title, fontsize=9)
        fig.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
        return fig

    model_name = result.model
    lags       = result.lags
    msd_emp    = result.msd_empirical

    # ── Left panel: linear scale ──────────────────────────────────────────────
    ax_lin.scatter(lags, msd_emp, s=8, color=_EMPIRICAL, alpha=0.7, label='Empirical MSD')
    ax_lin.plot(
        result.fit.lags_used, result.fit.msd_fitted,
        color=_THEORETICAL, lw=2.0, label=f'{model_name} fit',
    )
    ax_lin.set_xlabel('Time lag τ (s)', fontsize=10)
    ax_lin.set_ylabel('MSD', fontsize=10)
    ax_lin.set_title('MSD vs τ  (linear)', fontsize=10)
    ax_lin.legend(fontsize=8)
    ax_lin.spines[['top', 'right']].set_visible(False)

    # ── Right panel: log-log scale ────────────────────────────────────────────
    valid = (lags > 0) & (msd_emp > 0)
    ax_log.loglog(lags[valid], msd_emp[valid], '.', color=_EMPIRICAL,
                  markersize=4, alpha=0.7, label='Empirical MSD')

    fit_valid = (result.fit.lags_used > 0) & (result.fit.msd_fitted > 0)
    ax_log.loglog(
        result.fit.lags_used[fit_valid], result.fit.msd_fitted[fit_valid],
        color=_THEORETICAL, lw=2.0, label=f'{model_name} fit',
    )
    ax_log.set_xlabel('Time lag τ (s)', fontsize=10)
    ax_log.set_ylabel('MSD', fontsize=10)
    ax_log.set_title('MSD vs τ  (log-log)', fontsize=10)
    ax_log.legend(fontsize=8)
    ax_log.spines[['top', 'right']].set_visible(False)

    # ── Suptitle ──────────────────────────────────────────────────────────────
    mu  = result.fit.params.get('mu', float('nan'))
    r2  = result.fit.r_squared
    label_parts = [f'μ = {mu:.4f}', f'R² = {r2:.4f}']
    regime_label = result.regime
    if regime_label is not None:
        label_parts.append(regime_label)
    if title:
        label_parts.insert(0, title)
    fig.suptitle('  ·  '.join(label_parts), fontsize=9, y=1.02)

    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  Plot saved: {save_path}')

    return fig


# ── compare_bursts ────────────────────────────────────────────────────────────

def compare_bursts(
    paths: list,
    model: str = 'exponential',
    detrend_method: 'str | None' = 'linear',
    normalize: bool = False,
    max_lag_fraction: float = 1.0,
) -> 'ComparisonResult':
    """
    Compare multiple RadioJOVE burst CSVs with the same SWNA model.

    Wraps ``wn.compare()`` with RadioJOVE defaults and returns a
    :class:`~whitenoise.analysis.compare.ComparisonResult` that can be
    printed with ``wn.print_comparison()`` or plotted with
    ``wn.publish_comparison()``.

    Parameters
    ----------
    paths : list of str
        Paths to RadioJOVE CSV files to compare.
    model : str, default ``'exponential'``
        SWNA model applied to all files.
    detrend_method : str or None, default ``'linear'``
        Detrending method passed to ``wn.compare()``.
    normalize : bool, default ``False``
        Z-score normalize before fitting.
    max_lag_fraction : float, default 1.0
        Fraction of MSD lags to use for fitting (1.0 = all lags).

    Returns
    -------
    ComparisonResult
        Contains ``results`` (list of AnalysisResult), ``models_used``,
        and ``summary_df`` DataFrame.

    Example
    -------
    >>> paths = wn.radiojove.list_burst_csvs('Solar/Type 3 Bursts/first trials/')
    >>> groups = wn.radiojove.group_by_cadence(paths)
    >>> cr = wn.radiojove.compare_bursts(groups[0.1], model='exponential')
    >>> wn.print_comparison(cr)
    >>> wn.publish_comparison(cr)
    """
    import whitenoise as wn

    n = len(paths)
    print(f'Comparing {n} burst{"s" if n != 1 else ""} ({model} model)...')
    with contextlib.redirect_stdout(io.StringIO()):
        cr = wn.compare(
            paths,
            model=model,
            detrend_method=detrend_method,
            normalize=normalize,
            max_lag_fraction=max_lag_fraction,
        )

    succeeded = len(cr.results)
    print(f'Done. {succeeded}/{n} succeeded.')
    return cr


# ── Internal helpers ──────────────────────────────────────────────────────────

def _build_row(result, stem: str, model: str, path: str) -> 'dict | None':
    """Build a summary dict row from an AnalysisResult (or None)."""
    import math

    if result is None or result.fit is None:
        return {
            'dataset':    stem,
            'filename':   os.path.basename(path),
            'model':      model,
            'n_points':   None,
            'mu':         float('nan'),
            'mu_ci_low':  float('nan'),
            'mu_ci_high': float('nan'),
            'r_squared':  float('nan'),
            'regime':     None,
        }

    mu = result.fit.params.get('mu', float('nan'))
    lo, hi = result.fit.confidence_intervals.get('mu', (float('nan'), float('nan')))

    r2_pure   = getattr(result.fit, 'r_squared_pure',   float('nan'))
    r2_scaled = getattr(result.fit, 'r_squared_scaled', float('nan'))
    fit_mode  = getattr(result.fit, 'fit_mode', 'scaled')

    row: dict = {
        'dataset':          stem,
        'filename':         os.path.basename(path),
        'model':            model,
        'n_points':         len(result.values),
        'mu':               round(mu, 4),
        'mu_ci_low':        round(lo, 4),
        'mu_ci_high':       round(hi, 4),
        'r_squared':        round(result.fit.r_squared, 4),
        'r_squared_pure':   round(r2_pure,   4) if math.isfinite(r2_pure)   else float('nan'),
        'r_squared_scaled': round(r2_scaled, 4) if math.isfinite(r2_scaled) else float('nan'),
        'fit_mode':         fit_mode,
        # Model-family-specific classification (None if the model has no
        # confirmed scheme): cosine/sine -> alpha-based (Elnar et al. 2021),
        # exponential -> mu-based memory persistence (Tey et al. 2024).
        'regime':           result.regime,
    }

    # Second parameter varies by model
    if model in ('cosine', 'sine'):
        p2_name = 'nu'
    elif model == 'exponential':
        p2_name = 'beta'
    else:
        p2_name = 'H'

    p2_val = result.fit.params.get(p2_name, float('nan'))
    if not math.isnan(p2_val):
        row[p2_name] = round(p2_val, 6)

    return row
