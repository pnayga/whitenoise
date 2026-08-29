"""
whitenoise.exoplanet.pipeline
==============================
Step-by-step and one-shot SWNA pipeline for TESS transit windows.

:func:`analyze_transit` runs the full chain — window extraction, optional
CSV export, SWNA fit, and diagnostic plot — for a single transit.
:func:`batch_analyze_transits` runs it over a whole
``pipeline/<Planet>/<Planet>_T##.csv`` folder tree.

Typical workflow::

    import whitenoise as wn

    # Single transit, from full-sector time/flux + a known midpoint
    T14_d = wn.exoplanet.estimate_T14(P_days=2.175, Rs=2.350, Rp=2.060, a_AU=0.03670)
    t_mid, depth, snr, method = wn.exoplanet.find_empirical_midpoint(
        time, flux, t_lo, t_hi, T14_d,
    )
    out = wn.exoplanet.analyze_transit(
        time, flux, t_mid, T14_d,
        model='exponential',
        output_dir='results/swna/exponential/WASP-078/',
    )
    out['figure'].show()
    print(out['result'].fit.params.get('mu'))

    # Batch — a full pipeline/ tree
    df = wn.exoplanet.batch_analyze_transits(
        'data/selected_transits/pipeline/',
        models=('cosine', 'exponential'),
        output_dir='results/swna/',
    )
    print(df[['planet', 'transit', 'model', 'mu', 'r_squared']])

Model guidance
--------------
Run ``wn.list_models()`` to see all available models. The exoplanet
transit-window studies this module was built for use the ``exponential``
model for the primary analysis (see ``result.regime`` for the
memory-persistence classification); ``cosine`` is also computed for
comparison in the batch driver but is known to fit poorly on transit MSD
curves (it cannot follow the saturation at high lags).
"""

from __future__ import annotations

import math
import os

import numpy as np

from .io import write_pipeline_csv, list_transit_csvs
from .preprocess import extract_transit_window


# ── analyze_transit ────────────────────────────────────────────────────────────

def analyze_transit(
    time: 'np.ndarray',
    flux: 'np.ndarray',
    t_mid: float,
    T14_d: float,
    model: str = 'exponential',
    detrend_method: 'str | None' = None,
    window_periods: float = 1.5,
    cadence_min: float = 2.0,
    max_fill_gap_min: float = 60.0,
    csv_path: 'str | None' = None,
    output_dir: 'str | None' = None,
    dataset_name: str = 'transit',
    verbose: bool = True,
) -> dict:
    """
    Full SWNA pipeline for a single TESS transit.

    Extracts the transit window (:func:`~whitenoise.exoplanet.preprocess.extract_transit_window`),
    optionally writes it to a pipeline CSV, runs ``wn.analyze()``, and
    generates a diagnostic plot.

    Parameters
    ----------
    time, flux : array-like
        Full-sector light curve arrays, e.g. from
        :func:`~whitenoise.exoplanet.io.download_transit_lc` or
        :func:`~whitenoise.exoplanet.io.clean_lc`.
    t_mid : float
        Transit midpoint (days), e.g. from
        :func:`~whitenoise.exoplanet.preprocess.find_empirical_midpoint`.
    T14_d : float
        Total transit duration (days).
    model : str, default ``'exponential'``
        SWNA model. Run ``wn.list_models()`` to see all options.
    detrend_method : str or None, default ``None``
        Detrending applied before MSD computation. Transit-window studies
        typically use ``None`` (no detrending).
    window_periods : float, default 1.5
        Half-window width in units of T14 (``1.5`` gives the "3T" window).
    cadence_min : float, default 2.0
        Expected observing cadence in minutes.
    max_fill_gap_min : float, default 60.0
        Only gaps narrower than this (minutes) are NN-filled.
    csv_path : str, optional
        If provided, the extracted window is written here via
        :func:`~whitenoise.exoplanet.io.write_pipeline_csv` before analysis.
    output_dir : str, optional
        If provided, the diagnostic plot PNG is saved here as
        ``<dataset_name>_diagnostics.png``.
    dataset_name : str, default ``'transit'``
        Label used for the saved PNG filename and printed output.
    verbose : bool, default True
        Print window/fit status.

    Returns
    -------
    dict with keys:

        ``'window'``  — dict from :func:`extract_transit_window`, or ``None``
                        if the window failed quality gates
        ``'result'``  — :class:`~whitenoise.analysis.pipeline.AnalysisResult`,
                        or ``None`` if extraction or fitting failed
        ``'figure'``  — :class:`matplotlib.figure.Figure`, or ``None``
        ``'row'``     — dict summary row for building DataFrames, or ``None``

    Examples
    --------
    >>> out = wn.exoplanet.analyze_transit(time, flux, t_mid=1234.6, T14_d=0.0623)
    >>> print(out['result'].fit.params.get('mu'))
    """
    import whitenoise as wn

    if verbose:
        print(f'\n  {dataset_name}')
        print(f'  {"-" * 56}')

    window = extract_transit_window(
        time, flux, t_mid, T14_d,
        window_periods=window_periods,
        cadence_min=cadence_min,
        max_fill_gap_min=max_fill_gap_min,
    )

    if window is None:
        if verbose:
            print('  [FAIL] transit window rejected (out of bounds, too few points, or low coverage)')
        return {'window': None, 'result': None, 'figure': None, 'row': None}

    if csv_path is not None:
        write_pipeline_csv(window['time'], window['flux'], t_mid, csv_path)
        analyze_path = csv_path
    else:
        import tempfile
        fd, analyze_path = tempfile.mkstemp(suffix='.csv')
        os.close(fd)
        write_pipeline_csv(window['time'], window['flux'], t_mid, analyze_path)

    result = None
    try:
        result = wn.analyze(analyze_path, model=model, detrend_method=detrend_method)
    except Exception as exc:
        if verbose:
            print(f'  [FAIL] {dataset_name}: {exc}')
    finally:
        if csv_path is None and os.path.exists(analyze_path):
            os.remove(analyze_path)

    fig = None
    if result is not None:
        fig = wn.plot_diagnostics(result, show=False)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            png_path = os.path.join(output_dir, f'{dataset_name}_diagnostics.png')
            fig.savefig(png_path, dpi=150, bbox_inches='tight')

    row = _build_row(result, window, dataset_name, model)

    if verbose and result is not None and result.fit is not None:
        mu = row.get('mu', float('nan'))
        r2 = row.get('r_squared', float('nan'))
        regime = row.get('regime')
        print(f'  [OK]  mu={mu:.4f}  R2={r2:.4f}  [{regime}]  ({window["n_filled"]} pts NN-filled)')

    return {'window': window, 'result': result, 'figure': fig, 'row': row}


# ── batch_analyze_transits ────────────────────────────────────────────────────

def batch_analyze_transits(
    pipeline_dir: str,
    planets: 'list | None' = None,
    models: tuple = ('cosine', 'exponential'),
    detrend_method: 'str | None' = None,
    output_dir: 'str | None' = None,
    verbose: bool = True,
) -> 'pd.DataFrame':
    """
    Batch SWNA analysis over a ``pipeline/<Planet>/<Planet>_T##.csv`` tree.

    Expects each *pipeline_dir*/<planet>/ subfolder to contain 2-column
    (``time_min_from_mid,flux``) transit CSVs, e.g. as written by
    :func:`~whitenoise.exoplanet.io.write_pipeline_csv`.

    Parameters
    ----------
    pipeline_dir : str
        Root folder containing one subfolder per planet.
    planets : list of str, optional
        Planet subfolder names to process. Defaults to every subfolder
        found in *pipeline_dir*.
    models : tuple of str, default ``('cosine', 'exponential')``
        SWNA models to run on every transit CSV.
    detrend_method : str or None, default ``None``
        Detrending applied before MSD computation.
    output_dir : str, optional
        Directory for per-model/per-planet diagnostic plots and a
        ``swna_summary.csv`` / ``swna_summary.xlsx``. Created automatically.
    verbose : bool, default True
        Print per-transit results as they complete.

    Returns
    -------
    pd.DataFrame
        One row per (transit, model) with columns ``planet``, ``transit``,
        ``model``, ``r_squared``, ``regime``, plus fitted parameters
        (``mu``, ``beta``/``nu``/``H`` depending on model).

    Example
    -------
    >>> df = wn.exoplanet.batch_analyze_transits(
    ...     'data/selected_transits/pipeline/',
    ...     models=('cosine', 'exponential'),
    ...     output_dir='results/swna/',
    ... )
    >>> print(df[['planet', 'transit', 'model', 'mu', 'r_squared']])
    """
    import whitenoise as wn
    import pandas as pd

    if not os.path.isdir(pipeline_dir):
        raise FileNotFoundError(f'✗ Directory not found: {pipeline_dir}')

    if planets is None:
        planets = sorted(
            d for d in os.listdir(pipeline_dir)
            if os.path.isdir(os.path.join(pipeline_dir, d))
        )

    summary_rows = []

    for model in models:
        if verbose:
            print(f'=== Model: {model.upper()} ===')

        for planet in planets:
            planet_dir = os.path.join(pipeline_dir, planet)
            if not os.path.isdir(planet_dir):
                if verbose:
                    print(f'  [SKIP] {planet} -- folder not found')
                continue

            csv_files = list_transit_csvs(planet_dir)
            if not csv_files:
                if verbose:
                    print(f'  [SKIP] {planet} -- no CSVs found')
                continue

            if verbose:
                print(f'\n  {planet} ({len(csv_files)} transit(s))')

            out_dir = os.path.join(output_dir, model, planet) if output_dir else None
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            for csv_path in csv_files:
                transit_name = os.path.splitext(os.path.basename(csv_path))[0]

                try:
                    result = wn.analyze(csv_path, model=model, detrend_method=detrend_method)
                except Exception as exc:
                    if verbose:
                        print(f'    [FAIL] {transit_name}: {exc}')
                    continue

                if result.fit is None:
                    if verbose:
                        print(f'    [FAIL] {transit_name}: fit returned None')
                    continue

                fig = wn.plot_diagnostics(result, show=False)
                if out_dir:
                    png_path = os.path.join(out_dir, f'{transit_name}_diagnostics.png')
                    fig.savefig(png_path, dpi=150, bbox_inches='tight')
                import matplotlib.pyplot as plt
                plt.close(fig)

                row = {'planet': planet, 'transit': transit_name, 'model': model}
                row.update(_fit_row(result))
                summary_rows.append(row)

                if verbose:
                    mu = row.get('mu', float('nan'))
                    print(f'    [OK]  {transit_name:<22}  mu={mu:.4f}  R2={row["r_squared"]:.4f}  [{row["regime"]}]')

        if verbose:
            print()

    df = pd.DataFrame(summary_rows) if summary_rows else pd.DataFrame()

    if output_dir and not df.empty:
        id_cols = ['planet', 'transit', 'model', 'r_squared', 'regime']
        param_cols = [c for c in df.columns if c not in id_cols]
        df = df[id_cols + param_cols]

        os.makedirs(output_dir, exist_ok=True)
        csv_out = os.path.join(output_dir, 'swna_summary.csv')
        df.to_csv(csv_out, index=False)
        if verbose:
            print(f'Summary CSV  --> {csv_out}')

        try:
            xlsx_out = os.path.join(output_dir, 'swna_summary.xlsx')
            df.to_excel(xlsx_out, index=False, sheet_name='SWNA Results')
            if verbose:
                print(f'Summary XLSX --> {xlsx_out}')
        except ImportError:
            if verbose:
                print('[WARN] openpyxl not installed -- Excel skipped.')

    return df


# ── Internal helpers ──────────────────────────────────────────────────────────

def _fit_row(result) -> dict:
    """Build a dict of fit params/metrics from an AnalysisResult."""
    params = result.fit.params
    row = {
        'r_squared': result.fit.r_squared,
        'regime': result.regime,
    }
    row.update(params)
    return row


def _build_row(result, window, dataset_name: str, model: str) -> 'dict | None':
    """Build a summary dict row from an AnalysisResult (or None)."""
    if result is None or result.fit is None:
        return {
            'dataset': dataset_name,
            'model': model,
            'mu': float('nan'),
            'r_squared': float('nan'),
            'regime': None,
        }

    row: dict = {
        'dataset': dataset_name,
        'model': model,
        'n_points': len(result.values),
        'r_squared': round(result.fit.r_squared, 4),
        'regime': result.regime,
    }
    if window is not None:
        row['n_filled'] = window['n_filled']
        row['coverage'] = round(window['coverage'], 4)

    for key, val in result.fit.params.items():
        if isinstance(val, float) and math.isfinite(val):
            row[key] = round(val, 6)
        else:
            row[key] = val

    return row
