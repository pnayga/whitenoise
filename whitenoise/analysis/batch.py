"""
analysis/batch.py — Batch and model-search utilities.

batch_analyze()      Run one model on many files (or a folder), optionally in parallel.
batch_model_search() Run many models on one file; report the best fit.
"""

from __future__ import annotations

import contextlib
import glob as _glob
import io
import os
import sys

from .pipeline import analyze, AnalysisResult
from .compare import ComparisonResult, _result_row, _nan_row, _DF_COLS

import pandas as pd

# Path to the package root so child processes can find the package.
_PKG_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


# ── Module-level worker (must be picklable for ProcessPoolExecutor) ────────────

def _analyze_job(args: tuple):
    """
    Worker function for parallel batch execution.

    Must be defined at module level so multiprocessing can pickle it.
    """
    label, source, model, detrend_method, normalize, max_lag_fraction, fit_kwargs = args
    # Ensure the package is importable in the child process
    if _PKG_ROOT not in sys.path:
        sys.path.insert(0, _PKG_ROOT)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            return analyze(
                source,
                model=model,
                label=label,
                detrend_method=detrend_method,
                normalize=normalize,
                max_lag_fraction=max_lag_fraction,
                fit_kwargs=fit_kwargs or {},
            )
    except Exception:
        return None


# ── batch_analyze ──────────────────────────────────────────────────────────────

def batch_analyze(
    source: 'str | list[str]',
    model: str = 'cosine',
    pattern: str = '*.csv',
    detrend_method: str | None = None,
    normalize: bool = False,
    max_lag_fraction: float = 1.0,
    fit_kwargs: dict | None = None,
    n_jobs: int = 1,
) -> ComparisonResult:
    """
    Run the same SWNA model on many datasets, optionally in parallel.

    Three input modes
    -----------------
    * **Folder path** (str, no ``.csv`` extension, or is a directory) —
      reads all files matching *pattern* in that folder.
    * **Multi-column CSV path** (str ending in ``.csv``) —
      each non-time column is analyzed as a separate system via
      :func:`~whitenoise.io.reader.read_csv_multi`.
    * **List of CSV paths** — explicit list; labels auto-set from filenames.

    Parameters
    ----------
    source : str or list[str]
        Folder path, multi-column CSV path, or list of CSV paths.
    model : str, default ``'cosine'``
        SWNA model name.
    pattern : str, default ``'*.csv'``
        Glob pattern for folder mode.
    detrend_method : str or None, default ``None``
        Detrending method.  ``None`` skips detrending.
        Choices: ``'linear'``, ``'polynomial'``, ``'mean'``,
        ``'moving_average'``.
    normalize : bool, default ``False``
        Whether to z-score normalize before fitting.
    max_lag_fraction : float, default 1.0
        Fraction of lags used in fitting.
    fit_kwargs : dict, optional
        Extra keyword arguments for :func:`~whitenoise.core.fitting.fit_msd`.
    n_jobs : int, default 1
        Number of parallel workers.  ``1`` = serial.  ``>1`` = parallel
        (uses :class:`concurrent.futures.ThreadPoolExecutor`).

    Returns
    -------
    ComparisonResult

    Examples
    --------
    >>> # Folder of CSVs
    >>> results = wn.batch_analyze('data/xray_binaries/', model='cosine')

    >>> # Explicit list
    >>> results = wn.batch_analyze(
    ...     ['chile.csv', 'japan.csv', 'ph.csv'],
    ...     model='exponential'
    ... )

    >>> # Multi-column CSV
    >>> results = wn.batch_analyze('climate.csv', model='cosine')
    """
    if fit_kwargs is None:
        fit_kwargs = {}

    # ── Resolve input into list of (label, source) pairs ──────────────────────
    items: list[tuple[str, object]]  # (label, source)

    if isinstance(source, list):
        # Mode B: explicit list of CSV paths
        items = [
            (os.path.splitext(os.path.basename(p))[0], p)
            for p in source
        ]

    elif isinstance(source, str) and source.lower().endswith('.csv') \
            and not os.path.isdir(source):
        # Mode C: single CSV — check if it is multi-column
        from ..io.reader import read_csv_multi
        multi = read_csv_multi(source)
        if len(multi) > 1:
            # Each tuple is (time, values, metadata); label from y_name
            items = [
                (meta.get('y_name', f'col{i}'), values)
                for i, (_, values, meta) in enumerate(multi)
            ]
        else:
            # Single-column CSV — treat as one dataset
            name = os.path.splitext(os.path.basename(source))[0]
            items = [(name, source)]

    else:
        # Mode A: folder path — glob for CSVs
        folder = source
        csv_paths = sorted(_glob.glob(os.path.join(folder, pattern)))
        if not csv_paths:
            print(f'\u2717 No files matching {pattern!r} found in {folder!r}.')
        items = [
            (os.path.splitext(os.path.basename(p))[0], p)
            for p in csv_paths
        ]

    n = len(items)
    print(f'\u2713 Batch analyzing {n} dataset{"s" if n != 1 else ""} '
          f'(model={model}, n_jobs={n_jobs})...')

    job_args = [
        (label, source_i, model, detrend_method, normalize,
         max_lag_fraction, fit_kwargs)
        for label, source_i in items
    ]

    # ── Run jobs ──────────────────────────────────────────────────────────────
    ar_list: list[AnalysisResult | None]

    if n_jobs == 1:
        ar_list = [_analyze_job(args) for args in job_args]
    else:
        # ThreadPoolExecutor avoids Windows spawn/pickle issues with non-installed
        # packages while still providing genuine concurrency for I/O-bound work.
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            ar_list = list(pool.map(_analyze_job, job_args))

    # ── Collect results and build summary ─────────────────────────────────────
    succeeded   = 0
    results:   list[AnalysisResult] = []
    rows:      list[dict]           = []

    for (label, _), ar in zip(items, ar_list):
        if ar is not None:
            r2_str = (
                f'{ar.fit.r_squared:.4f}' if ar.fit is not None else 'N/A'
            )
            print(f'  \u2713 {label}  (R\u00b2={r2_str})')
            results.append(ar)
            rows.append(_result_row(ar))
            succeeded += 1
        else:
            print(f'  \u2717 {label}  (failed)')
            rows.append(_nan_row(label, model))

    summary_df = pd.DataFrame(rows, columns=_DF_COLS)
    print(f'\u2713 Batch complete. {succeeded}/{n} succeeded.')

    return ComparisonResult(
        results=results,
        models_used=[model],
        summary_df=summary_df,
    )


# ── batch_model_search ────────────────────────────────────────────────────────

def batch_model_search(
    path: str,
    models: list[str] | None = None,
    detrend_method: str | None = None,
    normalize: bool = False,
    max_lag_fraction: float = 1.0,
    fit_kwargs: dict | None = None,
) -> ComparisonResult:
    """
    Fit every available model to a single CSV and report the best.

    Stub models (``status != 'available'``) and models that raise
    :exc:`ValueError` or :exc:`NotImplementedError` are silently skipped.

    Parameters
    ----------
    path : str
        Path to a whitenoise-format CSV file.
    models : list[str] or None
        Models to try.  ``None`` → all available models from the MODELS registry.
    detrend_method : str or None, default ``None``
        Detrending method.  ``None`` skips detrending.
        Choices: ``'linear'``, ``'polynomial'``, ``'mean'``,
        ``'moving_average'``.
    normalize : bool, default ``False``
        Whether to z-score normalize before fitting.
    max_lag_fraction : float, default 1.0
        Fraction of lags used in fitting.
    fit_kwargs : dict, optional
        Extra keyword arguments for :func:`~whitenoise.core.fitting.fit_msd`.

    Returns
    -------
    ComparisonResult
        One row per successfully tried model, sorted by R² descending.
    """
    from ..core.models import MODELS

    if fit_kwargs is None:
        fit_kwargs = {}

    if models is None:
        models = [n for n, v in MODELS.items() if v['status'] == 'available']

    dataset_name = os.path.splitext(os.path.basename(path))[0]
    print(f'\u2713 Model search on \'{dataset_name}\' '
          f'across {len(models)} model(s)...')

    results: list[AnalysisResult] = []
    rows: list[dict] = []

    for m in models:
        # Silently skip stub or unknown models without adding a row
        if MODELS.get(m, {}).get('status') != 'available':
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ar = analyze(
                    path,
                    model=m,
                    detrend_method=detrend_method,
                    normalize=normalize,
                    max_lag_fraction=max_lag_fraction,
                    fit_kwargs=fit_kwargs,
                )
            r2 = ar.fit.r_squared if ar.fit is not None else float('nan')
            r2_str = f'{r2:.4f}' if ar.fit is not None else 'N/A'
            print(f'  {m} \u2192 R\u00b2={r2_str}')
            results.append(ar)
            rows.append(_result_row(ar))
        except (ValueError, NotImplementedError):
            # Silently skip stub / invalid models
            pass
        except Exception as exc:
            print(f'  {m} \u2192 ERROR: {exc}')
            rows.append(_nan_row(dataset_name, m))

    summary_df = pd.DataFrame(rows, columns=_DF_COLS)

    # Report best model
    valid = summary_df.dropna(subset=['r_squared'])
    if not valid.empty:
        best_idx  = valid['r_squared'].idxmax()
        best_name = valid.loc[best_idx, 'model']
        best_r2   = valid.loc[best_idx, 'r_squared']
        print(f'  Best model: {best_name}  (R\u00b2={best_r2:.4f})')

    return ComparisonResult(
        results=results,
        models_used=list(summary_df['model']) if not summary_df.empty else [],
        summary_df=summary_df,
    )
