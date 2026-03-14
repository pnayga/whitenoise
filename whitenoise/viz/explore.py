"""
viz/explore.py — Exploratory (interactive) plots for whitenoise analysis results.

All public functions accept an optional ax parameter and return a
matplotlib Figure object.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fd_bins(data: np.ndarray) -> int:
    """Freedman-Diaconis bin count for displacement histogram."""
    n = len(data)
    if n < 2:
        return 10
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return max(10, int(np.sqrt(n)))
    h = 2.0 * iqr * n ** (-1.0 / 3.0)
    data_range = np.max(data) - np.min(data)
    if h == 0 or data_range == 0:
        return 10
    return max(5, int(np.ceil(data_range / h)))


def _unit_label(name: str, unit: str) -> str:
    """Build 'name (unit)' or just 'name' if unit is empty/unitless."""
    if not unit or unit.lower() == 'unitless':
        return name
    return f'{name} ({unit})'


def _get_axes(ax, figsize=(7, 4.5)):
    """Return (fig, ax) — create new figure if ax is None."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.get_figure(), ax


# ── plot_series ───────────────────────────────────────────────────────────────

def plot_series(x, values, metadata=None, title='', ax=None, show=True) -> matplotlib.figure.Figure:
    """
    Plot raw sequential data directly from :func:`~whitenoise.io.reader.read_csv`.

    Use this for a quick visual inspection of your data *before* running
    :func:`~whitenoise.analysis.pipeline.analyze`.  No analysis needed.

    Parameters
    ----------
    x : array-like (1D)
        Independent variable (time, index, frequency, etc.).
    values : array-like (1D)
        Observable values.
    metadata : dict, optional
        The ``metadata`` dict returned by :func:`~whitenoise.io.reader.read_csv`.
        When provided, axis labels and title are set automatically from
        ``x_label`` and ``y_label``.
    title : str, optional
        Custom plot title.  Overrides the auto-title from metadata.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Creates a new figure if ``None``.
    show : bool, default ``True``
        If ``True``, call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> time, values, meta = wn.read_csv('sunspot.csv')
    >>> wn.plot_series(time, values, metadata=meta)

    >>> # After manual detrending — still works the same way
    >>> fluct = wn.detrend(values, method='moving_average', window=11)
    >>> wn.plot_series(time, fluct, metadata=meta, title='Sunspot fluctuations')
    """
    fig, ax = _get_axes(ax, figsize=(10, 3.5))

    x      = np.asarray(x,      dtype=float)
    values = np.asarray(values, dtype=float)

    # Auto-labels from metadata (x_label / y_label set by reader.py)
    if metadata is not None:
        xlabel = metadata.get('x_label', metadata.get('x_name', 'x'))
        ylabel = metadata.get('y_label', metadata.get('y_name', 'value'))
        auto_title = metadata.get('y_name', '')
    else:
        xlabel, ylabel, auto_title = 'x', 'value', ''

    ax.plot(x, values, color='#1B6CA8', linewidth=0.8, alpha=0.9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title if title else auto_title)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── plot_msd_empirical ────────────────────────────────────────────────────────

def plot_msd_empirical(lags, msd, metadata=None, title='', ax=None, show=True) -> matplotlib.figure.Figure:
    """
    Plot empirical MSD scatter directly from :func:`~whitenoise.core.msd.compute_msd`.

    Use this to inspect the shape of the MSD *before* fitting — useful for
    deciding which model to try or checking that the MSD looks reasonable.

    Parameters
    ----------
    lags : array-like (1D)
        Lag array returned by :func:`~whitenoise.core.msd.compute_msd`.
    msd : array-like (1D)
        MSD values returned by :func:`~whitenoise.core.msd.compute_msd`.
    metadata : dict, optional
        The ``metadata`` dict from :func:`~whitenoise.io.reader.read_csv`.
        Used to label the x-axis with the correct unit (e.g. 'Lag (months)').
    title : str, optional
        Custom plot title.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Creates a new figure if ``None``.
    show : bool, default ``True``
        If ``True``, call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> time, values, meta = wn.read_csv('sunspot.csv')
    >>> lags, msd = wn.compute_msd(values)
    >>> wn.plot_msd_empirical(lags, msd, metadata=meta)
    """
    fig, ax = _get_axes(ax, figsize=(7, 4.5))

    lags = np.asarray(lags, dtype=float)
    msd  = np.asarray(msd,  dtype=float)

    # x-axis label — use lag unit from metadata if available
    if metadata is not None:
        x_unit = metadata.get('x_unit', '')
        xlabel = f'Lag ({x_unit})' if x_unit else 'Lag'
    else:
        xlabel = 'Lag'

    finite = np.isfinite(msd)
    ax.scatter(lags[finite], msd[finite], s=12, color='#888888', alpha=0.7,
               label='Empirical MSD', zorder=3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel('MSD')
    ax.set_title(title if title else 'Empirical MSD')
    ax.legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── plot_msd ──────────────────────────────────────────────────────────────────

def plot_msd(result, ax=None, show=True) -> matplotlib.figure.Figure:
    """
    Plot empirical MSD (scatter) and fitted theoretical MSD (line).

    Parameters
    ----------
    result : AnalysisResult
        Output of :func:`~whitenoise.analysis.pipeline.analyze`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Creates a new figure if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_axes(ax, figsize=(7, 4.5))

    meta      = result.metadata
    x_unit    = meta.get('x_unit', '')
    obs_unit  = meta.get('y_unit', '')
    dname     = result.dataset_name

    xlabel = f'Lag ({x_unit})' if x_unit else 'Lag'
    ylabel = f'MSD ({obs_unit}\u00b2)' if obs_unit else 'MSD'

    # Empirical scatter
    ax.scatter(
        result.lags, result.msd_empirical,
        s=12, color='#888888', alpha=0.7, label='Empirical MSD', zorder=3,
    )

    # Theoretical line (if fit succeeded)
    if result.fit is not None:
        r2    = result.fit.r_squared
        label = f'Fitted {result.model} (R\u00b2={r2:.4f})'
        lags_used  = result.fit.lags_used
        msd_fitted = result.fit.msd_fitted
        finite = np.isfinite(msd_fitted)
        ax.plot(
            lags_used[finite], msd_fitted[finite],
            color='#1B6CA8', linewidth=2.0, label=label, zorder=4,
        )
    else:
        ax.annotate(
            'Fit not available',
            xy=(0.5, 0.85), xycoords='axes fraction',
            ha='center', fontsize=9, color='#C0392B',
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{dname} \u2014 MSD')
    ax.legend(loc='lower right', fontsize=9)

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── plot_pdf ──────────────────────────────────────────────────────────────────

def plot_pdf(result, lag_index=None, ax=None, show=True) -> matplotlib.figure.Figure:
    """
    Plot empirical displacement histogram vs theoretical Gaussian PDF.

    Parameters
    ----------
    result : AnalysisResult
        Output of :func:`~whitenoise.analysis.pipeline.analyze`.
    lag_index : int, optional
        Index into ``result.lags`` selecting lag T.
        Defaults to ``len(result.lags) // 4``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on.  Creates a new figure if ``None``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_axes(ax, figsize=(6, 4.5))

    meta     = result.metadata
    obs_unit = meta.get('y_unit', '')
    dname    = result.dataset_name

    lags   = result.lags
    values = result.values

    if lag_index is None:
        lag_index = max(0, len(lags) // 4)
    lag_index = min(lag_index, len(lags) - 1)
    T = float(lags[lag_index])
    lag_int = max(1, int(round(T)))

    xlabel = f'\u0394x ({obs_unit})' if obs_unit else '\u0394x'

    # Empirical displacements
    if lag_int < len(values):
        displacements = values[lag_int:] - values[:-lag_int]
        displacements = displacements[np.isfinite(displacements)]
    else:
        displacements = np.array([])

    if len(displacements) > 1:
        bins = _fd_bins(displacements)
        ax.hist(
            displacements, bins=bins, density=True,
            color='#888888', alpha=0.55, label='Empirical displacements',
        )

        # Theoretical Gaussian PDF (if fit is available)
        if result.fit is not None:
            from ..core.models import MODELS
            info   = MODELS.get(result.model, {})
            msd_fn = info.get('msd')
            params = result.fit.params

            if msd_fn is not None:
                phys_names = info.get('params', [])
                phys_vals  = [params[n] for n in phys_names if n in params]
                try:
                    sigma2 = float(msd_fn(T, *phys_vals)) * params.get('N', 1.0)
                    if np.isfinite(sigma2) and sigma2 > 0:
                        dx_range = np.linspace(displacements.min(), displacements.max(), 500)
                        pdf_vals = (
                            np.exp(-dx_range ** 2 / (2.0 * sigma2))
                            / np.sqrt(2.0 * np.pi * sigma2)
                        )
                        ax.plot(
                            dx_range, pdf_vals,
                            color='#C0392B', linewidth=2.0,
                            label=f'PDF (T={T:.3f})',
                        )
                except Exception:
                    pass

        ax.legend(loc='upper right', fontsize=9)
    else:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center', fontsize=10, color='#888888')

    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability density')
    ax.set_title(f'{dname} \u2014 PDF at lag T={T:.3f}')

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── plot_timeseries ───────────────────────────────────────────────────────────

def plot_timeseries(result, ax=None, show=True) -> matplotlib.figure.Figure:
    """
    Plot the preprocessed observable time series.

    Parameters
    ----------
    result : AnalysisResult
    ax : matplotlib.axes.Axes, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = _get_axes(ax, figsize=(9, 3.5))

    meta      = result.metadata
    x_unit    = meta.get('x_unit', '')
    obs_unit  = meta.get('y_unit', '')
    obs_name  = meta.get('y_name', 'value')
    dname     = result.dataset_name

    xlabel = f'{meta.get("x_name", "x")} ({x_unit})' if x_unit else meta.get('x_name', 'x')
    ylabel = _unit_label(obs_name, obs_unit)

    ax.plot(result.time, result.values, color='#1B6CA8', linewidth=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{dname} \u2014 Time Series')

    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ── plot_diagnostics ──────────────────────────────────────────────────────────

def plot_diagnostics(result, show=True) -> matplotlib.figure.Figure:
    """
    2×2 diagnostic figure: time series, MSD, PDF, and parameter summary.

    Parameters
    ----------
    result : AnalysisResult

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_ts, ax_msd, ax_pdf, ax_txt = axes.flat

    # Top-left: time series
    plot_timeseries(result, ax=ax_ts, show=False)

    # Top-right: MSD
    plot_msd(result, ax=ax_msd, show=False)

    # Bottom-left: PDF
    plot_pdf(result, ax=ax_pdf, show=False)

    # Bottom-right: parameter summary text box
    ax_txt.axis('off')
    lines = [
        f'Dataset : {result.dataset_name}',
        f'Model   : {result.model}',
        f'Points  : {len(result.values)}',
        f'Lags    : {len(result.lags)}',
        '',
    ]
    if result.fit is not None:
        lines.append(f'R\u00b2      : {result.fit.r_squared:.4f}')
        lines.append(f'Regime  : {result.regime}')
        lines.append('')
        for pname, pval in result.fit.params.items():
            se = result.fit.std_errors.get(pname, float('nan'))
            lines.append(f'{pname:<6} = {pval:.4f} \u00b1 {se:.4f}')
    else:
        lines.append('Fit     : N/A (fitting failed)')

    summary_text = '\n'.join(lines)
    ax_txt.text(
        0.05, 0.95, summary_text,
        transform=ax_txt.transAxes,
        va='top', ha='left', fontsize=9,
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='#F4F6F7', alpha=0.8),
    )

    fig.suptitle(
        f'SWNA Diagnostics \u2014 {result.dataset_name}',
        fontsize=13, fontweight='bold',
    )
    fig.tight_layout()
    if show:
        plt.show()
    return fig
