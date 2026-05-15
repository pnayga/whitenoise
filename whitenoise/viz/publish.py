"""
viz/publish.py — Publication-quality comparison plot for whitenoise results.

publish_comparison()  μ bar chart with 95% CI error bars (unique to this module).

publish_msd() and publish_pdf() are aliases of plot_msd() / plot_pdf() from
viz/explore.py — calling either name produces the same single figure.
"""

from __future__ import annotations

import contextlib
import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.figure
import pandas as pd


# ── Style and palette constants ────────────────────────────────────────────────

STYLE: dict = {
    'figure.dpi':       150,
    'font.family':      'serif',
    'font.size':        11,
    'axes.linewidth':   0.8,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'xtick.top':        True,
    'ytick.right':      True,
    'legend.frameon':   False,
}

PALETTES: dict[str, dict[str, str]] = {
    'default': {
        'empirical': '#555555',
        'theory':    '#1B3A6B',
        'pdf':       '#C0392B',
    },
    'colorblind': {
        'empirical': '#999999',
        'theory':    '#0072B2',
        'pdf':       '#D55E00',
    },
}


@contextlib.contextmanager
def _style_ctx():
    with matplotlib.rc_context(STYLE):
        yield


def _resolve_palette(palette: str) -> dict[str, str]:
    if palette not in PALETTES:
        raise ValueError(
            f"✗ Unknown palette '{palette}'. "
            f"Available: {list(PALETTES.keys())}"
        )
    return PALETTES[palette]


def _save(fig: matplotlib.figure.Figure, save_path: str | None) -> None:
    if save_path is None:
        return
    dirn = os.path.dirname(save_path)
    if dirn:
        os.makedirs(dirn, exist_ok=True)
    ext = os.path.splitext(save_path)[1].lstrip('.').lower()
    fmt = ext if ext in {'pdf', 'png', 'svg', 'eps', 'jpg', 'jpeg'} else 'pdf'
    fig.savefig(save_path, format=fmt, bbox_inches='tight')
    print(f'✓ Saved to {save_path}')


# ── publish_comparison ────────────────────────────────────────────────────────

def publish_comparison(
    cr,
    palette:   str = 'default',
    figsize:   tuple | None = None,
    save_path: str | None = None,
    show:      bool = True,
) -> matplotlib.figure.Figure:
    """
    Publication-quality μ comparison bar chart with 95 % CI error bars.

    Parameters
    ----------
    cr : ComparisonResult
    palette : str, default ``'default'``
        ``'default'`` or ``'colorblind'``.
    figsize : tuple, optional
        Defaults to ``(7, height)`` where height scales with the number of datasets.
    save_path : str, optional
        If provided, save the figure (PDF, PNG, SVG, or EPS based on extension).
    show : bool, default ``True``
        If ``True``, call ``plt.show()`` after drawing.

    Returns
    -------
    matplotlib.figure.Figure
    """
    colors  = _resolve_palette(palette)
    df      = cr.summary_df.dropna(subset=['mu'])
    n_sys   = len(df)

    labels  = list(df['dataset_name'])
    mu_vals = np.array(df['mu'], dtype=float)

    # Parse CI strings like "(1.100, 1.300)" → half-width error bars
    xerr = np.zeros(n_sys)
    for i, ci_str in enumerate(df['mu_ci']):
        try:
            lo_str, hi_str = str(ci_str).strip('()').split(',')
            lo, hi = float(lo_str), float(hi_str)
            xerr[i] = (hi - lo) / 2.0
        except Exception:
            xerr[i] = 0.0

    if figsize is None:
        figsize = (7, max(3, 0.5 * n_sys + 1.5))

    with _style_ctx():
        fig, ax = plt.subplots(figsize=figsize)

        y = np.arange(n_sys)
        ax.barh(y, mu_vals, color=colors['theory'], alpha=0.75, zorder=3)
        ax.errorbar(
            mu_vals, y, xerr=xerr,
            fmt='none', color='black', linewidth=1.2, capsize=4, zorder=4,
        )

        ax.axvline(1.0, color='#888888', linewidth=0.8, linestyle='--',
                   label='μ = 1 (Brownian)')

        ax.set_yticks(y)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Memory parameter μ')
        ax.set_title('Memory Parameter Comparison')
        ax.legend()
        fig.tight_layout()

    _save(fig, save_path)
    if show:
        plt.show()
    return fig


# ── Backward-compat aliases ───────────────────────────────────────────────────
# publish_msd and publish_pdf were removed in v0.1 because they were
# identical to plot_msd / plot_pdf. Aliased here so existing code doesn't break.

def publish_msd(result, **kwargs) -> matplotlib.figure.Figure:
    """Alias for :func:`~whitenoise.viz.explore.plot_msd`. Use ``wn.plot_msd()``."""
    from .explore import plot_msd
    return plot_msd(result, **kwargs)


def publish_pdf(result, **kwargs) -> matplotlib.figure.Figure:
    """Alias for :func:`~whitenoise.viz.explore.plot_pdf`. Use ``wn.plot_pdf()``."""
    from .explore import plot_pdf
    return plot_pdf(result, **kwargs)
