"""
whitenoise.genomics.pipeline
=============================
Step-by-step single-pair SWNA pipeline for genomic DNA sequences.

Each function accepts either raw arrays or a saved CSV path as input,
allowing the researcher to checkpoint and resume at any step without
recomputing expensive intermediate results.

Typical workflow
----------------
::

    import whitenoise as wn

    # Load and filter
    record = wn.genomics.read_fasta_single('chromosome.fasta')
    seq = wn.genomics.filter_sequence(record['sequence'])

    # Step by step, with checkpoints
    dist         = wn.genomics.extract_pair(seq, 'A', 'A', save_path='AA_distances.csv')
    lags, msd    = wn.genomics.compute_pair_msd(dist, max_lag=1500, save_path='AA_msd.csv')
    fit          = wn.genomics.fit_pair((lags, msd), model='dna')
    fig          = wn.genomics.plot_pair(lags, msd, fit, title='A→A | DNA model')

    # Later — refit from saved MSD without recomputing
    fit, fig = wn.genomics.refit_pair('AA_msd.csv', model='dna', p0=[28.0, 0.025, 3.0])
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .distances import get_transition_distances
from .io import read_fasta_single

# ── Palette (consistent with publish.py) ──────────────────────────────────────
_EMPIRICAL   = '#2C3E50'
_THEORETICAL = '#E74C3C'
_SUBTLE      = '#BDC3C7'


# ── 1. extract_pair ────────────────────────────────────────────────────────────

def extract_pair(
    sequence: str,
    from_nuc: str,
    to_nuc: str,
    save_path: str | None = None,
) -> np.ndarray:
    """
    Extract transition distances for one nucleotide pair.

    Computes the array of separation distances from each occurrence of
    *from_nuc* to the next occurrence of *to_nuc* along *sequence*.
    For same-type pairs (A→A) this gives consecutive distances between
    identical nucleotides, matching the Violanda et al. (2019) definition.

    Parameters
    ----------
    sequence : str
        Filtered ACGT sequence (use :func:`~whitenoise.genomics.io.filter_sequence`
        to remove ambiguous bases before calling this).
    from_nuc : str
        Source nucleotide — one of ``'A'``, ``'C'``, ``'G'``, ``'T'``.
    to_nuc : str
        Target nucleotide — one of ``'A'``, ``'C'``, ``'G'``, ``'T'``.
    save_path : str, optional
        If provided, saves the distance array to a whitenoise-format CSV
        at this path.  The file can be reloaded with ``wn.read_csv()``.

    Returns
    -------
    np.ndarray (1-D, int64)
        Array of separation distances.

    Examples
    --------
    >>> seq = wn.genomics.filter_sequence(record['sequence'])
    >>> dist = wn.genomics.extract_pair(seq, 'A', 'A', save_path='AA_dist.csv')
    >>> print(len(dist), dist.mean())

    >>> # Reload distances later
    >>> _, dist, _ = wn.read_csv('AA_dist.csv')
    """
    import whitenoise as wn

    distances = get_transition_distances(sequence, from_nuc, to_nuc)

    if len(distances) == 0:
        print(f'  {from_nuc}→{to_nuc}: no distances found')
        return distances

    print(
        f'  ✓ {from_nuc}→{to_nuc}: {len(distances):,} distances  '
        f'(mean={distances.mean():.2f}, median={float(np.median(distances)):.2f})'
    )

    if save_path is not None:
        wn.export_series(
            distances, save_path,
            label=f'dist_{from_nuc}{to_nuc}', unit='bp',
        )

    return distances


# ── 2. compute_pair_msd ────────────────────────────────────────────────────────

def compute_pair_msd(
    source,
    max_lag: int = 1500,
    save_path: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical MSD from a distances array or a saved CSV.

    Parameters
    ----------
    source : np.ndarray or str
        * ``np.ndarray`` — distances array (output of :func:`extract_pair`
          or :func:`~whitenoise.genomics.distances.get_transition_distances`).
        * ``str`` — path to a whitenoise-format CSV previously saved by
          :func:`extract_pair` (or :func:`~whitenoise.io.export.export_series`).
          The values column is used as the distances array.
    max_lag : int, default 1500
        Maximum lag to compute.  For the DNA model, the curve saturates at
        approximately ``L ≈ 3 / b``.  With the published bacterial genome
        value ``b ≈ 0.0024``, saturation occurs near L = 1250, so 1500 is
        a safe default.  For organisms with different ``b``, adjust accordingly.
    save_path : str, optional
        If provided, saves the (lags, msd) arrays to a whitenoise-format CSV.
        Reload with ``wn.read_csv()`` and pass to :func:`fit_pair`.

    Returns
    -------
    lags : np.ndarray (1-D, int)
    msd  : np.ndarray (1-D, float)

    Examples
    --------
    >>> lags, msd = wn.genomics.compute_pair_msd(dist, max_lag=1500, save_path='AA_msd.csv')

    >>> # From a saved distances CSV
    >>> lags, msd = wn.genomics.compute_pair_msd('AA_dist.csv', max_lag=1500)
    """
    import whitenoise as wn

    if isinstance(source, str):
        _, distances, _ = wn.read_csv(source)
    else:
        distances = np.asarray(source, dtype=float)

    lags, msd = wn.compute_msd(distances, max_lag=max_lag)

    print(
        f'  ✓ MSD computed: {len(lags)} lags,  '
        f'MSD[1]={msd[0]:.4f},  MSD[-1]={msd[-1]:.4f}'
    )

    if save_path is not None:
        wn.export_msd(lags, msd, save_path)

    return lags, msd


# ── 3. fit_pair ────────────────────────────────────────────────────────────────

def fit_pair(
    source,
    msd: np.ndarray | None = None,
    model: str = 'dna',
    p0: list | None = None,
    bounds: tuple | None = None,
    max_lag_fraction: float = 1.0,
    verbose: bool = True,
):
    """
    Fit an SWNA model to a MSD curve from arrays or a saved CSV.

    Parameters
    ----------
    source : (lags, msd) tuple, str, or np.ndarray
        * ``(lags, msd)`` tuple — arrays used directly.
        * ``str`` — path to a whitenoise-format MSD CSV (saved by
          :func:`compute_pair_msd`).  Column 1 = lags, column 2 = msd.
        * ``np.ndarray`` — lags array; supply *msd* as the second argument.
    msd : np.ndarray, optional
        MSD array.  Only used when *source* is a lags ``np.ndarray``.
    model : str, default ``'dna'``
        SWNA model name.  Run ``wn.list_models()`` to see all options.
        Use ``'dna'`` for same-type diagonal pairs (A→A, C→C, G→G, T→T).
        For off-diagonal cross-type pairs, try ``'exponential'`` or ``'cosine'``.
    p0 : list, optional
        Initial parameter guess (physical params only; N is appended internally).
        For the DNA model: ``[a, b, c]``.
        Default values often work for bacterial genomes but may need tuning
        for larger eukaryotic chromosomes.  Read the plateau and rise from
        your MSD plot to estimate starting values:

        * ``a`` ≈ plateau MSD value
        * ``c`` ≈ ``a − MSD(L=1)``
        * ``b`` ≈ ``6 / L_plateau``  (where ``L_plateau`` is the lag where the
          curve visually flattens)
    bounds : tuple, optional
        ``(lower_bounds, upper_bounds)`` for physical parameters only.
    max_lag_fraction : float, default 1.0
        Fraction of the lag range to use for fitting.
        Use ``< 1.0`` to exclude the noisy flat tail.
    verbose : bool, default True
        If ``True``, prints ``fit.summary()`` on success.

    Returns
    -------
    FitResult or None
        ``None`` if fitting fails.

    Examples
    --------
    >>> fit = wn.genomics.fit_pair((lags, msd), model='dna')

    >>> # From saved MSD CSV
    >>> fit = wn.genomics.fit_pair('AA_msd.csv', model='dna')

    >>> # Custom initial guess (plateau visible at ~27, rise fast)
    >>> fit = wn.genomics.fit_pair('AA_msd.csv', model='dna', p0=[27.0, 0.025, 2.0])

    >>> # Try a different model on an off-diagonal pair
    >>> fit = wn.genomics.fit_pair('AC_msd.csv', model='exponential')
    """
    import whitenoise as wn

    if isinstance(source, str):
        lags_raw, msd_raw, _ = wn.read_csv(source)
        lags = lags_raw.astype(int)
        msd  = msd_raw
    elif isinstance(source, tuple):
        lags, msd = source
    else:
        lags = np.asarray(source, dtype=int)
        if msd is None:
            raise ValueError('✗ When source is an array, supply msd= as the second argument.')

    fit = wn.fit_msd(
        lags, msd, model=model,
        p0=p0, bounds=bounds,
        max_lag_fraction=max_lag_fraction,
    )

    if verbose:
        if fit is not None:
            print(fit.summary())
        else:
            print(f'  ✗ Fitting failed for model \'{model}\'')

    return fit


# ── 4. plot_pair ───────────────────────────────────────────────────────────────

def plot_pair(
    lags: np.ndarray,
    msd: np.ndarray,
    fit=None,
    title: str = '',
    save_path: str | None = None,
) -> 'matplotlib.figure.Figure':
    """
    Publication-quality MSD plot for a single transition pair.

    Parameters
    ----------
    lags : np.ndarray
        Lag values from :func:`compute_pair_msd`.
    msd : np.ndarray
        Empirical MSD values.
    fit : FitResult, optional
        If provided, overlays the fitted theoretical curve and a parameter
        annotation box.
    title : str, optional
        Plot title (bold).
    save_path : str, optional
        If provided, saves the figure as PNG at this path (dpi=150).

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> fig = wn.genomics.plot_pair(lags, msd, fit, title='A→A | DNA model')
    >>> fig.show()

    >>> # Save directly
    >>> wn.genomics.plot_pair(lags, msd, fit, title='A→A', save_path='AA_fit.png')
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.spines[['top', 'right']].set_visible(False)

    ax.scatter(lags, msd, s=12, alpha=0.6, color=_EMPIRICAL, label='Empirical MSD', zorder=2)

    if fit is not None and fit.msd_fitted is not None:
        ax.plot(
            fit.lags_used, fit.msd_fitted,
            color=_THEORETICAL, lw=2.5,
            label=f'{fit.model} fit',
            zorder=3,
        )

        # Build annotation text
        p = fit.params
        if fit.model == 'dna':
            ann = (
                f"a = {p.get('a', float('nan')):.4f}\n"
                f"b = {p.get('b', float('nan')):.6f}\n"
                f"c = {p.get('c', float('nan')):.4f}\n"
                f"R² = {fit.r_squared:.4f}"
            )
        else:
            lines = [
                f"{k} = {v:.4f}"
                for k, v in p.items()
                if k != 'N'
            ]
            lines.append(f"R² = {fit.r_squared:.4f}")
            ann = '\n'.join(lines)

        ax.text(
            0.97, 0.05, ann,
            ha='right', va='bottom',
            transform=ax.transAxes,
            fontsize=8, family='monospace',
            bbox=dict(facecolor='white', edgecolor=_SUBTLE, alpha=0.9, pad=3),
        )

    ax.set_xlabel('Lag L', fontsize=11)
    ax.set_ylabel('MSD', fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.tick_params(labelsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  ✓ Saved to {save_path}')

    return fig


# ── 5. analyze_pair ────────────────────────────────────────────────────────────

def analyze_pair(
    source,
    from_nuc: str,
    to_nuc: str,
    model: str = 'dna',
    max_lag: int = 1500,
    save_dir: str | None = None,
    p0: list | None = None,
    bounds: tuple | None = None,
    verbose: bool = True,
) -> dict:
    """
    Full single-pair SWNA pipeline in one call.

    Runs all four steps — extract distances, compute MSD, fit model, plot —
    and optionally saves all intermediate outputs to *save_dir*.

    Parameters
    ----------
    source : str
        * FASTA file path (ends with ``.fasta``, ``.fa``, ``.fna``, ``.fas``) —
          loaded automatically with :func:`~whitenoise.genomics.io.read_fasta_single`.
        * Pre-loaded sequence string — used directly.
    from_nuc : str
        Source nucleotide (``'A'``, ``'C'``, ``'G'``, or ``'T'``).
    to_nuc : str
        Target nucleotide.
    model : str, default ``'dna'``
        SWNA model.  See :func:`~whitenoise.core.models.list_models`.
    max_lag : int, default 1500
        Maximum lag for MSD computation.
    save_dir : str, optional
        Directory to save intermediate files.  Creates three files:

        * ``{from_nuc}_{to_nuc}_distances.csv`` — distance array
        * ``{from_nuc}_{to_nuc}_msd.csv``       — MSD arrays
        * ``{from_nuc}_{to_nuc}_fit.png``        — plot
    p0 : list, optional
        Initial parameter guess passed to :func:`fit_pair`.
    bounds : tuple, optional
        Parameter bounds passed to :func:`fit_pair`.
    verbose : bool, default True
        Print progress and fit summary.

    Returns
    -------
    dict
        Keys: ``pair``, ``n_distances``, ``lags``, ``msd``, ``fit``, ``figure``.

    Examples
    --------
    >>> result = wn.genomics.analyze_pair(
    ...     'chromosome1.fasta', 'A', 'A',
    ...     model='dna', max_lag=1500, save_dir='results/',
    ... )
    >>> print(result['fit'].r_squared)
    >>> result['figure'].show()

    >>> # Pass a pre-loaded sequence (skips re-reading the file)
    >>> result = wn.genomics.analyze_pair(seq, 'G', 'G', model='dna')
    """
    _FASTA_EXTS = ('.fasta', '.fa', '.fna', '.fas', '.ffn')

    if isinstance(source, str) and any(source.lower().endswith(ext) for ext in _FASTA_EXTS):
        if verbose:
            print(f'[1/4] Loading FASTA: {os.path.basename(source)}')
        record = read_fasta_single(source)
        seq = record['sequence']
    else:
        seq = source

    pair_tag = f'{from_nuc}_{to_nuc}'

    dist_csv = os.path.join(save_dir, f'{pair_tag}_distances.csv') if save_dir else None
    msd_csv  = os.path.join(save_dir, f'{pair_tag}_msd.csv')       if save_dir else None
    png_path = os.path.join(save_dir, f'{pair_tag}_fit.png')        if save_dir else None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    if verbose:
        print(f'\n── {from_nuc}→{to_nuc}  |  model={model}  |  max_lag={max_lag} ──')

    distances        = extract_pair(seq, from_nuc, to_nuc, save_path=dist_csv)
    lags, msd        = compute_pair_msd(distances, max_lag=max_lag, save_path=msd_csv)
    fit              = fit_pair((lags, msd), model=model, p0=p0, bounds=bounds, verbose=verbose)
    fig              = plot_pair(
        lags, msd, fit,
        title=f'{from_nuc}→{to_nuc} | {model}',
        save_path=png_path,
    )

    return {
        'pair':         (from_nuc, to_nuc),
        'n_distances':  len(distances),
        'lags':         lags,
        'msd':          msd,
        'fit':          fit,
        'figure':       fig,
    }


# ── 6. refit_pair ──────────────────────────────────────────────────────────────

def refit_pair(
    msd_csv_path: str,
    model: str = 'dna',
    p0: list | None = None,
    bounds: tuple | None = None,
    max_lag_fraction: float = 1.0,
    save_path: str | None = None,
    verbose: bool = True,
) -> tuple:
    """
    Reload a saved MSD CSV and re-fit with a different model or initial parameters.

    This is the primary tool for improving a poor fit without recomputing
    the distances or MSD from scratch.

    Parameters
    ----------
    msd_csv_path : str
        Path to a whitenoise-format MSD CSV saved by :func:`compute_pair_msd`
        or :func:`~whitenoise.io.export.export_msd`.
        Column 1 = lags, column 2 = msd values.
    model : str, default ``'dna'``
        SWNA model to fit.  Run ``wn.list_models()`` to see all options.
    p0 : list, optional
        Custom initial parameter guess.  For the DNA model: ``[a, b, c]``.

        **How to estimate p0 from the plot:**

        1. Look at the MSD curve in the saved PNG.
        2. Read off:
           - ``a`` ≈ the plateau (where the curve flattens)
           - ``c`` ≈ ``a − MSD(L=1)`` (the initial gap below plateau)
           - ``b`` ≈ ``6 / L_sat``  where ``L_sat`` is the lag where
             the curve visually reaches the plateau.
        3. Pass these as ``p0=[a, b, c]``.
    bounds : tuple, optional
        ``(lower_bounds, upper_bounds)`` for physical params only.
    max_lag_fraction : float, default 1.0
        Use only the first fraction of lags for fitting.
        Set to ``0.7`` or ``0.8`` to exclude the noisy flat tail.
    save_path : str, optional
        If provided, saves the new plot as PNG.
    verbose : bool, default True
        Print fit summary.

    Returns
    -------
    (FitResult or None, matplotlib.figure.Figure)

    Examples
    --------
    >>> # Retry with better initial guess
    >>> fit, fig = wn.genomics.refit_pair(
    ...     'results/A_A_msd.csv',
    ...     model='dna',
    ...     p0=[27.0, 0.025, 2.0],
    ...     save_path='results/A_A_refit.png',
    ... )
    >>> print(fit.r_squared)

    >>> # Try exponential model on an off-diagonal pair
    >>> fit, fig = wn.genomics.refit_pair('results/A_C_msd.csv', model='exponential')

    >>> # Fit only first 70% of lags (exclude noisy plateau)
    >>> fit, fig = wn.genomics.refit_pair('results/A_A_msd.csv', max_lag_fraction=0.7)
    """
    import whitenoise as wn

    lags_raw, msd_raw, _ = wn.read_csv(msd_csv_path)
    lags = lags_raw.astype(int)
    msd  = msd_raw

    title = f'Refit: {model} — {os.path.basename(msd_csv_path)}'

    fit = fit_pair(
        (lags, msd),
        model=model, p0=p0, bounds=bounds,
        max_lag_fraction=max_lag_fraction,
        verbose=verbose,
    )

    fig = plot_pair(lags, msd, fit, title=title, save_path=save_path)

    return fit, fig
