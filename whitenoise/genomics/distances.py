"""
whitenoise.genomics.distances
==============================
Fast nucleotide distance extraction for SWNA on genomic DNA.

Implements the stochastic variable defined in Violanda et al. (2019):
the separation distance between consecutive identical (or transitioning)
nucleotides along a DNA sequence.

For same-type distances (A→A, G→G, etc.), this reproduces the method
used in the paper. For cross-type distances (A→C, A→G, etc.), this
implements the nearest-next interpretation: from each occurrence of
`from_nuc`, find the positional distance to the very next occurrence of
`to_nuc` downstream.

Performance
-----------
Uses NumPy vectorized operations throughout. No Python-level loops over
the sequence. For a 2.7 Mbp genome (~1.5M occurrences of a given base):
  - Sequence scanning:  O(N)   via np.where on byte array
  - Distance lookup:    O(N log M) via np.searchsorted (binary search)
  - Estimated runtime:  < 100ms per (from_nuc, to_nuc) pair

References
----------
Violanda R R, Bernido C C, Carpio-Bernido M V (2019).
"White noise functional integral for exponentially decaying memory:
nucleotide distribution in bacterial genomes."
Physica Scripta 94, 125006.
"""

import numpy as np
from itertools import product as _product

_NUCLEOTIDES = ('A', 'C', 'G', 'T')


def get_transition_distances(
    sequence: str,
    from_nuc: str,
    to_nuc: str,
) -> np.ndarray:
    """
    Extract the sequence of separation distances between nucleotides.

    For same-type pairs (from_nuc == to_nuc):
        Computes distances between consecutive occurrences of the same
        nucleotide — identical to the paper's method.
        distance[i] = position[i+1] - position[i]

    For cross-type pairs (from_nuc != to_nuc):
        For each occurrence of from_nuc, finds the positional distance to
        the very next occurrence of to_nuc downstream (nearest-next rule).
        Occurrences of from_nuc with no subsequent to_nuc are dropped.

    This sequence of distances is the stochastic variable x_i in the
    Violanda et al. framework. Pass it directly to wn.compute_msd() and
    wn.analyze() for SWNA fitting.

    Parameters
    ----------
    sequence : str
        DNA sequence string. Must be uppercase (use read_fasta() which
        auto-uppercases, or call sequence.upper() manually).
        Non-ACGT characters are silently ignored during position lookup.
    from_nuc : str
        Source nucleotide — one of 'A', 'C', 'G', 'T'.
    to_nuc : str
        Target nucleotide — one of 'A', 'C', 'G', 'T'.

    Returns
    -------
    np.ndarray (1D, dtype=int64)
        Array of separation distances. Length = number of valid transitions
        found. Returns empty array if fewer than 2 occurrences exist.

    Raises
    ------
    ValueError
        If from_nuc or to_nuc is not a single character in 'ACGT'.

    Notes
    -----
    The diagonal of the 4×4 transition matrix (from_nuc == to_nuc) gives
    the same result as the Violanda et al. paper's same-type distances,
    and can be used as a sanity check against published parameter values.

    Examples
    --------
    >>> seq = 'ACCTAGAGCGGAATGCTA'
    >>> d = get_transition_distances(seq, 'A', 'A')
    >>> d
    array([4, 2, 5, ...])

    >>> d = get_transition_distances(seq, 'A', 'C')
    >>> # From each A, distance to next C downstream
    """
    from_nuc = from_nuc.upper()
    to_nuc = to_nuc.upper()
    _validate_nucleotide(from_nuc, 'from_nuc')
    _validate_nucleotide(to_nuc, 'to_nuc')

    # Convert string to byte array for vectorized scanning — O(N)
    seq_bytes = np.frombuffer(sequence.encode('ascii'), dtype=np.uint8)

    from_positions = np.where(seq_bytes == ord(from_nuc))[0]

    if len(from_positions) < 2 and from_nuc == to_nuc:
        return np.array([], dtype=np.int64)
    if len(from_positions) == 0:
        return np.array([], dtype=np.int64)

    # Same-type: consecutive differences — O(N)
    if from_nuc == to_nuc:
        return np.diff(from_positions).astype(np.int64)

    # Cross-type: binary search for next to_nuc after each from_nuc — O(N log M)
    to_positions = np.where(seq_bytes == ord(to_nuc))[0]

    if len(to_positions) == 0:
        return np.array([], dtype=np.int64)

    # searchsorted returns the index in to_positions where from_pos would
    # be inserted to keep sorted order — i.e., the index of the first
    # to_pos that is strictly greater than from_pos
    indices = np.searchsorted(to_positions, from_positions, side='right')

    # Drop from_positions where no to_nuc exists downstream
    valid_mask = indices < len(to_positions)
    valid_from = from_positions[valid_mask]
    valid_to = to_positions[indices[valid_mask]]

    return (valid_to - valid_from).astype(np.int64)


def get_all_transition_distances(
    sequence: str,
    nucleotides: str = 'ACGT',
) -> dict:
    """
    Compute the full N×N transition distance matrix for a DNA sequence.

    Returns a dictionary of all pairwise transition distance arrays,
    forming the complete transition matrix described in Violanda et al.
    (2019) extended to include cross-type transitions.

    The diagonal entries (A→A, C→C, G→G, T→T) reproduce the same-type
    distances from the paper. Off-diagonal entries are the cross-type
    nearest-next distances.

    Parameters
    ----------
    sequence : str
        DNA sequence string (uppercase recommended).
    nucleotides : str
        Which nucleotides to include. Default 'ACGT' gives a 4×4 matrix
        (16 pairs). Use a subset like 'AG' for a 2×2 matrix (4 pairs).

    Returns
    -------
    dict
        Keys are (from_nuc, to_nuc) tuples, e.g. ('A', 'C').
        Values are np.ndarray of distances (may be empty for rare pairs).

        Example keys for nucleotides='ACGT':
          ('A','A'), ('A','C'), ('A','G'), ('A','T'),
          ('C','A'), ('C','C'), ('C','G'), ('C','T'),
          ('G','A'), ('G','C'), ('G','G'), ('G','T'),
          ('T','A'), ('T','C'), ('T','G'), ('T','T')

    Notes
    -----
    For a COI barcode (~658 bp), each array will have ~50-150 values.
    For a whole bacterial genome (~2.7 Mbp), each array will have
    ~100,000-700,000 values depending on nucleotide frequency.

    Empty arrays are returned (not errors) for pairs with insufficient
    data. Check len(distances) before passing to wn.compute_msd().

    Example
    -------
    >>> records = wn.genomics.read_fasta('genome.fasta')
    >>> seq = records[0]['sequence']
    >>> matrix = wn.genomics.get_all_transition_distances(seq)
    >>> print(len(matrix[('A', 'T')]))   # number of A→T transitions
    >>>
    >>> # Feed one pair into the SWNA pipeline
    >>> distances = matrix[('A', 'T')]
    >>> result = wn.analyze(distances, model='dna', label='A→T')
    >>> result.summary()
    """
    nucs = [n.upper() for n in nucleotides]
    for n in nucs:
        _validate_nucleotide(n, 'nucleotides')

    # Pre-compute byte array once — reused across all pairs
    seq_bytes = np.frombuffer(sequence.upper().encode('ascii'), dtype=np.uint8)

    # Pre-compute positions for each nucleotide — O(4N) total
    positions = {n: np.where(seq_bytes == ord(n))[0] for n in nucs}

    result = {}
    for from_nuc, to_nuc in _product(nucs, nucs):
        from_pos = positions[from_nuc]
        to_pos = positions[to_nuc]

        if from_nuc == to_nuc:
            if len(from_pos) < 2:
                result[(from_nuc, to_nuc)] = np.array([], dtype=np.int64)
            else:
                result[(from_nuc, to_nuc)] = np.diff(from_pos).astype(np.int64)
        else:
            if len(from_pos) == 0 or len(to_pos) == 0:
                result[(from_nuc, to_nuc)] = np.array([], dtype=np.int64)
            else:
                indices = np.searchsorted(to_pos, from_pos, side='right')
                valid_mask = indices < len(to_pos)
                valid_from = from_pos[valid_mask]
                valid_to = to_pos[indices[valid_mask]]
                result[(from_nuc, to_nuc)] = (valid_to - valid_from).astype(np.int64)

    return result


def summarize_matrix(matrix: dict) -> None:
    """
    Print a summary table of the 4×4 transition distance matrix.

    Shows the number of distance values and mean distance for each pair,
    useful for a quick sanity check before running SWNA analysis.

    Parameters
    ----------
    matrix : dict
        Output of get_all_transition_distances().

    Example
    -------
    >>> matrix = wn.genomics.get_all_transition_distances(seq)
    >>> wn.genomics.summarize_matrix(matrix)
    """
    nucs = sorted(set(k[0] for k in matrix))

    print(f"\n{'':>6}", end='')
    for to_n in nucs:
        print(f"  ->{to_n:>9}", end='')
    print()

    print('-' * (6 + 13 * len(nucs)))

    for from_n in nucs:
        print(f"  {from_n}-> ", end='')
        for to_n in nucs:
            arr = matrix.get((from_n, to_n), np.array([]))
            if len(arr) == 0:
                print(f"{'—':>12}", end='')
            else:
                print(f"  n={len(arr):>6}", end='')
        print()

    print()
    print("Values shown: number of distance observations per transition pair.")
    print("Pass matrix[(from, to)] to wn.compute_msd() for SWNA analysis.\n")


def _validate_nucleotide(nuc: str, param_name: str) -> None:
    if nuc not in ('A', 'C', 'G', 'T'):
        raise ValueError(
            f"[ERROR] Invalid nucleotide for '{param_name}': '{nuc}'. "
            f"Must be one of: 'A', 'C', 'G', 'T'."
        )
