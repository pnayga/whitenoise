"""
whitenoise.genomics.io
======================
FASTA file reader for genomic DNA sequences.

Accepts standard FASTA format as downloaded from NCBI GenBank, NCBI Genome,
or BOLD. No external dependencies — pure Python + NumPy only.

Optionally accepts a BioPython SeqRecord via parse_seqrecord() if the caller
already has BioPython loaded, but BioPython is NOT a required dependency.
"""

import os
import re

# Standard IUPAC ambiguity codes (non-ACGT)
_AMBIGUOUS = set('WSMKRYBDHVN')
_VALID_BASES = set('ACGT')


def read_fasta(path: str) -> list:
    """
    Read a FASTA file and return a list of sequence records.

    Handles:
    - Single-sequence and multi-sequence FASTA files
    - Sequences split across multiple lines (standard NCBI format)
    - Lowercase sequences (auto-uppercased)
    - Ambiguous IUPAC bases (warned, kept in sequence as-is)
    - Windows and Unix line endings

    Parameters
    ----------
    path : str
        Path to a FASTA file (.fasta, .fa, .fas, .fna, .ffn, .faa, .frn).
        Multi-record FASTA files return one dict per record.

    Returns
    -------
    list of dict, each containing:
        'id'          : str  — accession/identifier (first token after '>')
        'description' : str  — full header line (without '>')
        'sequence'    : str  — uppercase sequence string (all characters kept)
        'length'      : int  — total sequence length
        'n_ambiguous' : int  — count of non-ACGT characters in sequence

    Raises
    ------
    FileNotFoundError
        If the file does not exist at the given path.
    ValueError
        If the file contains no valid FASTA records (empty or wrong format).

    Notes
    -----
    Ambiguous bases (W, S, M, K, R, Y, B, D, H, V, N) are kept in the
    returned sequence string. The distance functions in genomics.distances
    will simply find no matches for non-ACGT characters — they are not
    removed silently, so the caller retains full control.

    A warning is printed if ambiguous bases are found, reporting the count
    and percentage so the researcher can decide whether to filter them.

    Examples
    --------
    >>> records = read_fasta('Strombus_labiatus_COI.fasta')
    >>> len(records)
    12
    >>> records[0]['id']
    'LC123456.1'
    >>> records[0]['length']
    658
    >>> seq = records[0]['sequence']

    >>> # Single-record genome
    >>> records = read_fasta('Synechococcus_elongatus.fasta')
    >>> seq = records[0]['sequence']
    >>> len(seq)
    2700000
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] File not found: {path}")

    records = []
    current_id = None
    current_desc = None
    current_seq_parts = []

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save previous record if any
                if current_id is not None:
                    records.append(_finalize_record(
                        current_id, current_desc, current_seq_parts, path
                    ))
                # Start new record
                header = line[1:].strip()
                parts = header.split(None, 1)
                current_id = parts[0] if parts else 'unknown'
                current_desc = header
                current_seq_parts = []
            else:
                if current_id is None:
                    # Sequence data before any header — likely not a FASTA file
                    raise ValueError(
                        f"[ERROR] File does not appear to be FASTA format: {path}\n"
                        "  Expected first non-empty line to start with '>'."
                    )
                current_seq_parts.append(line.upper())

    # Save last record
    if current_id is not None:
        records.append(_finalize_record(
            current_id, current_desc, current_seq_parts, path
        ))

    if not records:
        raise ValueError(
            f"[ERROR] No FASTA records found in: {path}\n"
            "  File may be empty or in an unsupported format."
        )

    return records


def _finalize_record(record_id, description, seq_parts, source_path):
    """Assemble and validate a single FASTA record."""
    sequence = ''.join(seq_parts)

    n_ambiguous = sum(1 for ch in sequence if ch not in _VALID_BASES and ch.isalpha())

    if n_ambiguous > 0:
        pct = 100.0 * n_ambiguous / max(len(sequence), 1)
        print(
            f"[WARN] Record '{record_id}': {n_ambiguous} ambiguous base(s) "
            f"({pct:.1f}% of {len(sequence)} bp). "
            f"These are kept as-is. Filter with filter_sequence() if needed."
        )

    return {
        'id':          record_id,
        'description': description,
        'sequence':    sequence,
        'length':      len(sequence),
        'n_ambiguous': n_ambiguous,
    }


def read_fasta_single(path: str) -> dict:
    """
    Read a FASTA file and return only the first record.

    Convenience wrapper around read_fasta() for single-sequence files
    (e.g., one COI barcode, one whole genome).

    Parameters
    ----------
    path : str
        Path to FASTA file.

    Returns
    -------
    dict
        Single record dict (same structure as read_fasta() list elements).

    Raises
    ------
    FileNotFoundError, ValueError
        Same as read_fasta().

    Example
    -------
    >>> record = read_fasta_single('genome.fasta')
    >>> seq = record['sequence']
    """
    records = read_fasta(path)
    if len(records) > 1:
        print(
            f"[WARN] File contains {len(records)} records. "
            f"Returning first record only ('{records[0]['id']}').\n"
            f"  Use read_fasta() to get all records."
        )
    return records[0]


def filter_sequence(sequence: str, keep: str = 'ACGT') -> str:
    """
    Remove characters not in `keep` from a sequence string.

    Use this before passing a sequence to get_transition_distances() if
    your data contains ambiguous IUPAC bases that you want to exclude.

    Parameters
    ----------
    sequence : str
        DNA sequence string (typically from a read_fasta() record).
    keep : str
        Characters to retain. Default 'ACGT'.

    Returns
    -------
    str
        Filtered sequence containing only characters in `keep`.

    Example
    -------
    >>> seq = 'ATGWNNTCGA'
    >>> filter_sequence(seq)
    'ATGTCGA'
    """
    keep_set = set(keep.upper())
    return ''.join(ch for ch in sequence.upper() if ch in keep_set)


def parse_seqrecord(seqrecord) -> dict:
    """
    Convert a BioPython SeqRecord into a whitenoise genomics record dict.

    This function exists for callers who already have BioPython loaded and
    want to pass SeqRecord objects directly. BioPython is NOT required by
    this package — only use this if you have it installed.

    Parameters
    ----------
    seqrecord : Bio.SeqRecord.SeqRecord
        A BioPython SeqRecord object.

    Returns
    -------
    dict
        Same structure as read_fasta() records.

    Example
    -------
    >>> from Bio import SeqIO
    >>> for sr in SeqIO.parse('genome.fasta', 'fasta'):
    ...     record = parse_seqrecord(sr)
    ...     distances = get_transition_distances(record['sequence'], 'A', 'T')
    """
    sequence = str(seqrecord.seq).upper()
    n_ambiguous = sum(1 for ch in sequence if ch not in _VALID_BASES and ch.isalpha())
    return {
        'id':          seqrecord.id,
        'description': seqrecord.description,
        'sequence':    sequence,
        'length':      len(sequence),
        'n_ambiguous': n_ambiguous,
    }
