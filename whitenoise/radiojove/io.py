"""
whitenoise.radiojove.io
=======================
I/O utilities for RadioJOVE radio burst CSV data.

RadioJOVE CSV format
--------------------
Files produced by the RadioJOVE region-selection GUI follow a naming
convention and a fixed two-column format::

    time_seconds,intensity_zscore
    0.000,-4.050984
    0.100,-4.054093
    ...

Column 1 is time in seconds; Column 2 is Z-score normalized intensity.
The header does not use whitenoise bracket notation — both columns are read
as unitless by the standard ``wn.read_csv()`` parser.

Filename convention::

    YYMMDDHHMMSS<Station>_region_<N>_<cadence>s.csv

Example::

    230728154000Higgins_Home_region_1_0.1s.csv
    230728154000Higgins_Home_region_1_1.0s.csv

The datetime prefix is 12 digits: YYMMDDHHMMSS (UTC or local depending on
the recording station's configuration).
"""

from __future__ import annotations

import os
import re
import glob


# ── read_radiojove_csv ─────────────────────────────────────────────────────────

def read_radiojove_csv(path: str) -> tuple:
    """
    Read a RadioJOVE CSV and return arrays with enriched metadata.

    Wraps ``wn.read_csv()`` and extends the returned metadata with station,
    region, cadence, and datetime parsed from the filename.

    Parameters
    ----------
    path : str
        Path to a RadioJOVE CSV file.

    Returns
    -------
    time : np.ndarray
        Time values in seconds.
    intensity : np.ndarray
        Z-score normalized intensity values.
    metadata : dict
        All fields from ``wn.read_csv()`` plus:

        - ``'station'``   : str   — station name (e.g. ``'Higgins_Home'``)
        - ``'region'``    : int   — region number (e.g. ``1``)
        - ``'cadence_s'`` : float — time cadence in seconds (e.g. ``0.1``)
        - ``'datetime'``  : str   — raw datetime string (``'YYMMDDHHMMSS'``)

    Raises
    ------
    FileNotFoundError
        Forwarded from ``wn.read_csv()`` if the file is missing.

    Example
    -------
    >>> import whitenoise as wn
    >>> time, intensity, meta = wn.radiojove.read_radiojove_csv(
    ...     '230728154000Higgins_Home_region_1_0.1s.csv'
    ... )
    >>> print(meta['station'], meta['cadence_s'])
    Higgins_Home 0.1
    """
    import whitenoise as wn

    time, intensity, metadata = wn.read_csv(path)

    filename = os.path.basename(path)
    file_meta = parse_filename_metadata(filename)
    metadata.update(file_meta)

    return time, intensity, metadata


# ── parse_filename_metadata ───────────────────────────────────────────────────

def parse_filename_metadata(filename: str) -> dict:
    """
    Extract RadioJOVE recording metadata from a CSV filename.

    Expected pattern::

        YYMMDDHHMMSS<Station>_region_<N>_<cadence>s.csv

    Parsing is done with regex and degrades gracefully — any field that
    cannot be extracted is returned as ``None`` rather than raising.

    Parameters
    ----------
    filename : str
        CSV filename (basename only, not a full path).

    Returns
    -------
    dict with keys:

        ``'filename'``  — the filename argument unchanged
        ``'datetime'``  — 12-character timestamp string (e.g. ``'230728154000'``)
        ``'station'``   — station name (e.g. ``'Higgins_Home'``)
        ``'region'``    — region number as int (``None`` if not found)
        ``'cadence_s'`` — cadence in seconds as float (``None`` if not found)

    Examples
    --------
    >>> meta = wn.radiojove.parse_filename_metadata(
    ...     '230728154000Higgins_Home_region_1_0.1s.csv'
    ... )
    >>> meta['datetime']   # '230728154000'
    >>> meta['station']    # 'Higgins_Home'
    >>> meta['region']     # 1
    >>> meta['cadence_s']  # 0.1

    >>> # Partial filename — still extracts what it can
    >>> meta = wn.radiojove.parse_filename_metadata('region_2_1.0s.csv')
    >>> meta['region']     # 2
    >>> meta['cadence_s']  # 1.0
    >>> meta['datetime']   # None
    """
    stem = os.path.splitext(filename)[0]

    result: dict = {
        'filename':  filename,
        'datetime':  None,
        'station':   None,
        'region':    None,
        'cadence_s': None,
    }

    # Full pattern: 12-digit timestamp + station + _region_N + _<float>s
    m = re.match(
        r'^(\d{12})'               # 12-digit YYMMDDHHMMSS
        r'(.+?)'                   # station name (non-greedy)
        r'_region_(\d+)'           # _region_N
        r'_(\d+(?:\.\d+)?)s$',    # _<cadence>s  (float, no unit char after)
        stem,
    )
    if m:
        result['datetime']  = m.group(1)
        result['station']   = m.group(2)
        result['region']    = int(m.group(3))
        result['cadence_s'] = float(m.group(4))
        return result

    # Partial fallback: extract each field independently
    dt_m = re.match(r'^(\d{12})', stem)
    if dt_m:
        result['datetime'] = dt_m.group(1)

    reg_m = re.search(r'region_(\d+)', stem)
    if reg_m:
        result['region'] = int(reg_m.group(1))

    cad_m = re.search(r'_(\d+(?:\.\d+)?)s$', stem)
    if cad_m:
        result['cadence_s'] = float(cad_m.group(1))

    # Station: everything between the 12-digit prefix and "_region_"
    station_m = re.match(r'^\d{12}(.+?)_region_', stem)
    if station_m:
        result['station'] = station_m.group(1)

    return result


# ── list_burst_csvs ───────────────────────────────────────────────────────────

def list_burst_csvs(folder: str, pattern: str = '*.csv') -> list:
    """
    Return a sorted list of RadioJOVE CSV file paths in *folder*.

    Parameters
    ----------
    folder : str
        Directory containing RadioJOVE CSV files.
    pattern : str
        Glob pattern to filter files. Default ``'*.csv'``.

    Returns
    -------
    list of str
        Absolute paths, sorted alphabetically.

    Raises
    ------
    FileNotFoundError
        If *folder* does not exist.

    Example
    -------
    >>> paths = wn.radiojove.list_burst_csvs('Solar/Type 3 Bursts/first trials/')
    >>> print(len(paths), 'files found')
    6 files found
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(f'✗ Directory not found: {folder}')

    return sorted(glob.glob(os.path.join(folder, pattern)))
