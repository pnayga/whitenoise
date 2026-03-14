"""
io/reader.py — CSV reading for the whitenoise package.

Handles the standard whitenoise CSV format::

    time [months], sunspot_number [count]
    1, 12.4
    2, 15.1

Column 1 is always the independent variable (x). Column 2 (and beyond) are
the dependent observables (y). Column 1 does not have to be time — it can be
any independent variable (distance, frequency, voltage, etc.).
Header format:  name [unit]   — space before bracket is optional.
"""

from __future__ import annotations

import os
import re

import numpy as np

# ── Validation constants ──────────────────────────────────────────────────────

# Common independent-variable (x-axis) column names used to detect possible
# column swaps (V2 check). Non-standard names are still accepted — this list
# only triggers a warning, never an error.
_KNOWN_INDEX_NAMES: set[str] = {
    'time', 't', 'index', 'idx', 'step', 'steps', 'sample',
    'samples', 'observation', 'observations', 'date', 'datetime',
    'timestamp', 'epoch', 'frame', 'lag', 'year', 'month', 'day',
    'hour', 'minute', 'second', 'yr', 'mo', 'hr', 'min', 'sec',
}

# Units that physically imply non-negative values, used to warn if negatives appear (V3 check).
# Non-standard or unlisted units are still accepted — this list only triggers a warning, never an error.
_NON_NEGATIVE_UNITS: set[str] = {
    'count', 'counts', 'number', 'numbers', 'freq', 'frequency',
    'intensity', 'flux', 'brightness', 'magnitude', 'population',
    'cases', 'events', 'price', 'usd', 'eur', 'ppm', 'ppb',
    'percent', '%', 'fraction', 'probability',
}

_HEADER_RE = re.compile(r'^\s*([^\[\]]+?)\s*(?:\[\s*([^\]]*?)\s*\])?\s*$')


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_header_column(header: str) -> tuple[str, str]:
    """
    Parse a single CSV header string into (name, unit), both lowercase.

    Handles all formatting variants::

        'time [months]'          → ('time', 'months')
        'time[months]'           → ('time', 'months')
        'Time [Months]'          → ('time', 'months')
        'sunspot_number [count]' → ('sunspot_number', 'count')
        'flux []'                → ('flux', '')
        'flux [  ]'              → ('flux', '')
        'value [unitless]'       → ('value', 'unitless')
        'distance'               → ('distance', '')
        '  time  [months]  '     → ('time', 'months')

    Parameters
    ----------
    header : str
        A single column header string.

    Returns
    -------
    name : str
        Column name, stripped and lowercased.
    unit : str
        Unit string, stripped and lowercased. Empty string if absent or blank.
    """
    m = _HEADER_RE.match(header)
    if not m:
        return header.strip().lower(), ''
    name = m.group(1).strip().lower()
    unit_raw = m.group(2)
    unit = unit_raw.strip().lower() if unit_raw is not None else ''
    return name, unit


def _make_axis_label(name: str, unit: str) -> str:
    """
    Build a matplotlib-ready axis label string.

    Parameters
    ----------
    name : str
        Column name (already lowercase).
    unit : str
        Unit string (already lowercase). Use '' or 'unitless' for no unit.

    Returns
    -------
    str
        ``name`` if unit is empty/unitless, else ``'name (unit)'``.

    Examples
    --------
    >>> _make_axis_label('sunspot_number', 'count')
    'sunspot_number (count)'
    >>> _make_axis_label('flux', '')
    'flux'
    >>> _make_axis_label('flux', 'unitless')
    'flux'
    """
    if not unit or unit == 'unitless':
        return name
    return f'{name} ({unit})'


def _validate(x: np.ndarray, y: np.ndarray, metadata: dict) -> None:
    """
    Run 3 non-fatal validation checks and print ⚠ warnings to stdout.

    Never raises; data is loaded regardless of warnings.

    Parameters
    ----------
    x : np.ndarray
        Column 1 (independent variable) values.
    y : np.ndarray
        Column 2 (dependent/observable) values.
    metadata : dict
        Must contain 'x_name', 'y_name', 'y_unit'.
    """
    # V1: Columns possibly swapped (column 1 has huge spread relative to its mean)
    if len(x) > 1:
        std_x = float(np.std(x))
        mean_x = float(np.mean(x))
        if std_x > 100.0 * (abs(mean_x) + 1e-10):
            print(
                f"⚠  Column order warning: column 1 "
                f"('{metadata['x_name']}') has unusually large spread "
                f"(std={std_x:.2f}, mean={mean_x:.2f}).\n"
                f"   If your columns are swapped, re-save the CSV with the "
                f"independent variable in column 1 and the observable in column 2."
            )

    # V2: Column 1 name doesn't match common time/index names.
    # Column 1 is not required to be time — it can be any independent variable
    # (e.g. distance [km], frequency [Hz], voltage [V]). This warning only fires
    # if the name is unfamiliar, to catch accidental column swaps.
    if metadata['x_name'].lower() not in _KNOWN_INDEX_NAMES:
        print(
            f"⚠  Column 1 name warning: '{metadata['x_name']}' is not a "
            f"commonly recognized time or index name.\n"
            f"   If column 1 is a non-time independent variable (e.g. distance, "
            f"frequency, voltage), this warning is expected — ignore it.\n"
            f"   If your columns are accidentally swapped, re-save the CSV with "
            f"the independent variable in column 1 and the observable in column 2."
        )

    # V3: Negative values in a unit that implies non-negative data
    y_unit = metadata['y_unit'].lower()
    if y_unit in _NON_NEGATIVE_UNITS and np.any(y < 0):
        n_neg = int(np.sum(y < 0))
        min_val = float(np.min(y))
        print(
            f"⚠  Value range warning: column '{metadata['y_name']}' has "
            f"unit '{metadata['y_unit']}' which typically implies "
            f"non-negative values, but {n_neg} negative value(s) were found "
            f"(min = {min_val:.4f}).\n"
            f"   If your data is already detrended, this is expected — ignore "
            f"this warning."
        )


def _build_metadata(
    x_name: str,
    x_unit: str,
    y_name: str,
    y_unit: str,
    source_file: str,
    n_points: int,
) -> dict:
    """Assemble the standard metadata dict returned by read functions."""
    return {
        'x_name':      x_name,
        'x_unit':      x_unit,
        'y_name':      y_name,
        'y_unit':      y_unit,
        'x_label':     _make_axis_label(x_name, x_unit),
        'y_label':     _make_axis_label(y_name, y_unit),
        'source_file': source_file,
        'n_points':    n_points,
    }


def _read_raw_lines(path: str) -> list[str]:
    """Open file, strip blank lines, raise clean errors."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f'✗ File not found: {path}')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            raw = fh.read()
    except Exception as exc:
        raise ValueError(f'✗ Could not read file "{path}": {exc}') from exc
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f'✗ File is empty: {path}')
    return lines


def _parse_header_line(header_line: str, min_cols: int = 2) -> list[tuple[str, str]]:
    """Split a header line on commas and parse each column. Returns list of (name, unit)."""
    cols = [c.strip() for c in header_line.split(',')]
    if len(cols) < min_cols:
        raise ValueError(
            f'✗ CSV must have at least {min_cols} columns '
            f'(independent variable and observable). '
            f'Found {len(cols)} column(s) in header: "{header_line}"'
        )
    return [_parse_header_column(c) for c in cols]


def _parse_data_rows(
    lines: list[str],
    n_cols: int,
    col_indices: list[int],
) -> list[list[float]]:
    """
    Parse data rows into lists of floats.

    Parameters
    ----------
    lines : list[str]
        Data lines (no header).
    n_cols : int
        Expected number of comma-separated columns.
    col_indices : list[int]
        Which column indices to extract.

    Returns
    -------
    list of lists — one inner list per col_index.
    """
    columns: list[list[float]] = [[] for _ in col_indices]
    for row_num, line in enumerate(lines, start=2):
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < n_cols:
            raise ValueError(
                f'✗ Row {row_num} has {len(parts)} value(s) but the header '
                f'declares {n_cols} column(s). Line: "{line}"'
            )
        for out_idx, col_idx in enumerate(col_indices):
            raw = parts[col_idx]
            try:
                columns[out_idx].append(float(raw))
            except ValueError:
                raise ValueError(
                    f'✗ Non-numeric value in row {row_num}, '
                    f'column {col_idx + 1}: "{raw}". '
                    f'All data cells must be numbers.'
                )
    return columns


# ── Public API ────────────────────────────────────────────────────────────────

def read_csv(path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Read a 2-column whitenoise-format CSV file.

    The expected format is::

        time [months], sunspot_number [count]
        1, 12.4
        2, 15.1
        3, 14.8

    Column 1 is always the independent variable (x-axis). Column 2 is always
    the observable (y-axis). Column 1 does not have to be time — any
    independent variable is accepted (distance, frequency, voltage, etc.).
    Header strings follow the ``name [unit]`` convention; the space before
    ``[`` is optional and capitalisation is ignored.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    x : np.ndarray
        1-D float array of independent variable values (column 1).
    y : np.ndarray
        1-D float array of observable values (column 2).
    metadata : dict
        Keys:

        * ``'x_name'``      — column 1 name, lowercased (e.g. ``'time'``)
        * ``'x_unit'``      — column 1 unit, lowercased (``''`` if absent)
        * ``'y_name'``      — column 2 name, lowercased (e.g. ``'sunspot_number'``)
        * ``'y_unit'``      — column 2 unit, lowercased (``''`` if absent)
        * ``'x_label'``     — axis-ready string (e.g. ``'time (months)'``)
        * ``'y_label'``     — axis-ready string (e.g. ``'sunspot_number (count)'``)
        * ``'source_file'`` — the ``path`` argument as given
        * ``'n_points'``    — number of data rows read

    Raises
    ------
    FileNotFoundError
        ``'✗ File not found: {path}'``
    ValueError
        ``'✗ {reason}'`` for malformed files (wrong column count,
        non-numeric data, empty file, etc.).

    Examples
    --------
    >>> x, y, meta = wn.read_csv('sunspot.csv')
    >>> meta['y_label']
    'sunspot_number (count)'
    >>> meta['x_label']
    'time (months)'
    """
    lines = _read_raw_lines(path)
    parsed_headers = _parse_header_line(lines[0], min_cols=2)
    x_name, x_unit = parsed_headers[0]
    y_name, y_unit = parsed_headers[1]

    data_lines = lines[1:]
    if not data_lines:
        raise ValueError(f'✗ File has a header but no data rows: "{path}"')

    columns = _parse_data_rows(data_lines, n_cols=len(parsed_headers), col_indices=[0, 1])

    x_arr = np.array(columns[0], dtype=float)
    y_arr = np.array(columns[1], dtype=float)

    metadata = _build_metadata(x_name, x_unit, y_name, y_unit, path, len(x_arr))
    _validate(x_arr, y_arr, metadata)

    return x_arr, y_arr, metadata


def read_csv_multi(path: str) -> list[tuple[np.ndarray, np.ndarray, dict]]:
    """
    Read a multi-column whitenoise-format CSV file.

    Column 1 is the shared independent variable (x-axis). Every additional
    column is treated as a separate observable (y) and returned as its own
    ``(x, y, metadata)`` tuple. Used internally by ``batch_analyze()`` and
    available for advanced users who want to process multi-system CSVs manually.

    The expected format is::

        time [yr], co2 [ppm], temperature [°C]
        1958, 315.2, 14.1
        1959, 315.9, 14.0

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    list of (x, y, metadata) tuples
        One tuple per observable column (i.e. ``n_columns - 1`` tuples).
        Each ``x`` array is the same shared independent variable.

    Raises
    ------
    FileNotFoundError
        ``'✗ File not found: {path}'``
    ValueError
        ``'✗ {reason}'`` for malformed files.

    Examples
    --------
    >>> datasets = wn.read_csv_multi('climate.csv')
    >>> len(datasets)      # 2 if CSV has x + 2 observable columns
    2
    >>> datasets[0][2]['y_name']
    'co2'
    """
    lines = _read_raw_lines(path)
    parsed_headers = _parse_header_line(lines[0], min_cols=2)
    x_name, x_unit = parsed_headers[0]
    n_cols = len(parsed_headers)

    data_lines = lines[1:]
    if not data_lines:
        raise ValueError(f'✗ File has a header but no data rows: "{path}"')

    all_col_indices = list(range(n_cols))
    columns = _parse_data_rows(data_lines, n_cols=n_cols, col_indices=all_col_indices)

    x_arr = np.array(columns[0], dtype=float)
    n_points = len(x_arr)

    results: list[tuple[np.ndarray, np.ndarray, dict]] = []
    for j in range(1, n_cols):
        y_name, y_unit = parsed_headers[j]
        y_arr = np.array(columns[j], dtype=float)
        meta = _build_metadata(x_name, x_unit, y_name, y_unit, path, n_points)
        _validate(x_arr, y_arr, meta)
        results.append((x_arr, y_arr, meta))

    return results
