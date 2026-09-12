"""
Core cleaning + profiling logic, separate from the UI.
"""

import re
import numpy as np
import pandas as pd


# --------------------------------------------------------------------
# Profiling
# --------------------------------------------------------------------

def profile_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-row-per-column summary: dtype, missing %, unique count, and
    (for numeric columns) basic stats. Useful as a first look before
    deciding what to clean.
    """
    rows = []
    for col in df.columns:
        series = df[col]
        row = {
            "column": col,
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_pct": round(100 * series.isna().mean(), 1),
            "unique_count": int(series.nunique(dropna=True)),
            "is_constant": series.nunique(dropna=True) <= 1,
        }
        if pd.api.types.is_numeric_dtype(series):
            row["min"] = series.min()
            row["max"] = series.max()
            row["mean"] = round(series.mean(), 2) if series.notna().any() else None
        else:
            row["min"] = None
            row["max"] = None
            row["mean"] = None
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# Column name / text cleanup
# --------------------------------------------------------------------

def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """snake_case, lowercase, no leading/trailing whitespace, no repeats."""
    def clean_name(name: str) -> str:
        name = str(name).strip().lower()
        name = re.sub(r"[^\w]+", "_", name)
        name = re.sub(r"_+", "_", name).strip("_")
        return name or "column"

    new_cols = [clean_name(c) for c in df.columns]
    seen = {}
    final_cols = []
    for c in new_cols:
        if c in seen:
            seen[c] += 1
            final_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            final_cols.append(c)

    out = df.copy()
    out.columns = final_cols
    return out


def trim_whitespace(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        mask_na = df[col].isna()
        out[col] = out[col].astype(str).str.strip()
        out.loc[mask_na, col] = np.nan
    return out


def standardize_case(df: pd.DataFrame, columns: list, case: str = "lower") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        mask_na = df[col].isna()
        if case == "lower":
            out[col] = out[col].astype(str).str.lower()
        elif case == "upper":
            out[col] = out[col].astype(str).str.upper()
        elif case == "title":
            out[col] = out[col].astype(str).str.title()
        out.loc[mask_na, col] = np.nan
    return out


# --------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------

def handle_missing_numeric(df: pd.DataFrame, columns: list, strategy: str = "median") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if strategy == "median":
            out[col] = out[col].fillna(out[col].median())
        elif strategy == "mean":
            out[col] = out[col].fillna(out[col].mean())
        elif strategy == "zero":
            out[col] = out[col].fillna(0)
        elif strategy == "drop":
            out = out.dropna(subset=[col])
    return out


def handle_missing_categorical(df: pd.DataFrame, columns: list, strategy: str = "mode") -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if strategy == "mode":
            mode_vals = out[col].mode(dropna=True)
            fill_value = mode_vals.iloc[0] if not mode_vals.empty else "unknown"
            out[col] = out[col].fillna(fill_value)
        elif strategy == "unknown":
            out[col] = out[col].fillna("unknown")
        elif strategy == "drop":
            out = out.dropna(subset=[col])
    return out


# --------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------

def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> pd.Series:
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - multiplier * iqr, q3 + multiplier * iqr
    return (series < lower) | (series > upper)


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    mean, std = series.mean(), series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(False, index=series.index)
    return ((series - mean).abs() / std) > threshold


# --------------------------------------------------------------------
# Structural cleanup
# --------------------------------------------------------------------

def drop_duplicate_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    before = len(df)
    out = df.drop_duplicates()
    return out, before - len(out)


def drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    return df.drop(columns=constant_cols), constant_cols


def try_convert_to_datetime(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, list]:
    """Attempt to parse each given column as a date. Reports which succeeded."""
    out = df.copy()
    converted = []
    for col in columns:
        try:
            parsed = pd.to_datetime(out[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                out[col] = parsed
                converted.append(col)
        except Exception:
            continue
    return out, converted


# --------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    numeric_columns: list,
    text_columns: list,
    standardize_names: bool = True,
    trim_text: bool = True,
    text_case: str = "none",
    drop_dupes: bool = True,
    drop_constants: bool = True,
    numeric_missing_strategy: str = "median",
    text_missing_strategy: str = "mode",
    outlier_method: str = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Runs the full cleaning sequence and returns (clean_df, removed_outliers_df, report).
    """
    report = {"original_rows": len(df), "original_columns": len(df.columns)}
    working = df.copy()

    if standardize_names:
        working = standardize_column_names(working)
        name_map = dict(zip(df.columns, working.columns))
        numeric_columns = [name_map.get(c, c) for c in numeric_columns]
        text_columns = [name_map.get(c, c) for c in text_columns]

    if trim_text and text_columns:
        working = trim_whitespace(working, text_columns)
    if text_case != "none" and text_columns:
        working = standardize_case(working, text_columns, text_case)

    if drop_dupes:
        working, dupes_removed = drop_duplicate_rows(working)
    else:
        dupes_removed = 0
    report["duplicate_rows_removed"] = dupes_removed

    if drop_constants:
        working, constant_cols = drop_constant_columns(working)
        report["constant_columns_removed"] = constant_cols
        numeric_columns = [c for c in numeric_columns if c not in constant_cols]
        text_columns = [c for c in text_columns if c not in constant_cols]
    else:
        report["constant_columns_removed"] = []

    missing_report = {c: int(working[c].isna().sum()) for c in numeric_columns + text_columns}
    if numeric_columns:
        working = handle_missing_numeric(working, numeric_columns, numeric_missing_strategy)
    if text_columns:
        working = handle_missing_categorical(working, text_columns, text_missing_strategy)
    report["missing_values_filled"] = missing_report

    outlier_mask = pd.Series(False, index=working.index)
    outliers_by_column = {}
    for col in numeric_columns:
        if col not in working.columns:
            continue
        mask = (
            detect_outliers_iqr(working[col], iqr_multiplier)
            if outlier_method == "iqr"
            else detect_outliers_zscore(working[col], zscore_threshold)
        )
        outliers_by_column[col] = int(mask.sum())
        outlier_mask |= mask

    removed_rows = working[outlier_mask]
    clean = working[~outlier_mask]

    report["outliers_by_column"] = outliers_by_column
    report["outlier_rows_removed"] = int(outlier_mask.sum())
    report["final_rows"] = len(clean)
    report["final_columns"] = len(clean.columns)

    return clean, removed_rows, report
