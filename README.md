# Data cleaner

A web app: upload any CSV, it profiles the data, cleans it, and gives you
a downloadable clean file. Everything runs locally in your browser via
Streamlit — no data leaves your machine.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Your browser opens automatically to `http://localhost:8501`.

## Try it immediately

A ready-made messy sample file is included: `sample_messy_data.csv`.
It has messy column names, extra whitespace, inconsistent casing
(`"new york"` vs `"NEW YORK"`), a few planted outliers, missing values,
duplicate rows, and a useless constant column — upload it and try every
option to see what each one catches.

## What it does

**Tab 1 — Profile**: shows dtypes, missing %, unique counts, and flags
obvious problems (duplicates, constant columns, missing data) before
you touch anything.

**Tab 2 — Clean**: pick which cleaning steps to apply:
- Standardize column names to `snake_case`
- Trim whitespace / fix inconsistent casing in text columns
- Drop exact duplicate rows
- Drop constant columns (same value in every row — no information)
- Fill missing numeric values (median / mean / zero / drop)
- Fill missing text values (most common value / "unknown" / drop)
- Detect and remove outliers, via IQR or z-score, with an adjustable
  strictness slider

**Tab 3 — Results**: shows exactly what changed (row/column counts,
per-column outlier counts, missing values filled), a before/after view
of a chosen numeric column, the cleaned data itself, and download
buttons for both the cleaned data and the rows that got removed (so
you can double check nothing important was thrown away).



- Add Excel support: `pd.read_excel()` instead of / alongside `read_csv`
- Add more outlier methods (e.g. isolation forest) in `cleaning.py`
- Add column-level renaming controls in the UI
- Persist the last-used settings with `st.session_state`
