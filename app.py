"""
Data cleaning web app.

Run with:
    streamlit run app.py

Upload a CSV -> see a profile of it -> choose cleaning options -> see
exactly what changed -> download the cleaned file.
"""

import pandas as pd
import streamlit as st

from cleaning import clean_dataframe, profile_dataframe, try_convert_to_datetime

st.set_page_config(page_title="Data Cleaner", layout="wide")
st.title("Data cleaner")
st.caption("Upload a CSV. Profile it, clean it (outliers, missing values, duplicates, formatting), download the result.")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Couldn't read that file as a CSV: {exc}")
    st.stop()

if df.empty:
    st.warning("That file has no rows.")
    st.stop()

tab_profile, tab_clean, tab_results = st.tabs(["1. Profile", "2. Clean", "3. Results"])

# ------------------------------------------------------------------
# Tab 1: Profile
# ------------------------------------------------------------------
with tab_profile:
    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.caption(f"{len(df)} rows, {len(df.columns)} columns")

    st.subheader("Column profile")
    profile_df = profile_dataframe(df)
    st.dataframe(profile_df, use_container_width=True)

    problems = []
    n_dupes = df.duplicated().sum()
    if n_dupes:
        problems.append(f"{n_dupes} exact duplicate row(s)")
    n_constant = int(profile_df["is_constant"].sum())
    if n_constant:
        problems.append(f"{n_constant} constant column(s) (no useful information)")
    n_missing_cols = int((profile_df["missing_count"] > 0).sum())
    if n_missing_cols:
        problems.append(f"{n_missing_cols} column(s) with missing values")

    if problems:
        st.warning("Found: " + "; ".join(problems) + ". Head to the Clean tab to fix these.")
    else:
        st.success("No duplicates, constant columns, or missing values detected.")

# ------------------------------------------------------------------
# Tab 2: Clean
# ------------------------------------------------------------------
numeric_cols = df.select_dtypes(include="number").columns.tolist()
text_cols = df.select_dtypes(include="object").columns.tolist()

with tab_clean:
    st.subheader("Column formatting")
    c1, c2, c3 = st.columns(3)
    with c1:
        standardize_names = st.checkbox("Standardize column names (snake_case)", value=True)
    with c2:
        trim_text = st.checkbox("Trim whitespace in text columns", value=True)
    with c3:
        text_case = st.selectbox("Text case", ["none", "lower", "upper", "title"], index=0)

    st.subheader("Structural cleanup")
    c1, c2 = st.columns(2)
    with c1:
        drop_dupes = st.checkbox("Drop exact duplicate rows", value=True)
    with c2:
        drop_constants = st.checkbox("Drop constant columns (same value in every row)", value=True)

    st.subheader("Missing values")
    c1, c2 = st.columns(2)
    with c1:
        numeric_missing_strategy = st.selectbox(
            "Numeric columns: fill missing with", ["median", "mean", "zero", "drop"]
        )
    with c2:
        text_missing_strategy = st.selectbox(
            "Text columns: fill missing with", ["mode", "unknown", "drop"]
        )

    st.subheader("Outlier detection")
    selected_numeric = st.multiselect(
        "Numeric columns to check for outliers", numeric_cols, default=numeric_cols
    )
    c1, c2 = st.columns(2)
    with c1:
        outlier_method = st.radio("Method", ["iqr", "zscore"], horizontal=True)
    with c2:
        if outlier_method == "iqr":
            iqr_multiplier = st.slider("IQR multiplier (lower = stricter)", 1.0, 3.0, 1.5, 0.1)
            zscore_threshold = 3.0
        else:
            zscore_threshold = st.slider("Z-score threshold (lower = stricter)", 1.5, 5.0, 3.0, 0.1)
            iqr_multiplier = 1.5

    run_clicked = st.button("Clean data", type="primary")

# ------------------------------------------------------------------
# Tab 3: Results
# ------------------------------------------------------------------
with tab_results:
    if not run_clicked:
        st.info("Set your options in the Clean tab, then click 'Clean data'.")
        st.stop()

    clean_df, removed_df, report = clean_dataframe(
        df,
        numeric_columns=selected_numeric,
        text_columns=text_cols,
        standardize_names=standardize_names,
        trim_text=trim_text,
        text_case=text_case,
        drop_dupes=drop_dupes,
        drop_constants=drop_constants,
        numeric_missing_strategy=numeric_missing_strategy,
        text_missing_strategy=text_missing_strategy,
        outlier_method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        zscore_threshold=zscore_threshold,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", report["final_rows"], delta=report["final_rows"] - report["original_rows"])
    m2.metric("Columns", report["final_columns"], delta=report["final_columns"] - report["original_columns"])
    m3.metric("Duplicates removed", report["duplicate_rows_removed"])
    m4.metric("Outlier rows removed", report["outlier_rows_removed"])

    with st.expander("Full cleaning report"):
        st.json(report)

    if numeric_cols:
        st.subheader("Before / after distribution")
        chart_col = st.selectbox("Column to compare", [c for c in selected_numeric if c in df.columns])
        if chart_col:
            before_after = pd.DataFrame({
                "before": df[chart_col].dropna().reset_index(drop=True),
            })
            clean_col_name = chart_col.strip().lower().replace(" ", "_") if standardize_names else chart_col
            if clean_col_name in clean_df.columns:
                after_series = clean_df[clean_col_name].dropna().reset_index(drop=True)
                max_len = max(len(before_after), len(after_series))
                before_after = before_after.reindex(range(max_len))
                before_after["after"] = after_series.reindex(range(max_len))
            st.line_chart(before_after)
            st.caption("Not a time series — just before/after values side by side to eyeball the effect of cleaning.")

    st.subheader("Cleaned data")
    st.dataframe(clean_df, use_container_width=True)
    st.download_button(
        "Download cleaned CSV",
        clean_df.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )

    if not removed_df.empty:
        st.subheader("Removed outlier rows (for review)")
        st.dataframe(removed_df, use_container_width=True)
        st.download_button(
            "Download removed rows CSV",
            removed_df.to_csv(index=False).encode("utf-8"),
            file_name="removed_rows.csv",
            mime="text/csv",
        )
