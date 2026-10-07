# %% [markdown]
# Task 3 — Professional Data Cleaning

# %%
import pandas as pd
import numpy as np
from pathlib import Path

DATA_FILE = Path("messy_dataset.csv")
OUTPUT_FILE = Path("cleaned_dataset.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError("Place the messy dataset as 'messy_dataset.csv' in this folder.")

df = pd.read_csv(DATA_FILE)
before_rows = len(df)
before_duplicates = df.duplicated().sum()
before_nulls = int(df.isnull().sum().sum())

print("Initial shape:", df.shape)
print("\nNulls per column:\n", df.isnull().sum())
print("\nDuplicate rows:", before_duplicates)
print("\nData types:\n", df.dtypes)

# %%
# Data quality report
quality = pd.DataFrame({
    "null_count": df.isnull().sum(),
    "dtype": df.dtypes.astype(str),
    "unique_values": df.nunique(dropna=True),
})
print("Data Quality Report")
print(quality)

# %%
# Missing-data strategy
# Numeric -> median (robust to outliers)
# Categorical/text -> mode
# Date-like -> forward fill where appropriate, otherwise leave unparseable values as NaT
for col in df.columns:
    if df[col].isna().sum() == 0:
        continue
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
        print(f"{col}: missing numeric values filled with median.")
    else:
        mode = df[col].mode(dropna=True)
        if not mode.empty:
            df[col] = df[col].fillna(mode.iloc[0])
            print(f"{col}: missing categorical values filled with mode.")
        else:
            df[col] = df[col].fillna("Unknown")
            print(f"{col}: no mode available; filled with 'Unknown'.")

# %%
# Remove duplicates
df = df.drop_duplicates().copy()
print("Duplicates removed:", before_duplicates)

# %%
# Standardise common text columns
for col in df.select_dtypes(include=["object", "string"]).columns:
    s = df[col].astype("string").str.strip()
    # Standardise common gender values
    if "gender" in col.lower():
        mapping = {
            "m": "Male", "male": "Male", "man": "Male",
            "f": "Female", "female": "Female", "woman": "Female"
        }
        s_lower = s.str.lower()
        df[col] = s_lower.map(mapping).fillna(s)
    else:
        df[col] = s

# Try converting columns containing date/time names to datetime
for col in df.columns:
    if any(token in col.lower() for token in ["date", "time"]):
        converted = pd.to_datetime(df[col], errors="coerce")
        if converted.notna().mean() >= 0.7:
            df[col] = converted

# Try converting numeric-looking object columns
for col in df.select_dtypes(include=["object", "string"]).columns:
    converted = pd.to_numeric(
        df[col].astype("string").str.replace(",", "", regex=False),
        errors="coerce"
    )
    if converted.notna().mean() >= 0.8:
        df[col] = converted

# %%
# Outlier detection using IQR
outlier_report = []
for col in df.select_dtypes(include=np.number).columns:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        outlier_count = 0
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = int(((df[col] < lower) | (df[col] > upper)).sum())
    outlier_report.append([col, outlier_count])

outlier_report = pd.DataFrame(outlier_report, columns=["column", "outlier_count"])
print("IQR Outlier Report:")
print(outlier_report)

# Decision: retain outliers unless there is evidence they are data-entry errors.
# This preserves legitimate extreme observations for analysis.
print("\nOutlier decision: detected outliers are retained because the dataset context is unknown; "
      "they should only be removed/capped when confirmed as erroneous.")

# %%
# ID columns as strings where identifiable
for col in df.columns:
    if any(token in col.lower() for token in ["id", "code"]):
        df[col] = df[col].astype("string")

# %%
# Before vs after summary
after_rows = len(df)
after_duplicates = df.duplicated().sum()
after_nulls = int(df.isnull().sum().sum())

before_after = pd.DataFrame({
    "Metric": ["Row count", "Null cells", "Duplicate rows"],
    "Before": [before_rows, before_nulls, before_duplicates],
    "After": [after_rows, after_nulls, after_duplicates]
})
print("\nBefore vs After Cleaning:")
print(before_after)

print("\nFinal dtypes:")
print(df.dtypes)

# %%
# Save cleaned dataset
df.to_csv(OUTPUT_FILE, index=False)
print(f"\nCleaned dataset saved to: {OUTPUT_FILE.resolve()}")
