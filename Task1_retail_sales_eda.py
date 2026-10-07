# %% [markdown]
# Task 1 — EDA on Retail Sales Data

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid")
DATA_FILE = Path("retail_sales.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError("Place the retail dataset as 'retail_sales.csv' in this folder.")

df = pd.read_csv(DATA_FILE)

print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nNull values:")
print(df.isnull().sum())

# %%
# Automatic column detection
def find_col(candidates):
    lower = {c.lower().strip(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    for actual in df.columns:
        a = actual.lower().replace("_", " ").strip()
        if any(c.lower() in a for c in candidates):
            return actual
    return None

date_col = find_col(["date", "order date", "transaction date"])
age_col = find_col(["age", "customer age"])
gender_col = find_col(["gender", "customer gender"])
product_col = find_col(["product", "product name", "item", "item name"])
category_col = find_col(["category", "product category"])
sales_col = find_col(["sales", "revenue", "amount", "total amount", "price"])

print("\nDetected columns:")
print({
    "date": date_col, "age": age_col, "gender": gender_col,
    "product": product_col, "category": category_col, "sales": sales_col
})

if date_col is None or sales_col is None:
    raise ValueError("Could not identify Date and Sales columns. Update the column mapping.")

df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
df[sales_col] = pd.to_numeric(df[sales_col], errors="coerce")

# %%
# Descriptive statistics for numerical columns
numeric_cols = df.select_dtypes(include=np.number).columns
print("Mean:\n", df[numeric_cols].mean())
print("\nMedian:\n", df[numeric_cols].median())
print("\nMode:\n", df[numeric_cols].mode().iloc[0])
print("\nStandard deviation:\n", df[numeric_cols].std())
print("\nDescribe:\n", df[numeric_cols].describe())

# %%
# Monthly and quarterly sales trends
time_df = df.dropna(subset=[date_col, sales_col]).copy()
time_df["Month"] = time_df[date_col].dt.to_period("M").astype(str)
time_df["Quarter"] = time_df[date_col].dt.to_period("Q").astype(str)

monthly = time_df.groupby("Month")[sales_col].sum()
quarterly = time_df.groupby("Quarter")[sales_col].sum()

plt.figure(figsize=(12,5))
monthly.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Observation: Monthly sales show the periods of higher and lower demand.")

plt.figure(figsize=(10,5))
quarterly.plot(kind="line", marker="o")
plt.title("Quarterly Sales Trend")
plt.xlabel("Quarter")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
print("Observation: Quarterly aggregation highlights broader seasonal sales movement.")

# %%
# Customer demographics: age groups and gender
if age_col:
    df["Age Group"] = pd.cut(
        df[age_col], bins=[0, 18, 25, 35, 45, 55, 65, 120],
        labels=["<18", "18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
        include_lowest=True
    )
    plt.figure(figsize=(9,5))
    df["Age Group"].value_counts().sort_index().plot(kind="bar")
    plt.title("Customer Age Group Distribution")
    plt.xlabel("Age Group")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.show()
    print("Observation: The chart shows the dominant customer age groups.")

if gender_col:
    plt.figure(figsize=(7,5))
    df[gender_col].value_counts(dropna=False).plot(kind="bar")
    plt.title("Gender Breakdown")
    plt.xlabel("Gender")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.show()
    print("Observation: The chart shows the distribution of customers by gender.")

# %%
# Product analysis
if product_col:
    top10 = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(10)
    plt.figure(figsize=(11,6))
    top10.sort_values().plot(kind="barh")
    plt.title("Top 10 Best-Selling Products by Revenue")
    plt.xlabel("Sales")
    plt.ylabel("Product")
    plt.tight_layout()
    plt.show()
    print("Observation: These products generate the highest sales and can be prioritized.")

if category_col:
    cat_sales = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    cat_sales.plot(kind="bar")
    plt.title("Revenue by Product Category")
    plt.xlabel("Category")
    plt.ylabel("Revenue")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    print("Observation: Categories differ in their contribution to total revenue.")

# %%
# Correlation heatmap
corr = df[numeric_cols].corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix of Numerical Variables")
plt.tight_layout()
plt.show()
print("Observation: Strong positive/negative correlations identify variables that move together.")

# %%
# Additional non-obvious insight: average sales by weekday
time_df["Weekday"] = time_df[date_col].dt.day_name()
weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
weekday_sales = time_df.groupby("Weekday")[sales_col].mean().reindex(weekday_order)

plt.figure(figsize=(10,5))
weekday_sales.plot(kind="bar")
plt.title("Average Sales by Day of Week")
plt.xlabel("Day")
plt.ylabel("Average Sales")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()
print("Observation: Weekday-level averages can reveal demand patterns that monthly totals hide.")

# %%
# Actionable recommendations
recommendations = [
    "Prioritize inventory and promotions around months/quarters with consistently high demand.",
    "Use the top-selling products and strongest categories in targeted campaigns and cross-selling bundles.",
    "Use customer demographic patterns to tailor promotions, messaging, and product recommendations."
]
print("Actionable Business Recommendations:")
for i, r in enumerate(recommendations, 1):
    print(f"{i}. {r}")
