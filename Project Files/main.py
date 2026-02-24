import os
import glob
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

RAW_DIR = "raw"
OUT_DIR = "output"

os.makedirs(OUT_DIR, exist_ok=True)

def clean_numeric(series):
    s = series.astype(str)
    s = s.str.replace("₹","",regex=False)
    s = s.str.replace(",","",regex=False)
    s = s.str.replace("cr","",case=False,regex=False)
    s = s.str.replace("crore","",case=False,regex=False)
    return pd.to_numeric(s, errors="coerce")

def find_column(df, keys):
    for c in df.columns:
        name = c.lower()
        if any(k in name for k in keys):
            return c
    return None

def parse_year(val):
    if pd.isna(val):
        return None
    for y in ["2021","2022","2023","2024"]:
        if y in str(val):
            return int(y)
    return None

def process_file(file):
    df = pd.read_csv(file)

    year_col = find_column(df, ["year","fy","fiscal"])
    sector_col = find_column(df, ["sector"])
    ministry_col = find_column(df, ["ministry","department"])

    # best numeric column for allocation
    best_col = max(df.columns, key=lambda c: clean_numeric(df[c]).notna().sum())
    alloc = clean_numeric(df[best_col])

    out = pd.DataFrame()
    out["Year"] = df[year_col].apply(parse_year) if year_col else None
    out["Sector"] = df[sector_col] if sector_col else None
    out["Ministry"] = df[ministry_col] if ministry_col else None
    out["Allocation_Cr"] = alloc

    out = out.dropna(subset=["Year","Allocation_Cr"])
    return out

# ---------- Load all CSVs ----------
files = glob.glob(os.path.join(RAW_DIR, "*.csv"))

all_data = []

for f in files:
    all_data.append(process_file(f))

combined = pd.concat(all_data, ignore_index=True)

# Save cleaned data for Tableau
combined.to_csv("output/union_budget_clean.csv", index=False)

# ---------- Forecast sector wise ----------
sector_year = combined.groupby(["Sector","Year"])["Allocation_Cr"].sum().reset_index()

predictions = []

for sector, sdf in sector_year.groupby("Sector"):
    sdf = sdf.sort_values("Year")
    y = sdf["Allocation_Cr"].values

    next_year = sdf["Year"].max() + 1
    pred = y[-1]

    try:
        model = ARIMA(y, order=(1,0,0))
        fit = model.fit()
        pred = fit.forecast()[0]
    except:
        pass

    predictions.append([sector, next_year, round(pred,2)])

forecast_df = pd.DataFrame(predictions, columns=["Sector","Next_Year","Predicted_Allocation_Cr"])
forecast_df.to_csv("output/sector_forecast.csv", index=False)

print("✅ DONE!")
print("Check output folder for CSV files")
