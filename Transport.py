import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------- CONSTANTS ----------------
STUDENT_ID = "24174419"
FN1 = "2019data1.csv"
FN2 = "2022data1.csv"

# ---------------- Helper utilities ----------------
def locate_file(fn):
    """Try CWD then script directory (__file__), raise if not found."""
    if os.path.exists(fn):
        return fn
    try:
        base_dir = os.path.dirname(__file__) or os.getcwd()
    except NameError:
        base_dir = os.getcwd()
    p = os.path.join(base_dir, fn)
    if os.path.exists(p):
        return p
    raise FileNotFoundError(f"Could not find {fn} in CWD or script dir ({base_dir}).")

def find_column_case_insensitive(cols, tokens, require_all=False):
    for c in cols:
        name = c.lower()
        if require_all:
            if all(tok.lower() in name for tok in tokens):
                return c
        else:
            if any(tok.lower() in name for tok in tokens):
                return c
    return None

def find_columns_with_any_tokens(cols, tokens):
    return [c for c in cols if any(tok.lower() in c.lower() for tok in tokens)]

def is_numeric_series(s):
    return pd.api.types.is_numeric_dtype(s)

# ---------------- Load files ----------------
try:
    fn1 = locate_file(FN1)
    fn2 = locate_file(FN2)
    df19 = pd.read_csv(fn1)
    df22 = pd.read_csv(fn2)
    print("Loaded files:", fn1, fn2)
except FileNotFoundError as e:
    print(e)
    # Creating dummy data for demonstration if files are missing
    df19 = pd.DataFrame(columns=['Date', 'pax_bus', 'pax_metro', 'bus_rev', 'metro_rev'])
    df22 = pd.DataFrame(columns=['date', 'distance', 'duration', 'price', 'mode'])

# ---------------- Normalise date column in df22 ----------------
date_col_22 = find_column_case_insensitive(df22.columns, ["date"])
if date_col_22 is None:
    date_col_22 = find_column_case_insensitive(df22.columns, ["time", "timestamp", "datetime"])

if date_col_22:
    df22 = df22.rename(columns={date_col_22: "date"})
    df22['date'] = pd.to_datetime(df22['date'], errors='coerce')

# ---------------- Normalise numeric columns ----------------
distance_col_22 = find_column_case_insensitive(df22.columns, ["dist", "distance", "length", "km"])
duration_col_22 = find_column_case_insensitive(df22.columns, ["dur", "duration", "minutes", "mins", "travel_time"])
price_col_22 = find_column_case_insensitive(df22.columns, ["price", "fare", "ticket_price", "cost"])
mode_col_22 = find_column_case_insensitive(df22.columns, ["mode", "transport", "vehicle", "type"])

if distance_col_22:
    df22 = df22.rename(columns={distance_col_22: "distance"})
if duration_col_22:
    df22 = df22.rename(columns={duration_col_22: "duration"})
if price_col_22:
    df22 = df22.rename(columns={price_col_22: "price"})
if mode_col_22:
    df22 = df22.rename(columns={mode_col_22: "mode"})

# Cleaning and converting
if 'distance' in df22.columns:
    df22['distance'] = pd.to_numeric(df22['distance'], errors='coerce')
    df22['distance_km'] = np.where(df22['distance'] > 1000, df22['distance'] / 1000.0, df22['distance'])

if 'duration' in df22.columns:
    df22['duration'] = pd.to_numeric(df22['duration'], errors='coerce')
    df22['duration_min'] = np.where(df22['duration'] > 3600, df22['duration'] / 60.0, df22['duration'])

if 'price' in df22.columns:
    df22['price'] = pd.to_numeric(df22['price'], errors='coerce')

# ---------------- Prepare df19 ----------------
pax_candidates = find_columns_with_any_tokens(df19.columns, ["pax", "pass", "passengers", "n_pass"])
if pax_candidates:
    for c in pax_candidates:
        df19[c] = pd.to_numeric(df19[c], errors='coerce').fillna(0.0)
    df19['total_pax'] = df19[pax_candidates].sum(axis=1)

date_col_19 = find_column_case_insensitive(df19.columns, ["date"])
if date_col_19 is None:
    date_col_19 = find_column_case_insensitive(df19.columns, ["time", "day"])

if date_col_19:
    df19[date_col_19] = pd.to_datetime(df19[date_col_19], errors='coerce')
    df19['day_of_year'] = df19[date_col_19].dt.dayofyear

# ---------------- Build daily series ----------------
def build_daily_series_2019(df):
    N = 365
    days = np.arange(1, N + 1)
    s = pd.Series(0.0, index=days)
    if 'day_of_year' in df.columns:
        agg = df.groupby('day_of_year')['total_pax'].sum()
        s.update(agg)
    return s

def build_daily_series_2022(df):
    N = 365
    days = np.arange(1, N + 1)
    s = pd.Series(0.0, index=days)
    if 'date' in df.columns:
        df['doy'] = df['date'].dt.dayofyear
        agg = df.groupby('doy').size()
        s.update(agg)
    return s

daily_19 = build_daily_series_2019(df19)
daily_22 = build_daily_series_2022(df22)

daily_19_interp = daily_19.replace(0, np.nan).interpolate(method='linear').fillna(0.0)
daily_22_interp = daily_22.replace(0, np.nan).interpolate(method='linear').fillna(0.0)

# ---------------- Fourier smoothing ----------------
def smooth_fourier(y, terms=8):
    y = np.asarray(y, dtype=float)
    N = len(y)
    Y = np.fft.fft(y)
    Yf = np.zeros_like(Y)
    Yf[0] = Y[0]
    upper = min(terms, N//2 - 1)
    Yf[1:upper+1] = Y[1:upper+1]
    Yf[-upper:] = Y[-upper:]
    return np.fft.ifft(Yf).real

smooth19 = smooth_fourier(daily_19_interp, 8)
smooth22 = smooth_fourier(daily_22_interp, 8)

# ---------------- FIGURE 1 / 2 ----------------
plt.figure(figsize=(12, 6))
days = np.arange(1, 366)
plt.scatter(days, daily_19, s=6, alpha=0.6, label="2019 daily")
plt.scatter(days, daily_22, s=6, alpha=0.6, label="2022 daily")
plt.plot(days, smooth19, linewidth=2, label="2019 smoothed")
plt.plot(days, smooth22, linewidth=2, label="2022 smoothed")
plt.title(f"Daily Passenger Numbers ID {STUDENT_ID}")
plt.legend()
plt.grid(axis='y', alpha=0.3)

def sum_revenue_columns(df, tokens):
    token_list = [tok.lower() for tok in tokens]
    cols = [c for c in df.columns if any(tok in c.lower() and "rev" in c.lower() for tok in token_list)]
    if not cols: return 0.0
    return df[cols].apply(pd.to_numeric, errors='coerce').sum().sum()

X_2019 = sum_revenue_columns(df19, ["bus"])
Y_2019 = sum_revenue_columns(df19, ["tram"])
Z_2019 = sum_revenue_columns(df19, ["metro"])

txt = f"X={X_2019:.2f} EUR\nY={Y_2019:.2f} EUR\nZ={Z_2019:.2f} EUR"
plt.gca().text(0.98, 0.95, txt, transform=plt.gca().transAxes, ha='right', va='top', bbox=dict(boxstyle="round", alpha=0.2))
plt.tight_layout()
plt.savefig("Figure1.png")
plt.show()

# ---------------- FIGURE 3 ----------------
plt.figure(figsize=(10, 6))
# Example: Plotting distance vs price for 2022
if 'distance_km' in df22.columns and 'price' in df22.columns:
    plt.scatter(df22['distance_km'], df22['price'], alpha=0.5)
    plt.xlabel("Distance (km)")
    plt.ylabel("Price (EUR)")
plt.title(f"Distance vs Price Analysis ID {STUDENT_ID}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Figure3.png")
plt.show()

# ---------------- FIGURE 4 ----------------
plt.figure(figsize=(10, 6))
# Example: Duration Distribution
if 'duration_min' in df22.columns:
    df22['duration_min'].hist(bins=30, alpha=0.7)
    plt.xlabel("Duration (min)")
    plt.ylabel("Frequency")
plt.title(f"Travel Duration Distribution ID {STUDENT_ID}")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("Figure4.png")
plt.show()

# ---------------- Revenues 2022 summary ----------------
def sum_revenue_22(df, tokens):
    if 'price' not in df.columns or 'mode' not in df.columns:
        return 0.0
    mask = df['mode'].astype(str).str.contains('|'.join(tokens), case=False, na=False)
    return float(pd.to_numeric(df.loc[mask, 'price'], errors='coerce').sum())

X2 = sum_revenue_22(df22, ['bus'])
Y2 = sum_revenue_22(df22, ['tram'])
Z2 = sum_revenue_22(df22, ['metro','train','subway','underground'])

print("\n--- Revenue Summary ---")
print(f"2019 Bus Revenue (X): {X_2019}")
print(f"2022 Bus Revenue (X2): {X2}")

with open("revenues_summary.txt", "w") as f:
    f.write(f"X (2019 Bus): {X_2019:.2f}\n")
    f.write(f"Y (2019 Tram): {Y_2019:.2f}\n")
    f.write(f"Z (2019 Metro): {Z_2019:.2f}\n")
    f.write(f"X2 (2022 Bus): {X2:.2f}\n")
    f.write(f"Y2 (2022 Tram): {Y2:.2f}\n")
    f.write(f"Z2 (2022 Metro): {Z2:.2f}\n")

print("\nProcess Complete. Saved Figures and revenues_summary.txt")
