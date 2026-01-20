import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "train.csv"
MODEL_PATH = "xgb_tomorrow_price.pkl"

# ---------- Load model bundle ----------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
BASE_FEATURES = bundle["base_features"]
LAG_FEATURES = bundle["lag_features"]
FEATURE_COLS = bundle["feature_columns"]
TARGET_TRANSFORM = bundle.get("target_transform", "none")  # "log1p" or "none"

FEATURES_NEEDED = BASE_FEATURES + LAG_FEATURES

# ---------- Load data ----------
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().upper() for c in df.columns]
df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")

df = df.dropna(subset=["DATE", "ITEM_ID", "MARKET_ID", "TODAYS_PRICE"]).copy()
df = df.sort_values(["ITEM_ID", "MARKET_ID", "DATE"]).reset_index(drop=True)

# ---------- Build the same features as training ----------
# lags
df["LAG_1"] = df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"].shift(1)
df["LAG_2"] = df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"].shift(2)
df["LAG_7"] = df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"].shift(7)
df["ROLL_MEAN_7"] = (
    df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"]
      .shift(1)
      .rolling(7)
      .mean()
)

# tomorrow target
df["TOMORROWS_PRICE"] = df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"].shift(-1)

# date features
df["DOW"] = df["DATE"].dt.dayofweek
df["MONTH"] = df["DATE"].dt.month
df["DOM"] = df["DATE"].dt.day
df["WOY"] = df["DATE"].dt.isocalendar().week.astype(int)

# keep rows where we can evaluate
df = df.dropna(subset=FEATURES_NEEDED + ["TOMORROWS_PRICE"]).copy()

# ---------- Time-based split (last 20% dates as backtest) ----------
unique_dates = sorted(df["DATE"].unique())
split_point = int(len(unique_dates) * 0.8)
val_dates = set(unique_dates[split_point:])

val_df = df[df["DATE"].isin(val_dates)].copy()
print(f"Backtest rows: {len(val_df)} | Dates in backtest: {len(val_dates)}")
print(f"Target transform in model: {TARGET_TRANSFORM}")

# ---------- Prepare X exactly like training ----------
X_val = val_df[FEATURES_NEEDED].copy()

# one-hot encode IDs
X_val["ITEM_ID"] = X_val["ITEM_ID"].astype(int).astype("category")
X_val["MARKET_ID"] = X_val["MARKET_ID"].astype(int).astype("category")
X_val = pd.get_dummies(X_val, columns=["ITEM_ID", "MARKET_ID"], drop_first=False)

# align to training columns
X_val = X_val.reindex(columns=FEATURE_COLS, fill_value=0)

# ---------- Predict ----------
y_true = val_df["TOMORROWS_PRICE"].astype(float).values
y_pred_raw = model.predict(X_val)

# If model trained on log1p(target), invert it
if TARGET_TRANSFORM == "log1p":
    y_pred = np.expm1(y_pred_raw)
else:
    y_pred = y_pred_raw

# ---------- Metrics (price scale) ----------
mae = mean_absolute_error(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5

print("\n==== Backtest Metrics (price scale, last 20% dates) ====")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Optional: normal-range metrics (so spikes don't dominate)
normal_mask = y_true <= 1000
if normal_mask.any():
    mae_normal = mean_absolute_error(y_true[normal_mask], y_pred[normal_mask])
    rmse_normal = mean_squared_error(y_true[normal_mask], y_pred[normal_mask]) ** 0.5
    print("\n==== Backtest Metrics (price <= 1000) ====")
    print(f"MAE : {mae_normal:.2f}")
    print(f"RMSE: {rmse_normal:.2f}")
    print(f"Rows <= 1000: {normal_mask.mean() * 100:.1f}%")

# ---------- Show examples ----------
out = val_df[["DATE", "ITEM_ID", "MARKET_ID", "TODAYS_PRICE", "TOMORROWS_PRICE"]].copy()
out["PRED_TOMORROW"] = y_pred
out["ABS_ERR"] = (out["PRED_TOMORROW"] - out["TOMORROWS_PRICE"]).abs()

print("\n---- Worst 20 predictions ----")
print(out.sort_values("ABS_ERR", ascending=False).head(20).to_string(index=False))

print("\n---- Sample 20 predictions ----")
print(out.sample(min(20, len(out)), random_state=42).sort_values("DATE").to_string(index=False))
