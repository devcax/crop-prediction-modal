import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

DATA_PATH = "train.csv"
MODEL_PATH = "xgb_tomorrow_price.pkl"

# --- Load ---
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().upper() for c in df.columns]

# Parse date
df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")

# Drop bad rows
df = df.dropna(subset=["DATE", "ITEM_ID", "MARKET_ID", "TODAYS_PRICE"]).copy()

# Sort oldest -> newest per item+market
df = df.sort_values(["ITEM_ID", "MARKET_ID", "DATE"]).reset_index(drop=True)

# --- Lag features ---
df["LAG_1"] = df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"].shift(1)
df["LAG_2"] = df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"].shift(2)
df["LAG_7"] = df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"].shift(7)
df["ROLL_MEAN_7"] = (
    df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"]
      .shift(1)
      .rolling(7)
      .mean()
)

# --- Tomorrow target ---
df["TOMORROWS_PRICE"] = df.groupby(["ITEM_ID", "MARKET_ID"])["TODAYS_PRICE"].shift(-1)
df = df.dropna(subset=["TOMORROWS_PRICE"]).copy()

# --- Date features ---
df["DOW"] = df["DATE"].dt.dayofweek
df["MONTH"] = df["DATE"].dt.month
df["DOM"] = df["DATE"].dt.day
df["WOY"] = df["DATE"].dt.isocalendar().week.astype(int)

# --- Features ---
BASE_FEATURES = [
    "ITEM_ID",
    "MARKET_ID",
    "TODAYS_PRICE",
    "YESTERDAYS_PRICE",
    "IS_HOLIDAY",
    "FUEL_PRICE",
    "DOW",
    "MONTH",
    "DOM",
    "WOY",
]
LAG_FEATURES = ["LAG_1", "LAG_2", "LAG_7", "ROLL_MEAN_7"]
FEATURES = BASE_FEATURES + LAG_FEATURES

df = df.dropna(subset=FEATURES).copy()

# --- Build X ---
X = df[FEATURES].copy()

# One-hot encode categorical IDs
X["ITEM_ID"] = X["ITEM_ID"].astype(int).astype("category")
X["MARKET_ID"] = X["MARKET_ID"].astype(int).astype("category")
X = pd.get_dummies(X, columns=["ITEM_ID", "MARKET_ID"], drop_first=False)

# --- y (LOG SCALE) ---
y_log = np.log1p(df["TOMORROWS_PRICE"].astype(float))

# --- Time-based split ---
unique_dates = sorted(df["DATE"].unique())
split_point = int(len(unique_dates) * 0.8)
train_dates = set(unique_dates[:split_point])
val_dates = set(unique_dates[split_point:])

train_mask = df["DATE"].isin(train_dates)
val_mask = df["DATE"].isin(val_dates)

X_train, y_train = X[train_mask], y_log[train_mask]
X_val, y_val = X[val_mask], y_log[val_mask]

# --- Model ---
model = XGBRegressor(
    n_estimators=4000,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror",
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=100,
    # early_stopping_rounds=80
)

# --- Evaluate (REAL PRICE SCALE) ---
pred_log = model.predict(X_val)
pred_price = np.expm1(pred_log)

true_price = df.loc[val_mask, "TOMORROWS_PRICE"].astype(float).values

mae = mean_absolute_error(true_price, pred_price)
rmse = mean_squared_error(true_price, pred_price) ** 0.5

print("\n==== Validation (price scale) ====")
print(f"MAE : {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# --- Save bundle ---
bundle = {
    "model": model,
    "base_features": BASE_FEATURES,
    "lag_features": LAG_FEATURES,
    "feature_columns": X.columns.tolist(),
    "target_transform": "log1p",   # IMPORTANT
}
joblib.dump(bundle, MODEL_PATH)
print(f"\nSaved: {MODEL_PATH}")
