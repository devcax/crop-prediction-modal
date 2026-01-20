import joblib
import pandas as pd
import numpy as np
from datetime import timedelta
from math import radians, sin, cos, sqrt, atan2

DATA_PATH = "train.csv"
MODEL_PATH = "xgb_tomorrow_price.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
BASE_FEATURES = bundle["base_features"]
LAG_FEATURES = bundle["lag_features"]
FEATURE_COLS = bundle["feature_columns"]
TARGET_TRANSFORM = bundle.get("target_transform", "none")
FEATURES_NEEDED = BASE_FEATURES + LAG_FEATURES

# Market registry (IDs you use in training)
# Coordinates are approximate city/market coords:
# Pettah ~ 6.9376, 79.8491 (Colombo/Pettah)  :contentReference[oaicite:0]{index=0}
# Dambulla ~ 7.866239, 80.651695 (Dambulla Dedicated Economic Centre) :contentReference[oaicite:1]{index=1}
MARKETS = {
    0: {"name": "Pettah", "lat": 6.9376, "lon": 79.8491},
    1: {"name": "Dambulla", "lat": 7.866239, "lon": 80.651695},
}

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def _load_history(item_id: int, market_id: int):
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")

    hist = df[(df["ITEM_ID"] == item_id) & (df["MARKET_ID"] == market_id)] \
        .dropna(subset=["DATE", "TODAYS_PRICE"]) \
        .sort_values("DATE") \
        .copy()

    if len(hist) < 3:
        raise ValueError(f"Not enough history for ITEM {item_id}, MARKET {market_id}. Need >=3 rows, got {len(hist)}.")

    return hist

def _build_features_from_history(hist: pd.DataFrame, today_dt: pd.Timestamp):
    """
    Uses the latest available row in CSV as the reference 'current' state.
    If today_dt is beyond CSV latest date, we warn and use last available day as 'today state'.
    """
    last_date_in_csv = hist["DATE"].max()
    gap_days = (today_dt - last_date_in_csv).days

    warnings = []
    if gap_days > 0:
        warnings.append(
            f"CSV is {gap_days} day(s) behind for this item+market (latest: {last_date_in_csv.date()}). Using latest available data."
        )

    # Use history up to its latest point (cannot invent real prices)
    hist2 = hist[hist["DATE"] <= last_date_in_csv].copy()

    # Compute lags
    hist2["LAG_1"] = hist2["TODAYS_PRICE"].shift(1)
    hist2["LAG_2"] = hist2["TODAYS_PRICE"].shift(2)
    hist2["LAG_7"] = hist2["TODAYS_PRICE"].shift(7)
    hist2["ROLL_MEAN_7"] = hist2["TODAYS_PRICE"].shift(1).rolling(7).mean()

    last = hist2.iloc[-1]
    todays_price = float(last["TODAYS_PRICE"])
    yesterdays_price = float(hist2.iloc[-2]["TODAYS_PRICE"]) if len(hist2) >= 2 else float(last["TODAYS_PRICE"])

    lag_1 = last["LAG_1"]
    lag_2 = last["LAG_2"]
    lag_7 = last["LAG_7"]
    roll_mean_7 = last["ROLL_MEAN_7"]

    # Graceful fallback for missing
    if pd.isna(lag_1):
        lag_1 = yesterdays_price
        warnings.append("LAG_1 missing → used yesterdays_price")
    if pd.isna(lag_2):
        lag_2 = float(lag_1)
        warnings.append("LAG_2 missing → used LAG_1")
    if pd.isna(lag_7):
        lag_7 = float(hist2["TODAYS_PRICE"].mean())
        warnings.append("LAG_7 missing → used historical mean")
    if pd.isna(roll_mean_7):
        roll_mean_7 = float(hist2["TODAYS_PRICE"].tail(3).mean())
        warnings.append("ROLL_MEAN_7 missing → used recent mean")

    return {
        "todays_price": todays_price,
        "yesterdays_price": yesterdays_price,
        "lag_1": float(lag_1),
        "lag_2": float(lag_2),
        "lag_7": float(lag_7),
        "roll_mean_7": float(roll_mean_7),
        "warnings": warnings,
        "last_date_in_csv": last_date_in_csv,
    }

def _predict_one_step(item_id, market_id, today_dt, is_holiday, fuel_price, state):
    row = {
        "DATE": today_dt,
        "ITEM_ID": item_id,
        "MARKET_ID": market_id,
        "TODAYS_PRICE": state["todays_price"],
        "YESTERDAYS_PRICE": state["yesterdays_price"],
        "IS_HOLIDAY": int(is_holiday),
        "FUEL_PRICE": float(fuel_price),
        "LAG_1": state["lag_1"],
        "LAG_2": state["lag_2"],
        "LAG_7": state["lag_7"],
        "ROLL_MEAN_7": state["roll_mean_7"],
    }
    df_in = pd.DataFrame([row])

    # date features
    df_in["DOW"] = df_in["DATE"].dt.dayofweek
    df_in["MONTH"] = df_in["DATE"].dt.month
    df_in["DOM"] = df_in["DATE"].dt.day
    df_in["WOY"] = df_in["DATE"].dt.isocalendar().week.astype(int)

    X = df_in[FEATURES_NEEDED].copy()
    X["ITEM_ID"] = X["ITEM_ID"].astype(int).astype("category")
    X["MARKET_ID"] = X["MARKET_ID"].astype(int).astype("category")
    X = pd.get_dummies(X, columns=["ITEM_ID", "MARKET_ID"], drop_first=False)
    X = X.reindex(columns=FEATURE_COLS, fill_value=0)

    pred_raw = model.predict(X)[0]
    if TARGET_TRANSFORM == "log1p":
        return float(np.expm1(pred_raw))
    return float(pred_raw)

def predict_market_price(item_id: int, market_id: int, start_date: str, horizon_days: int,
                         is_holiday: int, fuel_price: float):
    """
    Returns predicted price for (start_date + horizon_days) using chained forecasts.
    Uses CSV to auto-build lags + today/yesterday for that market.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if horizon_days > 30:
        raise ValueError("horizon_days cannot exceed 30")

    today_dt = pd.to_datetime(start_date, dayfirst=True, errors="coerce")
    if pd.isna(today_dt):
        raise ValueError("Invalid start_date")

    hist = _load_history(item_id, market_id)
    state = _build_features_from_history(hist, today_dt)

    # chaining
    current_date = today_dt
    current_price = state["todays_price"]
    prev_price = state["yesterdays_price"]

    # We'll update state lags from the evolving chain values:
    # simplest: treat chain as: today's_price becomes predicted, and lags shift accordingly.
    # (If you want full recomputation of roll mean using a buffer, we can do that too.)
    warnings = list(state["warnings"])

    # Keep a small buffer of last prices to support roll mean in chain
    buffer_prices = list(hist["TODAYS_PRICE"].tail(7).astype(float).values)
    if len(buffer_prices) < 7:
        # pad if needed
        buffer_prices = ([float(np.mean(buffer_prices))] * (7 - len(buffer_prices))) + buffer_prices

    pred_price = None
    for _ in range(horizon_days):
        # refresh lags from buffer for chaining
        lag_1 = buffer_prices[-1]
        lag_2 = buffer_prices[-2]
        lag_7 = buffer_prices[0]
        roll_mean_7 = float(np.mean(buffer_prices))

        chain_state = {
            "todays_price": float(current_price),
            "yesterdays_price": float(prev_price),
            "lag_1": float(lag_1),
            "lag_2": float(lag_2),
            "lag_7": float(lag_7),
            "roll_mean_7": float(roll_mean_7),
        }

        pred_price = _predict_one_step(item_id, market_id, current_date, is_holiday, fuel_price, chain_state)

        # roll forward
        next_date = current_date + timedelta(days=1)
        prev_price = current_price
        current_price = pred_price
        current_date = next_date

        # update buffer (drop oldest, add newest predicted)
        buffer_prices.pop(0)
        buffer_prices.append(float(pred_price))

    target_date = (today_dt + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    return {
        "market_id": market_id,
        "target_date": target_date,
        "predicted_price": float(pred_price),
        "warnings": warnings,
    }

def recommend_best_market(user_lat: float, user_lon: float, item_id: int,
                          start_date: str, horizon_days: int,
                          is_holiday: int, fuel_price: float,
                          transport_cost_per_km: float = 0.0):
    """
    transport_cost_per_km: optional (LKR per km per unit). If 0, recommendation is based on price only.
    """
    results = []

    for market_id, info in MARKETS.items():
        # distance
        dist_km = haversine_km(user_lat, user_lon, info["lat"], info["lon"])

        # predict
        pred = predict_market_price(
            item_id=item_id,
            market_id=market_id,
            start_date=start_date,
            horizon_days=horizon_days,
            is_holiday=is_holiday,
            fuel_price=fuel_price,
        )

        predicted_price = pred["predicted_price"]
        transport_cost = dist_km * float(transport_cost_per_km)
        net_est = predicted_price - transport_cost

        results.append({
            "market_id": market_id,
            "market_name": info["name"],
            "distance_km": round(dist_km, 1),
            "predicted_price": round(predicted_price, 2),
            "transport_cost_est": round(transport_cost, 2),
            "net_est_price": round(net_est, 2),
            "warnings": pred["warnings"],
            "target_date": pred["target_date"]
        })

    # Decide best
    if transport_cost_per_km > 0:
        best = max(results, key=lambda x: x["net_est_price"])
        basis = "net (price - transport estimate)"
    else:
        best = max(results, key=lambda x: x["predicted_price"])
        basis = "predicted price (transport not included)"

    # Dynamic short response
    summary = (
        f"Prediction for {results[0]['target_date']} (Item {item_id}, horizon {horizon_days} day(s)):\n"
        f"- {results[0]['market_name']}: Rs {results[0]['predicted_price']} (distance {results[0]['distance_km']} km)\n"
        f"- {results[1]['market_name']}: Rs {results[1]['predicted_price']} (distance {results[1]['distance_km']} km)\n\n"
        f"✅ Best market to sell (based on {basis}): **{best['market_name']}**\n"
    )

    # Collect warnings (stale CSV etc.)
    all_warn = []
    for r in results:
        for w in r["warnings"]:
            all_warn.append(f"{r['market_name']}: {w}")
    all_warn = list(dict.fromkeys(all_warn))  # unique preserve order

    return {
        "target_date": results[0]["target_date"],
        "item_id": item_id,
        "horizon_days": horizon_days,
        "basis": basis,
        "markets": results,
        "best_market": best,
        "message": summary,
        "warnings": all_warn
    }


if __name__ == "__main__":
    resp = recommend_best_market(
        user_lat=6.9271,  # user GPS
        user_lon=79.8612,
        item_id=1,        # crop id
        start_date="2026-03-01",
        horizon_days=1,   # tomorrow
        is_holiday=0,
        fuel_price=340,
        transport_cost_per_km=2.0  # optional (set 0 if you don’t want cost)
    )

    print(resp["message"])
    print("Warnings:", resp["warnings"])
    print(resp["markets"])
