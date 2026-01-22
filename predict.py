import joblib
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

DATA_PATH = "train.csv"
MODEL_PATH = "xgb_tomorrow_price.pkl"
HOLIDAY_ICS_PATH = "2026.ics"

# ---------------- Load model bundle ----------------
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
BASE_FEATURES = bundle["base_features"]
LAG_FEATURES = bundle["lag_features"]
FEATURE_COLS = bundle["feature_columns"]
TARGET_TRANSFORM = bundle.get("target_transform", "none")
FEATURES_NEEDED = BASE_FEATURES + LAG_FEATURES

# ---------------- Market registry ----------------
MARKETS = {
    0: {"name": "Pettah", "lat": 6.9376, "lon": 79.8491},
    1: {"name": "Dambulla", "lat": 7.866239, "lon": 80.651695},
}

# ---------------- Crop -> ITEM_ID mapping ----------------
# TODO: Update these IDs to match your dataset catalog exactly.
CROP_TO_ITEM_ID = {
    "beans": 1,
    "carrot": 2,
    "cabbage": 3,
    "tomato": 4,
    "brinjal": 5,
    "pumpkin": 6,
    "snake gourd": 7,
    "green chilli": 8,
    "lime": 9,
    "red onion": 10,
    "big onion": 11,
    "potato": 12,
    "red dhal": 13,
    "banana": 14,
    "papaw": 15,
    "pineapple": 16,
}

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c


def load_train_csv() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df.columns = [c.strip().upper() for c in df.columns]
    df["DATE"] = pd.to_datetime(df["DATE"], dayfirst=True, errors="coerce")
    return df


def get_latest_fuel_price(train_df: pd.DataFrame) -> float:
    # Use latest non-null fuel price in the dataset
    tmp = train_df.dropna(subset=["DATE", "FUEL_PRICE"]).sort_values("DATE")
    if tmp.empty:
        # last resort fallback
        return 0.0
    return float(tmp.iloc[-1]["FUEL_PRICE"])


def parse_ics_holidays(ics_path: str):
    """
    Parses the local .ics file and returns set of YYYY-MM-DD strings for public holidays.
    Returns (holiday_dates_set, warning_message or None)
    """
    try:
        if not Path(ics_path).exists():
            return set(), f"Holiday file not found: {ics_path}"
        
        with open(ics_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        dates = set()
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            # Parse DTSTART lines in the format: DTSTART;VALUE=DATE:YYYYMMDD
            if line.startswith('DTSTART;VALUE=DATE:'):
                date_str = line.split(':')[1]
                # Convert YYYYMMDD to YYYY-MM-DD
                if len(date_str) == 8 and date_str.isdigit():
                    formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
                    dates.add(formatted_date)
        
        if not dates:
            return set(), "No holidays found in ICS file"
        
        return dates, None
    except Exception as e:
        return set(), f"Failed to parse ICS file: {e}"


def build_is_holiday_for_dates(dates):
    """
    dates: list[pd.Timestamp]
    returns dict[YYYY-MM-DD] -> 0/1, warnings list
    """
    warnings = []
    
    # Parse the local ICS file once
    holiday_dates, warn = parse_ics_holidays(HOLIDAY_ICS_PATH)
    if warn:
        warnings.append(warn)

    out = {}
    for d in dates:
        s = d.strftime("%Y-%m-%d")
        out[s] = 1 if s in holiday_dates else 0

    # If parsing failed completely, still return zeros but warn
    if holiday_dates == set() and warnings:
        warnings.append("Holiday list empty (using IS_HOLIDAY=0 for all days)")

    return out, warnings


def _load_history(train_df: pd.DataFrame, item_id: int, market_id: int) -> pd.DataFrame:
    hist = train_df[
        (train_df["ITEM_ID"] == item_id) &
        (train_df["MARKET_ID"] == market_id)
    ].dropna(subset=["DATE", "TODAYS_PRICE"]).sort_values("DATE").copy()

    if len(hist) < 3:
        raise ValueError(f"Not enough history for ITEM {item_id}, MARKET {market_id}. Need >=3 rows, got {len(hist)}.")

    return hist


def _predict_one_step(item_id, market_id, today_dt, is_holiday, fuel_price, state) -> float:
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


def predict_market_price_horizon(
    train_df: pd.DataFrame,
    item_id: int,
    market_id: int,
    start_date: str,
    horizon_days: int,
    fuel_price: float,
    is_holiday_map: dict,
):
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")
    if horizon_days > 30:
        raise ValueError("❌ horizon_days cannot exceed 30")

    start_dt = pd.to_datetime(start_date, format="%Y-%m-%d", errors="coerce")
    if pd.isna(start_dt):
        raise ValueError("Invalid start_date (expected YYYY-MM-DD)")

    hist = _load_history(train_df, item_id, market_id)

    # stale CSV warning
    warnings = []
    last_date_in_csv = hist["DATE"].max()
    gap_days = (start_dt - last_date_in_csv).days
    if gap_days > 0:
        warnings.append(f"CSV is {gap_days} day(s) behind for {item_id}/{market_id} (latest: {last_date_in_csv.date()}). Using latest available data.")

    # buffer for chain lags
    buffer_prices = list(hist["TODAYS_PRICE"].tail(7).astype(float).values)
    if len(buffer_prices) < 7:
        pad = float(np.mean(buffer_prices)) if buffer_prices else 0.0
        buffer_prices = ([pad] * (7 - len(buffer_prices))) + buffer_prices
        warnings.append("Not enough last-7 prices in CSV; padded rolling window.")

    # state start: use latest available (not inventing missing days)
    current_date = start_dt
    current_price = float(hist.iloc[-1]["TODAYS_PRICE"])
    prev_price = float(hist.iloc[-2]["TODAYS_PRICE"]) if len(hist) >= 2 else current_price

    pred_price = None
    for step in range(1, horizon_days + 1):
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

        # holiday for the CURRENT day used as feature
        is_holiday = is_holiday_map.get(current_date.strftime("%Y-%m-%d"), 0)

        pred_price = _predict_one_step(item_id, market_id, current_date, is_holiday, fuel_price, chain_state)

        # roll forward
        next_date = current_date + timedelta(days=1)
        prev_price = current_price
        current_price = pred_price
        current_date = next_date

        buffer_prices.pop(0)
        buffer_prices.append(float(pred_price))

    target_date = (start_dt + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    return {
        "market_id": market_id,
        "target_date": target_date,
        "predicted_price": float(pred_price),
        "warnings": warnings,
    }


def recommend_best_market(payload: dict):
    """
    Expected payload keys:
      crop (str), quantity_kg (float), horizon_days (int),
      latitude (float), longitude (float)
    """
    # ---- validate inputs ----
    crop = str(payload["crop"]).strip().lower()
    if crop not in CROP_TO_ITEM_ID:
        raise ValueError(f"Unknown crop '{payload['crop']}'. Add it to CROP_TO_ITEM_ID mapping.")

    quantity_kg = float(payload["quantity_kg"])
    horizon_days = int(payload["horizon_days"])
    if horizon_days < 1 or horizon_days > 30:
        raise ValueError("horizon_days must be between 1 and 30")

    user_lat = float(payload["latitude"])
    user_lon = float(payload["longitude"])
    
    # Use current date as start_date
    start_dt = pd.Timestamp.now().normalize()
    start_date = start_dt.strftime("%Y-%m-%d")

    item_id = CROP_TO_ITEM_ID[crop]

    # ---- load train + get fuel ----
    train_df = load_train_csv()
    fuel_price = get_latest_fuel_price(train_df)

    # ---- build holiday flags for every date used in chain (start..start+horizon-1) ----

    chain_dates = [start_dt + timedelta(days=d) for d in range(0, horizon_days)]
    is_holiday_map, holiday_warnings = build_is_holiday_for_dates(chain_dates)

    # ---- transport settings ----
    # You said: "For now transport is 10rs per km"
    # I implement as 10 Rs per km per kg (scales with quantity).
    TRANSPORT_RS_PER_KM_PER_KG = 2.0

    results = []
    for market_id, info in MARKETS.items():
        dist_km = haversine_km(user_lat, user_lon, info["lat"], info["lon"])

        pred = predict_market_price_horizon(
            train_df=train_df,
            item_id=item_id,
            market_id=market_id,
            start_date=start_date,
            horizon_days=horizon_days,
            fuel_price=fuel_price,
            is_holiday_map=is_holiday_map,
        )

        predicted_price_per_kg = pred["predicted_price"]

        # Profit math:
        revenue = predicted_price_per_kg * quantity_kg

        # If you meant 10 Rs per km TOTAL TRIP (not per kg),
        # replace the next line with: transport_cost = dist_km * 10.0
        transport_cost = dist_km * TRANSPORT_RS_PER_KM_PER_KG * quantity_kg

        net_profit_est = revenue - transport_cost

        results.append({
            "market_id": market_id,
            "market_name": info["name"],
            "distance_km": round(dist_km, 1),
            "predicted_price_per_kg": f"LKR {predicted_price_per_kg:,.2f}",
            "estimated_revenue": f"LKR {revenue:,.2f}",
            "estimated_transport_cost": f"LKR {transport_cost:,.2f}",
            "estimated_net_profit": f"LKR {net_profit_est:,.2f}",
            "target_date": pred["target_date"],
            "warnings": pred["warnings"],
        })

    best = max(results, key=lambda x: x["estimated_net_profit"])

    # ---- build dynamic short message ----
    pettah = next(r for r in results if r['market_id']==0)
    dambulla = next(r for r in results if r['market_id']==1)
    
    msg_lines = [
        f"Forecast for {best['target_date']} (crop: {payload['crop']}, qty: {quantity_kg:.0f} kg, horizon: {horizon_days} day(s))",
        f"- Pettah:   {pettah['predicted_price_per_kg']}/kg | {pettah['distance_km']} km",
        f"- Dambulla: {dambulla['predicted_price_per_kg']}/kg | {dambulla['distance_km']} km",
        "",
        f"✅ Best market to sell: {best['market_name']} (highest estimated net profit after transport)."
    ]

    # ---- collect warnings ----
    warnings = []
    warnings.extend(holiday_warnings)
    for r in results:
        for w in r["warnings"]:
            warnings.append(f"{r['market_name']}: {w}")
    # unique
    warnings = list(dict.fromkeys([w for w in warnings if w]))

    # Determine if prediction is ready (successful)
    prediction_ready = True  # True if prediction completed successfully

    return {
        "crop": payload["crop"],
        "item_id": item_id,
        "quantity_kg": quantity_kg,
        "start_date": start_date,
        "target_date": best["target_date"],
        "horizon_days": horizon_days,
        "fuel_price_used": fuel_price,
        "transport_assumption": "10 Rs per km per kg (change if you mean per trip)",
        "prediction_ready": prediction_ready,
        "markets": results,
        "best_market": {
            "market_id": best["market_id"],
            "market_name": best["market_name"],
            "estimated_net_profit": best["estimated_net_profit"],
        },
        "message": "\n".join(msg_lines),
        "warnings": warnings
    }


# ---------------- Example local run ----------------
if __name__ == "__main__":
    api_payload = {
        "crop": "beans",
        "quantity_kg": 500,
        "horizon_days": 1,
        "latitude": 6.9271,
        "longitude": 79.8612
    }

    resp = recommend_best_market(api_payload)
    print(resp["message"])
    if resp["warnings"]:
        print("\nWarnings:")
        for w in resp["warnings"]:
            print("-", w)
    print("\nMarkets JSON:")
    print(resp["markets"])
