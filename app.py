from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from predict import recommend_best_market
from typing import Optional, List, Dict, Any

app = FastAPI(title="Market Price Prediction API")

class PredictionRequest(BaseModel):
    user_lat: float = Field(..., description="User's latitude")
    user_lon: float = Field(..., description="User's longitude")
    item_id: int = Field(..., description="Item ID to predict price for")
    start_date: str = Field(..., description="Start date in YYYY-MM-DD format")
    horizon_days: int = Field(..., description="Prediction horizon in days (1-30)")
    is_holiday: int = Field(..., description="1 if holiday, 0 otherwise")
    fuel_price: float = Field(..., description="Current fuel price")
    transport_cost_per_km: float = Field(default=0.0, description="Transport cost per km")

class MarketPrediction(BaseModel):
    market_id: int
    market_name: str
    distance_km: float
    predicted_price: float
    transport_cost_est: float
    net_est_price: float
    warnings: List[str]
    target_date: str

class PredictionResponse(BaseModel):
    target_date: str
    item_id: int
    horizon_days: int
    basis: str
    markets: List[MarketPrediction]
    best_market: MarketPrediction
    message: str
    warnings: List[str]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Market Price Prediction API"}

@app.post("/predict", response_model=PredictionResponse)
def predict_price(request: PredictionRequest):
    try:
        # Call the existing function from predict.py
        result = recommend_best_market(
            user_lat=request.user_lat,
            user_lon=request.user_lon,
            item_id=request.item_id,
            start_date=request.start_date,
            horizon_days=request.horizon_days,
            is_holiday=request.is_holiday,
            fuel_price=request.fuel_price,
            transport_cost_per_km=request.transport_cost_per_km
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

# To run the app:
# uvicorn app:app --reload
