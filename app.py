from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from predict import recommend_best_market  # <-- your function that takes payload dict


app = FastAPI(title="Market Price Prediction API")


class PredictRequest(BaseModel):
    crop: str = Field(..., description="Crop name (e.g., tomato)")
    quantity_kg: float = Field(..., gt=0, description="Quantity in KG")
    horizon_days: int = Field(..., ge=1, le=30, description="Forecast horizon (1-30)")
    latitude: float = Field(..., description="User latitude")
    longitude: float = Field(..., description="User longitude")


class MarketResult(BaseModel):
    market_id: int
    market_name: str
    distance_km: float
    predicted_price_per_kg: str
    estimated_revenue: str
    estimated_transport_cost: str
    estimated_net_profit: str
    target_date: str
    warnings: List[str]


class PredictResponse(BaseModel):
    crop: str
    item_id: int
    quantity_kg: float
    start_date: str
    target_date: str
    horizon_days: int
    fuel_price_used: float
    transport_assumption: str
    prediction_ready: bool
    markets: List[MarketResult]
    best_market: Dict[str, Any]
    message: str
    warnings: List[str]


@app.get("/")
def read_root():
    return {"message": "Welcome to the Market Price Prediction API"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        payload = req.model_dump()

        # call your existing logic (it fetches fuel + holiday itself)
        result = recommend_best_market(payload)
        return result

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
