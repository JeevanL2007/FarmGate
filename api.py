"""
FarmGate REST API Backend Service
Built with FastAPI. Exposes REST endpoints for:
- Markets list & details
- Commodities
- Cross-market latest prices
- Smart selling recommendation calculations
- ML price forecast trajectories
Interactive Swagger Documentation available at: http://localhost:8000/docs
"""

import os
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure project root is in python path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import FarmGateDB
from src.decision_engine import SmartSellingEngine
from src.ml_model import PriceForecaster
from src.utils import KARNATAKA_FARMER_ORIGINS, VEHICLE_PROFILES

app = FastAPI(
    title="FarmGate Agricultural Decision Support API",
    description="REST API for agricultural market price comparison, freight-adjusted net profit optimization, and ML forecasting.",
    version="1.0.0"
)

db = FarmGateDB()
engine = SmartSellingEngine()


# --- Pydantic Request/Response Models ---
class MarketItem(BaseModel):
    id: int
    name: str
    district: str
    state: str
    latitude: float
    longitude: float
    cess_rate: float
    handling_charge_per_qtl: float


class CommodityItem(BaseModel):
    id: int
    name: str
    kannada_name: str
    category: str
    perishable_level: str
    standard_bag_weight_kg: float


class RecommendationRequest(BaseModel):
    commodity_id: int = Field(..., example=1, description="Commodity ID (e.g. 1 for Tomato)")
    quantity_qtl: float = Field(..., gt=0, example=25.0, description="Quantity in Quintals")
    origin_name: Optional[str] = Field("Malur (Kolar District)", description="Pre-configured taluk/origin")
    custom_lat: Optional[float] = Field(None, description="Optional custom latitude")
    custom_lon: Optional[float] = Field(None, description="Optional custom longitude")
    vehicle_type: Optional[str] = Field(None, description="Vehicle model name")
    custom_rate_per_km: Optional[float] = Field(None, description="Custom freight rate ₹/km")
    custom_loading_per_qtl: Optional[float] = Field(None, description="Custom labor loading ₹/qtl")
    price_basis: str = Field("modal_price", description="modal_price, min_price, or max_price")


# --- API Endpoints ---

@app.get("/")
def root():
    return {
        "service": "FarmGate Agricultural Decision Support API",
        "status": "online",
        "documentation": "/docs",
        "endpoints": [
            "/api/markets",
            "/api/commodities",
            "/api/prices/latest",
            "/api/recommend",
            "/api/forecast"
        ]
    }


@app.get("/api/markets", response_model=List[MarketItem])
def get_all_markets():
    """Retrieve all monitored APMC agricultural markets."""
    df = db.get_markets()
    return df.to_dict(orient="records")


@app.get("/api/commodities", response_model=List[CommodityItem])
def get_all_commodities():
    """Retrieve all supported agricultural commodities."""
    df = db.get_commodities()
    return df.to_dict(orient="records")


@app.get("/api/prices/latest")
def get_latest_market_prices(
    commodity_id: Optional[int] = Query(None, description="Filter by Commodity ID"),
    date: Optional[str] = Query(None, description="Specific trade date (YYYY-MM-DD)")
):
    """Fetch latest APMC market prices across Karnataka."""
    df = db.get_latest_prices(commodity_id=commodity_id, target_date=date)
    return df.to_dict(orient="records")


@app.post("/api/recommend")
def recommend_optimal_market(req: RecommendationRequest):
    """
    Core Smart Selling Algorithm:
    Computes freight, labor, and cess deductions to rank APMC markets by net take-home earnings.
    """
    # Determine coordinates
    if req.custom_lat is not None and req.custom_lon is not None:
        lat, lon = req.custom_lat, req.custom_lon
        origin_label = "Custom Location"
    elif req.origin_name in KARNATAKA_FARMER_ORIGINS:
        info = KARNATAKA_FARMER_ORIGINS[req.origin_name]
        lat, lon = info["lat"], info["lon"]
        origin_label = req.origin_name
    else:
        raise HTTPException(status_code=400, detail="Invalid origin or coordinates specified.")

    prices_df = db.get_latest_prices(commodity_id=req.commodity_id)
    if prices_df.empty:
        raise HTTPException(status_code=404, detail="No price records found for this commodity.")

    results = engine.evaluate_markets(
        candidate_markets_df=prices_df,
        farmer_lat=lat,
        farmer_lon=lon,
        quantity_qtl=req.quantity_qtl,
        vehicle_type=req.vehicle_type,
        custom_rate_per_km=req.custom_rate_per_km,
        custom_loading_per_qtl=req.custom_loading_per_qtl,
        price_basis=req.price_basis
    )

    comm_name = prices_df["commodity_name"].iloc[0]
    summary = engine.generate_recommendation_summary(
        results=results,
        commodity_name=comm_name,
        origin_name=origin_label
    )

    return {
        "status": "success",
        "best_market": summary["best_market"].to_dict(),
        "nearest_market": summary["nearest_market"].to_dict(),
        "profit_gain_vs_nearest": round(summary["profit_gain_vs_nearest"], 2),
        "is_best_the_nearest": summary["is_best_the_nearest"],
        "break_even_nearest_price": summary["break_even_nearest_price"],
        "strategy_text": summary["strategy_text"],
        "all_ranked_markets": [r.to_dict() for r in results]
    }


@app.get("/api/forecast")
def get_ml_price_forecast(
    commodity_id: int = Query(..., description="Commodity ID"),
    market_id: int = Query(..., description="APMC Market ID"),
    horizon_days: int = Query(7, ge=3, le=14, description="Forecast horizon in days"),
    model_type: str = Query("RandomForest", description="RandomForest, GradientBoosting, or Ridge")
):
    """
    Train regression model on historical time series and project 7-14 day price path.
    """
    history_df = db.get_historical_prices(commodity_id=commodity_id, market_ids=[market_id])
    if len(history_df) < 40:
        raise HTTPException(status_code=400, detail="Insufficient historical records to train ML model.")

    forecaster = PriceForecaster(model_type=model_type)
    train_res = forecaster.train(history_df, test_size_days=30)
    forecast_df = forecaster.forecast_next_days(history_df, horizon_days=horizon_days)

    latest_modal = float(history_df["modal_price"].iloc[-1])
    timing_signal = forecaster.generate_selling_timing_signal(
        current_price=latest_modal,
        forecast_df=forecast_df
    )

    return {
        "commodity_id": commodity_id,
        "market_id": market_id,
        "metrics": train_res["metrics"],
        "signal": timing_signal,
        "forecast": forecast_df.to_dict(orient="records")
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
