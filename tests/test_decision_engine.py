"""
Unit Tests for Smart Selling Decision Engine
"""

import pytest
import pandas as pd
from src.decision_engine import SmartSellingEngine, DecisionResult
from src.utils import haversine_distance_km, VEHICLE_PROFILES


@pytest.fixture
def sample_market_df():
    data = [
        {
            "market_id": 1,
            "market_name": "Local APMC (Near)",
            "district": "Kolar",
            "latitude": 13.13,
            "longitude": 78.12,
            "cess_rate": 1.5,
            "handling_charge_per_qtl": 14.0,
            "min_price": 2000.0,
            "max_price": 2400.0,
            "modal_price": 2200.0
        },
        {
            "market_id": 2,
            "market_name": "City APMC (Far & High Price)",
            "district": "Bengaluru",
            "latitude": 13.02,
            "longitude": 77.55,
            "cess_rate": 1.5,
            "handling_charge_per_qtl": 18.0,
            "min_price": 2800.0,
            "max_price": 3200.0,
            "modal_price": 3000.0
        }
    ]
    return pd.DataFrame(data)


def test_haversine_distance():
    # Distance between Bengaluru (13.02, 77.55) and Kolar (13.13, 78.12) is ~65-75 km with winding
    d = haversine_distance_km(13.02, 77.55, 13.13, 78.12)
    assert 60.0 < d < 100.0


def test_smart_decision_ranking(sample_market_df):
    engine = SmartSellingEngine()
    # Farmer located at Kolar (13.13, 78.12)
    results = engine.evaluate_markets(
        candidate_markets_df=sample_market_df,
        farmer_lat=13.13,
        farmer_lon=78.12,
        quantity_qtl=30.0,  # 30 quintals = 3 Tons
        vehicle_type="Bolero Maxi Truck / Pickup"
    )

    assert len(results) == 2
    # Verify ranking assigns 1 and 2
    assert results[0].rank == 1
    assert results[1].rank == 2

    # Verify gross revenue calculation for top result
    top = results[0]
    assert top.gross_revenue == top.quantity_qtl * top.modal_price
    assert top.net_profit == top.gross_revenue - top.total_deductions
    assert top.net_price_per_qtl == top.net_profit / top.quantity_qtl


def test_decision_summary_generation(sample_market_df):
    engine = SmartSellingEngine()
    results = engine.evaluate_markets(
        candidate_markets_df=sample_market_df,
        farmer_lat=13.13,
        farmer_lon=78.12,
        quantity_qtl=30.0
    )

    summary = engine.generate_recommendation_summary(
        results=results,
        commodity_name="Tomato",
        origin_name="Kolar Cluster"
    )

    assert summary["status"] == "success"
    assert "best_market" in summary
    assert "nearest_market" in summary
    assert "profit_gain_vs_nearest" in summary
    assert "strategy_text" in summary
    assert not summary["ranking_table"].empty
