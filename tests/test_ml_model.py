"""
Unit Tests for Machine Learning Price Forecaster
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.ml_model import PriceForecaster


@pytest.fixture
def synthetic_history():
    np.random.seed(42)
    dates = [datetime(2026, 1, 1) + timedelta(days=i) for i in range(120)]
    prices = 2500 + 300 * np.sin(np.linspace(0, 4 * np.pi, 120)) + np.random.normal(0, 50, 120)
    arrivals = 1000 + np.random.normal(0, 100, 120)

    df = pd.DataFrame({
        "date": [d.strftime("%Y-%m-%d") for d in dates],
        "modal_price": prices,
        "arrivals_in_qtl": arrivals
    })
    return df


def test_price_forecaster_training(synthetic_history):
    forecaster = PriceForecaster(model_type="RandomForest")
    train_res = forecaster.train(synthetic_history, test_size_days=20)

    assert "metrics" in train_res
    metrics = train_res["metrics"]
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert metrics["MAE"] >= 0
    assert metrics["R2"] > -1.0  # Reasonable fit on smooth synthetic curve


def test_multi_day_forecasting(synthetic_history):
    forecaster = PriceForecaster(model_type="Ridge")
    forecaster.train(synthetic_history, test_size_days=20)

    forecast_df = forecaster.forecast_next_days(synthetic_history, horizon_days=7)
    assert len(forecast_df) == 7
    assert "Forecasted Price (₹/qtl)" in forecast_df.columns
    assert "Lower CI (₹/qtl)" in forecast_df.columns
    assert "Upper CI (₹/qtl)" in forecast_df.columns

    # Check bounds
    for _, row in forecast_df.iterrows():
        assert row["Lower CI (₹/qtl)"] <= row["Forecasted Price (₹/qtl)"] <= row["Upper CI (₹/qtl)"]


def test_selling_signal(synthetic_history):
    forecaster = PriceForecaster(model_type="Ridge")
    forecaster.train(synthetic_history, test_size_days=20)
    forecast_df = forecaster.forecast_next_days(synthetic_history, horizon_days=7)

    current_price = synthetic_history["modal_price"].iloc[-1]
    signal = forecaster.generate_selling_timing_signal(
        current_price=current_price,
        forecast_df=forecast_df,
        perishable_level="Medium"
    )

    assert "signal" in signal
    assert "rationale" in signal
    assert signal["signal"] in [
        "SELL IMMEDIATELY",
        "HOLD FOR HIGHER GAINS",
        "HOLD WITH CAUTION (1-2 DAYS)",
        "STABLE / SELL AT WILL"
    ]
