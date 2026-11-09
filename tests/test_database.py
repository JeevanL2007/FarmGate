"""
Unit Tests for Database Module
"""

import os
import pytest
import pandas as pd
from src.database import FarmGateDB


@pytest.fixture
def temp_db(tmp_path):
    test_db_path = str(tmp_path / "test_farmgate.db")
    db = FarmGateDB(db_path=test_db_path)
    return db


def test_db_initialization(temp_db):
    """Test that all tables are created on initialization."""
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        assert "markets" in tables
        assert "commodities" in tables
        assert "daily_prices" in tables
        assert "farmer_origins" in tables
        assert "farmer_decisions" in tables


def test_insert_and_get_markets(temp_db):
    """Test inserting and retrieving markets."""
    with temp_db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO markets (name, district, state, latitude, longitude, cess_rate, handling_charge_per_qtl)
            VALUES ('Test APMC', 'Mysuru', 'Karnataka', 12.30, 76.65, 1.5, 15.0)
        """)
        conn.commit()

    markets = temp_db.get_markets()
    assert len(markets) == 1
    assert markets.iloc[0]["name"] == "Test APMC"
    assert markets.iloc[0]["district"] == "Mysuru"


def test_log_farmer_decision(temp_db):
    """Test decision logging for analytics."""
    temp_db.log_farmer_decision(
        origin_name="Malur",
        commodity_name="Tomato",
        quantity_qtl=25.0,
        vehicle_type="Bolero Maxi Truck / Pickup",
        recommended_market="Yeshwanthpur APMC",
        recommended_net_profit=54000.0,
        alt_market="Kolar APMC",
        net_profit_difference=8500.0
    )

    history = temp_db.get_recent_decisions(limit=5)
    assert len(history) == 1
    assert history.iloc[0]["origin_name"] == "Malur"
    assert history.iloc[0]["recommended_net_profit"] == 54000.0
