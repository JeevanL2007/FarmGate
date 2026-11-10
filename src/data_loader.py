"""
FarmGate Data Loader & Database Seeder
Seeds APMC markets, commodities, farmer origins, and generates realistic
Agmarknet-compliant historical price records for Karnataka.
Includes auto-updater to keep dates continuously synchronized with the current calendar date.
"""

import os
import random
from datetime import datetime, timedelta, date
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from .database import FarmGateDB, DEFAULT_DB_PATH
from .utils import KARNATAKA_FARMER_ORIGINS


# Karnataka APMC Markets
KARNATAKA_MARKETS = [
    {
        "name": "Kolar APMC",
        "district": "Kolar",
        "state": "Karnataka",
        "latitude": 13.1378,
        "longitude": 78.1291,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 14.0
    },
    {
        "name": "Yeshwanthpur APMC (Bengaluru)",
        "district": "Bengaluru Urban",
        "state": "Karnataka",
        "latitude": 13.0238,
        "longitude": 77.5529,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 18.0
    },
    {
        "name": "Kalasipalya APMC (Bengaluru)",
        "district": "Bengaluru Urban",
        "state": "Karnataka",
        "latitude": 12.9616,
        "longitude": 77.5772,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 18.0
    },
    {
        "name": "Mysuru APMC (Bandipalya)",
        "district": "Mysuru",
        "state": "Karnataka",
        "latitude": 12.2743,
        "longitude": 76.6749,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 15.0
    },
    {
        "name": "Hubballi APMC (Amaragol)",
        "district": "Dharwad",
        "state": "Karnataka",
        "latitude": 15.3907,
        "longitude": 75.1018,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 16.0
    },
    {
        "name": "Belagavi APMC",
        "district": "Belagavi",
        "state": "Karnataka",
        "latitude": 15.8647,
        "longitude": 74.5098,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 15.0
    },
    {
        "name": "Chikkaballapur APMC",
        "district": "Chikkaballapur",
        "state": "Karnataka",
        "latitude": 13.4355,
        "longitude": 77.7275,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 14.0
    },
    {
        "name": "Hassan APMC",
        "district": "Hassan",
        "state": "Karnataka",
        "latitude": 13.0033,
        "longitude": 76.1004,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 14.0
    },
    {
        "name": "Shivamogga APMC",
        "district": "Shivamogga",
        "state": "Karnataka",
        "latitude": 13.9299,
        "longitude": 75.5681,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 14.0
    },
    {
        "name": "Davanagere APMC",
        "district": "Davanagere",
        "state": "Karnataka",
        "latitude": 14.4644,
        "longitude": 75.9218,
        "cess_rate": 1.5,
        "handling_charge_per_qtl": 14.0
    }
]

# Primary agricultural commodities specified in synopsis & syllabus
COMMODITIES = [
    {
        "name": "Tomato",
        "kannada_name": "ಟೊಮೆಟೊ",
        "category": "Vegetable",
        "perishable_level": "High",
        "standard_bag_weight_kg": 25.0,
        "base_price": 2400.0,
        "volatility": 0.28,
        "base_arrivals": 1200.0
    },
    {
        "name": "Onion",
        "kannada_name": "ಈರುಳ್ಳಿ",
        "category": "Vegetable",
        "perishable_level": "Medium",
        "standard_bag_weight_kg": 50.0,
        "base_price": 2800.0,
        "volatility": 0.22,
        "base_arrivals": 2200.0
    },
    {
        "name": "Potato",
        "kannada_name": "ಆಲೂಗಡ್ಡೆ",
        "category": "Vegetable",
        "perishable_level": "Low",
        "standard_bag_weight_kg": 50.0,
        "base_price": 2100.0,
        "volatility": 0.16,
        "base_arrivals": 1800.0
    },
    {
        "name": "Green Beans",
        "kannada_name": "ಹುರುಳಿಕಾಯಿ (ಬೀನ್ಸ್)",
        "category": "Vegetable",
        "perishable_level": "High",
        "standard_bag_weight_kg": 50.0,
        "base_price": 4200.0,
        "volatility": 0.30,
        "base_arrivals": 600.0
    },
    {
        "name": "Green Chilli",
        "kannada_name": "ಹಸಿರು ಮೆಣಸಿನಕಾಯಿ",
        "category": "Spices / Vegetable",
        "perishable_level": "Medium",
        "standard_bag_weight_kg": 40.0,
        "base_price": 4600.0,
        "volatility": 0.25,
        "base_arrivals": 750.0
    },
    {
        "name": "Carrot",
        "kannada_name": "ಕ್ಯಾರೆಟ್",
        "category": "Vegetable",
        "perishable_level": "Medium",
        "standard_bag_weight_kg": 50.0,
        "base_price": 2700.0,
        "volatility": 0.18,
        "base_arrivals": 900.0
    },
    {
        "name": "Capsicum",
        "kannada_name": "ದಪ್ಪ ಮೆಣಸಿನಕಾಯಿ (ಕ್ಯಾಪ್ಸಿಕಂ)",
        "category": "Vegetable",
        "perishable_level": "High",
        "standard_bag_weight_kg": 30.0,
        "base_price": 3800.0,
        "volatility": 0.26,
        "base_arrivals": 500.0
    }
]

# Market price multiplier profile
MARKET_PRICE_FACTORS = {
    "Kolar APMC": {"Tomato": 0.88, "Onion": 0.98, "Potato": 0.96, "Green Beans": 0.92, "Green Chilli": 0.94, "Carrot": 0.94, "Capsicum": 0.90},
    "Yeshwanthpur APMC (Bengaluru)": {"Tomato": 1.15, "Onion": 1.12, "Potato": 1.10, "Green Beans": 1.16, "Green Chilli": 1.14, "Carrot": 1.12, "Capsicum": 1.15},
    "Kalasipalya APMC (Bengaluru)": {"Tomato": 1.12, "Onion": 1.10, "Potato": 1.08, "Green Beans": 1.18, "Green Chilli": 1.16, "Carrot": 1.10, "Capsicum": 1.14},
    "Mysuru APMC (Bandipalya)": {"Tomato": 1.04, "Onion": 1.02, "Potato": 1.03, "Green Beans": 1.05, "Green Chilli": 1.03, "Carrot": 1.04, "Capsicum": 1.06},
    "Hubballi APMC (Amaragol)": {"Tomato": 0.95, "Onion": 0.89, "Potato": 0.98, "Green Beans": 0.96, "Green Chilli": 0.92, "Carrot": 0.97, "Capsicum": 0.95},
    "Belagavi APMC": {"Tomato": 0.96, "Onion": 0.91, "Potato": 0.92, "Green Beans": 0.98, "Green Chilli": 0.93, "Carrot": 0.95, "Capsicum": 0.94},
    "Chikkaballapur APMC": {"Tomato": 0.90, "Onion": 0.96, "Potato": 0.95, "Green Beans": 0.93, "Green Chilli": 0.95, "Carrot": 0.92, "Capsicum": 0.92},
    "Hassan APMC": {"Tomato": 1.00, "Onion": 0.99, "Potato": 0.87, "Green Beans": 1.02, "Green Chilli": 0.97, "Carrot": 0.98, "Capsicum": 0.99},
    "Shivamogga APMC": {"Tomato": 1.03, "Onion": 1.02, "Potato": 1.01, "Green Beans": 1.04, "Green Chilli": 1.02, "Carrot": 1.01, "Capsicum": 1.03},
    "Davanagere APMC": {"Tomato": 0.98, "Onion": 0.96, "Potato": 0.97, "Green Beans": 0.99, "Green Chilli": 0.96, "Carrot": 0.97, "Capsicum": 0.98}
}


def seed_database_and_generate_data(db: FarmGateDB, num_days: int = 180):
    """
    Seeds initial reference tables and populates daily price history for 180+ days.
    """
    np.random.seed(42)
    random.seed(42)

    with db.get_connection() as conn:
        cursor = conn.cursor()

        # 1. Insert Markets
        for m in KARNATAKA_MARKETS:
            cursor.execute("""
                INSERT OR IGNORE INTO markets (name, district, state, latitude, longitude, cess_rate, handling_charge_per_qtl)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (m["name"], m["district"], m["state"], m["latitude"], m["longitude"], m["cess_rate"], m["handling_charge_per_qtl"]))

        # 2. Insert Commodities
        for c in COMMODITIES:
            cursor.execute("""
                INSERT OR IGNORE INTO commodities (name, kannada_name, category, perishable_level, standard_bag_weight_kg)
                VALUES (?, ?, ?, ?, ?)
            """, (c["name"], c["kannada_name"], c["category"], c["perishable_level"], c["standard_bag_weight_kg"]))

        # 3. Insert Farmer Origins
        for name, data in KARNATAKA_FARMER_ORIGINS.items():
            cursor.execute("""
                INSERT OR IGNORE INTO farmer_origins (name, district, latitude, longitude)
                VALUES (?, ?, ?, ?)
            """, (name, data["district"], data["lat"], data["lon"]))

        conn.commit()

        # Retrieve IDs
        cursor.execute("SELECT id, name FROM markets")
        market_map = {row[1]: row[0] for row in cursor.fetchall()}

        cursor.execute("SELECT id, name FROM commodities")
        commodity_map = {row[1]: row[0] for row in cursor.fetchall()}

        # 4. Generate Time Series Data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=num_days)

        records = []
        date_list = [start_date + timedelta(days=d) for d in range(num_days + 1)]

        for comm in COMMODITIES:
            c_name = comm["name"]
            c_id = commodity_map[c_name]
            base_p = comm["base_price"]
            volatility = comm["volatility"]
            base_arr = comm["base_arrivals"]

            # Generate a seasonal trend wave over the timeline
            t = np.linspace(0, 2 * np.pi * (num_days / 180), len(date_list))
            macro_trend = 1.0 + 0.18 * np.sin(t) + 0.08 * np.cos(2 * t)

            for m_name, m_id in market_map.items():
                m_factor = MARKET_PRICE_FACTORS.get(m_name, {}).get(c_name, 1.0)

                for day_idx, d in enumerate(date_list):
                    date_str = d.strftime("%Y-%m-%d")
                    day_of_week = d.weekday()  # 0=Monday, 6=Sunday

                    weekend_volume_mod = 0.6 if day_of_week == 6 else (1.3 if day_of_week == 0 else 1.0)
                    day_macro_price = base_p * macro_trend[day_idx] * m_factor

                    noise_pct = np.random.normal(0, volatility * 0.3)
                    modal_price = round(day_macro_price * (1.0 + noise_pct), -1)
                    modal_price = max(250.0, modal_price)

                    min_price = round(modal_price * np.random.uniform(0.82, 0.93), -1)
                    max_price = round(modal_price * np.random.uniform(1.07, 1.22), -1)

                    arrival_factor = (base_p / modal_price) ** 0.8
                    arrivals = round(base_arr * arrival_factor * weekend_volume_mod * np.random.uniform(0.8, 1.25), 1)

                    records.append((m_id, c_id, date_str, arrivals, min_price, max_price, modal_price))

        cursor.executemany("""
            INSERT OR REPLACE INTO daily_prices (
                market_id, commodity_id, date, arrivals_in_qtl, min_price, max_price, modal_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()


def auto_update_daily_prices(db: FarmGateDB) -> int:
    """
    Checks the latest date in daily_prices against the current calendar date (today).
    If missing dates exist between the last recorded date and today, it seamlessly generates
    continuous and realistic market auction records up to today.
    Returns the count of newly appended records.
    """
    today = datetime.now().date()

    with db.get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT MAX(date) FROM daily_prices")
        row = cursor.fetchone()
        max_date_str = row[0] if row and row[0] else None

        if not max_date_str:
            seed_database_and_generate_data(db, num_days=180)
            return 12670

        max_date = datetime.strptime(max_date_str, "%Y-%m-%d").date()

        # If already up to date, nothing to do
        if max_date >= today:
            return 0

        # Retrieve market and commodity lookups
        cursor.execute("SELECT id, name FROM markets")
        markets = cursor.fetchall()

        cursor.execute("SELECT id, name FROM commodities")
        commodities = cursor.fetchall()
        comm_meta = {c["name"]: c for c in COMMODITIES}

        # Calculate days gap
        days_gap = (today - max_date).days
        new_records = []

        # Iterate forward day-by-day
        for day_offset in range(1, days_gap + 1):
            target_date = max_date + timedelta(days=day_offset)
            target_date_str = target_date.strftime("%Y-%m-%d")
            day_of_week = target_date.weekday()
            weekend_volume_mod = 0.6 if day_of_week == 6 else (1.3 if day_of_week == 0 else 1.0)

            for m_id, m_name in markets:
                for c_id, c_name in commodities:
                    c_info = comm_meta.get(c_name, {})
                    base_p = c_info.get("base_price", 2500.0)
                    volatility = c_info.get("volatility", 0.20)
                    base_arr = c_info.get("base_arrivals", 1000.0)
                    m_factor = MARKET_PRICE_FACTORS.get(m_name, {}).get(c_name, 1.0)

                    # Fetch the latest known modal price for continuity
                    cursor.execute("""
                        SELECT modal_price FROM daily_prices 
                        WHERE market_id = ? AND commodity_id = ? 
                        ORDER BY date DESC LIMIT 1
                    """, (m_id, c_id))
                    last_row = cursor.fetchone()
                    prev_modal = float(last_row[0]) if last_row else (base_p * m_factor)

                    # Continuous random walk with mean reversion to base_price * factor
                    mean_revert_pull = 0.05 * ((base_p * m_factor) - prev_modal)
                    daily_noise = np.random.normal(0, volatility * 0.12)
                    new_modal = round(prev_modal + mean_revert_pull + (prev_modal * daily_noise), -1)
                    new_modal = max(250.0, new_modal)

                    new_min = round(new_modal * np.random.uniform(0.84, 0.93), -1)
                    new_max = round(new_modal * np.random.uniform(1.07, 1.20), -1)

                    arrival_factor = (base_p / new_modal) ** 0.8
                    arrivals = round(base_arr * arrival_factor * weekend_volume_mod * np.random.uniform(0.85, 1.2), 1)

                    new_records.append((m_id, c_id, target_date_str, arrivals, new_min, new_max, new_modal))

        if new_records:
            cursor.executemany("""
                INSERT OR REPLACE INTO daily_prices (
                    market_id, commodity_id, date, arrivals_in_qtl, min_price, max_price, modal_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, new_records)
            conn.commit()

        return len(new_records)


if __name__ == "__main__":
    db = FarmGateDB()
    added = auto_update_daily_prices(db)
    print(f"Auto-update complete. {added} new daily records added up to today.")
