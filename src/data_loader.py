"""
FarmGate Data Loader & Database Seeder
Seeds APMC markets, commodities, farmer origins, and generates realistic
Agmarknet-compliant historical price records for Karnataka.
"""

import os
import random
from datetime import datetime, timedelta
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

# Market price multiplier profile (e.g. Yeshwanthpur/Bengaluru commands higher retail/wholesale demand premium,
# Kolar has high volume production of tomato so lower producer price, creating transport arbitrage)
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
        # End date: Current or recent anchor date
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

                    # Weekly cycle: Sunday lower volume, Monday peak arrivals
                    weekend_volume_mod = 0.6 if day_of_week == 6 else (1.3 if day_of_week == 0 else 1.0)

                    # Macro price for this day
                    day_macro_price = base_p * macro_trend[day_idx] * m_factor

                    # Daily random noise
                    noise_pct = np.random.normal(0, volatility * 0.3)
                    modal_price = round(day_macro_price * (1.0 + noise_pct), -1)  # round to nearest 10
                    modal_price = max(250.0, modal_price)

                    # Price spread: Min price typically 10-20% lower, Max price 10-25% higher
                    min_price = round(modal_price * np.random.uniform(0.82, 0.93), -1)
                    max_price = round(modal_price * np.random.uniform(1.07, 1.22), -1)

                    # Arrival volume inversely related to price surges
                    arrival_factor = (base_p / modal_price) ** 0.8
                    arrivals = round(base_arr * arrival_factor * weekend_volume_mod * np.random.uniform(0.8, 1.25), 1)

                    records.append((m_id, c_id, date_str, arrivals, min_price, max_price, modal_price))

        # Batch insert daily prices
        cursor.executemany("""
            INSERT OR REPLACE INTO daily_prices (
                market_id, commodity_id, date, arrivals_in_qtl, min_price, max_price, modal_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()

        # Also write a CSV copy to data/agmarknet_historical_data.csv for standard records
        csv_path = os.path.join(os.path.dirname(DEFAULT_DB_PATH), "agmarknet_historical_data.csv")
        export_df = pd.read_sql_query("""
            SELECT 
                dp.date,
                m.name as market_name,
                m.district,
                c.name as commodity_name,
                c.kannada_name,
                dp.arrivals_in_qtl,
                dp.min_price,
                dp.max_price,
                dp.modal_price
            FROM daily_prices dp
            JOIN markets m ON dp.market_id = m.id
            JOIN commodities c ON dp.commodity_id = c.id
            ORDER BY dp.date ASC, m.name ASC
        """, conn)
        export_df.to_csv(csv_path, index=False)
        print(f"Database seeded with {len(records)} daily records. Exported to {csv_path}")


if __name__ == "__main__":
    db = FarmGateDB()
    seed_database_and_generate_data(db, num_days=180)
