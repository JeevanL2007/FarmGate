"""
FarmGate Database Module
Provides SQLite database connection, schema initialization, and query helpers.
"""

import os
import sqlite3
from typing import List, Dict, Any, Optional
import pandas as pd


DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "farmgate.db"
)


class FarmGateDB:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initializes database schema if tables do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # 1. Markets Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS markets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    district TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'Karnataka',
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    cess_rate REAL DEFAULT 1.5,
                    handling_charge_per_qtl REAL DEFAULT 15.0
                )
            """)

            # 2. Commodities Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS commodities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    kannada_name TEXT NOT NULL,
                    category TEXT NOT NULL,
                    perishable_level TEXT NOT NULL DEFAULT 'Medium',
                    standard_bag_weight_kg REAL DEFAULT 50.0
                )
            """)

            # 3. Daily APMC Prices Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id INTEGER NOT NULL,
                    commodity_id INTEGER NOT NULL,
                    date TEXT NOT NULL,
                    arrivals_in_qtl REAL NOT NULL,
                    min_price REAL NOT NULL,
                    max_price REAL NOT NULL,
                    modal_price REAL NOT NULL,
                    FOREIGN KEY (market_id) REFERENCES markets(id),
                    FOREIGN KEY (commodity_id) REFERENCES commodities(id),
                    UNIQUE(market_id, commodity_id, date)
                )
            """)

            # 4. Farmer Origins / Hubs Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS farmer_origins (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    district TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL
                )
            """)

            # 5. Farmer Decision Audit Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS farmer_decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    origin_name TEXT NOT NULL,
                    commodity_name TEXT NOT NULL,
                    quantity_qtl REAL NOT NULL,
                    vehicle_type TEXT NOT NULL,
                    recommended_market TEXT NOT NULL,
                    recommended_net_profit REAL NOT NULL,
                    alt_market TEXT,
                    net_profit_difference REAL
                )
            """)

            conn.commit()

    def get_markets(self) -> pd.DataFrame:
        """Returns all registered markets as a DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM markets ORDER BY name ASC", conn)

    def get_commodities(self) -> pd.DataFrame:
        """Returns all registered commodities as a DataFrame."""
        with self.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM commodities ORDER BY name ASC", conn)

    def get_farmer_origins(self) -> pd.DataFrame:
        """Returns known farmer origin hubs."""
        with self.get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM farmer_origins ORDER BY name ASC", conn)

    def get_latest_prices(self, commodity_id: Optional[int] = None, target_date: Optional[str] = None) -> pd.DataFrame:
        """
        Retrieves latest available market prices for a commodity.
        If target_date is not provided, defaults to the max available date for each market.
        """
        with self.get_connection() as conn:
            query = """
                SELECT 
                    dp.id,
                    dp.date,
                    m.id as market_id,
                    m.name as market_name,
                    m.district,
                    m.latitude,
                    m.longitude,
                    m.cess_rate,
                    m.handling_charge_per_qtl,
                    c.id as commodity_id,
                    c.name as commodity_name,
                    c.kannada_name,
                    dp.arrivals_in_qtl,
                    dp.min_price,
                    dp.max_price,
                    dp.modal_price
                FROM daily_prices dp
                JOIN markets m ON dp.market_id = m.id
                JOIN commodities c ON dp.commodity_id = c.id
            """
            conditions = []
            params = []

            if commodity_id is not None:
                conditions.append("dp.commodity_id = ?")
                params.append(commodity_id)

            if target_date is not None:
                conditions.append("dp.date = ?")
                params.append(target_date)
            else:
                # Latest date per market for the commodity
                conditions.append("""
                    dp.date = (
                        SELECT MAX(sub.date) 
                        FROM daily_prices sub 
                        WHERE sub.commodity_id = dp.commodity_id 
                        AND sub.market_id = dp.market_id
                    )
                """)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY dp.modal_price DESC"
            return pd.read_sql_query(query, conn, params=params)

    def get_historical_prices(
        self,
        commodity_id: int,
        market_ids: Optional[List[int]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieves historical price records for time-series analysis and ML modeling.
        """
        with self.get_connection() as conn:
            query = """
                SELECT 
                    dp.date,
                    m.id as market_id,
                    m.name as market_name,
                    m.district,
                    c.id as commodity_id,
                    c.name as commodity_name,
                    dp.arrivals_in_qtl,
                    dp.min_price,
                    dp.max_price,
                    dp.modal_price
                FROM daily_prices dp
                JOIN markets m ON dp.market_id = m.id
                JOIN commodities c ON dp.commodity_id = c.id
                WHERE dp.commodity_id = ?
            """
            params: List[Any] = [commodity_id]

            if market_ids:
                placeholders = ",".join("?" for _ in market_ids)
                query += f" AND dp.market_id IN ({placeholders})"
                params.extend(market_ids)

            if start_date:
                query += " AND dp.date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND dp.date <= ?"
                params.append(end_date)

            query += " ORDER BY dp.date ASC, m.name ASC"
            return pd.read_sql_query(query, conn, params=params)

    def get_available_dates(self, commodity_id: Optional[int] = None) -> List[str]:
        """Returns sorted distinct dates available in daily_prices."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if commodity_id:
                cursor.execute(
                    "SELECT DISTINCT date FROM daily_prices WHERE commodity_id = ? ORDER BY date DESC",
                    (commodity_id,)
                )
            else:
                cursor.execute("SELECT DISTINCT date FROM daily_prices ORDER BY date DESC")
            return [row[0] for row in cursor.fetchall()]

    def log_farmer_decision(
        self,
        origin_name: str,
        commodity_name: str,
        quantity_qtl: float,
        vehicle_type: str,
        recommended_market: str,
        recommended_net_profit: float,
        alt_market: Optional[str] = None,
        net_profit_difference: Optional[float] = None
    ):
        """Logs farmer queries for analytics and audit trails."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO farmer_decisions (
                    origin_name, commodity_name, quantity_qtl,
                    vehicle_type, recommended_market,
                    recommended_net_profit, alt_market, net_profit_difference
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                origin_name, commodity_name, quantity_qtl,
                vehicle_type, recommended_market,
                recommended_net_profit, alt_market, net_profit_difference
            ))
            conn.commit()

    def get_recent_decisions(self, limit: int = 10) -> pd.DataFrame:
        """Returns the most recent logged farmer decision queries."""
        with self.get_connection() as conn:
            return pd.read_sql_query(
                "SELECT * FROM farmer_decisions ORDER BY id DESC LIMIT ?",
                conn,
                params=(limit,)
            )

    def insert_daily_price(
        self,
        market_id: int,
        commodity_id: int,
        date_str: str,
        arrivals: float,
        min_p: float,
        max_p: float,
        modal_p: float
    ):
        """Insert or replace a single daily price record."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO daily_prices (
                    market_id, commodity_id, date, arrivals_in_qtl,
                    min_price, max_price, modal_price
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (market_id, commodity_id, date_str, arrivals, min_p, max_p, modal_p))
            conn.commit()
