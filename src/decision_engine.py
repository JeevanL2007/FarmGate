"""
FarmGate Smart Selling Decision Engine
Calculates gross revenues, transport expenses, APMC cess, labor charges,
and ranks markets based on net earnings to recommend the optimal destination.
"""

import math
from typing import Dict, List, Any, Optional
import pandas as pd

from .utils import (
    haversine_distance_km,
    VEHICLE_PROFILES,
    get_default_vehicle_for_quantity
)


class DecisionResult:
    def __init__(
        self,
        market_id: int,
        market_name: str,
        district: str,
        distance_km: float,
        quantity_qtl: float,
        modal_price: float,
        min_price: float,
        max_price: float,
        gross_revenue: float,
        transport_cost: float,
        labor_cost: float,
        apmc_cess: float,
        total_deductions: float,
        net_profit: float,
        net_price_per_qtl: float,
        net_price_per_kg: float,
        expense_ratio_pct: float,
        rank: int = 0,
        is_nearest: bool = False
    ):
        self.market_id = market_id
        self.market_name = market_name
        self.district = district
        self.distance_km = distance_km
        self.quantity_qtl = quantity_qtl
        self.modal_price = modal_price
        self.min_price = min_price
        self.max_price = max_price
        self.gross_revenue = gross_revenue
        self.transport_cost = transport_cost
        self.labor_cost = labor_cost
        self.apmc_cess = apmc_cess
        self.total_deductions = total_deductions
        self.net_profit = net_profit
        self.net_price_per_qtl = net_price_per_qtl
        self.net_price_per_kg = net_price_per_kg
        self.expense_ratio_pct = expense_ratio_pct
        self.rank = rank
        self.is_nearest = is_nearest

    def to_dict(self) -> Dict[str, Any]:
        return {
            "Rank": self.rank,
            "Market": self.market_name,
            "District": self.district,
            "Distance (km)": round(self.distance_km, 1),
            "Market Modal Price (₹/qtl)": round(self.modal_price, 2),
            "Gross Revenue (₹)": round(self.gross_revenue, 2),
            "Transport Cost (₹)": round(self.transport_cost, 2),
            "Labor/Handling (₹)": round(self.labor_cost, 2),
            "APMC Cess (₹)": round(self.apmc_cess, 2),
            "Total Expenses (₹)": round(self.total_deductions, 2),
            "Net Profit (₹)": round(self.net_profit, 2),
            "Realized Price (₹/qtl)": round(self.net_price_per_qtl, 2),
            "Realized Price (₹/kg)": round(self.net_price_per_kg, 2),
            "Expense Burden (%)": round(self.expense_ratio_pct, 1),
            "Is Nearest": self.is_nearest
        }


class SmartSellingEngine:
    def __init__(self):
        pass

    def evaluate_markets(
        self,
        candidate_markets_df: pd.DataFrame,
        farmer_lat: float,
        farmer_lon: float,
        quantity_qtl: float,
        vehicle_type: Optional[str] = None,
        custom_rate_per_km: Optional[float] = None,
        custom_loading_per_qtl: Optional[float] = None,
        price_basis: str = "modal_price"  # 'modal_price', 'min_price', or 'max_price'
    ) -> List[DecisionResult]:
        """
        Evaluates and ranks all markets for a given commodity harvest.
        
        Parameters:
            candidate_markets_df: DataFrame with market metadata and current prices.
                                  Must include columns: market_id, market_name, district,
                                  latitude, longitude, cess_rate, handling_charge_per_qtl,
                                  min_price, max_price, modal_price.
            farmer_lat: Origin latitude
            farmer_lon: Origin longitude
            quantity_qtl: Total harvest in quintals (1 qtl = 100 kg)
            vehicle_type: Vehicle model chosen (or auto-chosen based on quantity)
            custom_rate_per_km: Optional user override for transport rate (₹/km)
            custom_loading_per_qtl: Optional user override for labor
            price_basis: Which price field to use for revenue calculation
        """
        if candidate_markets_df.empty:
            return []

        if not vehicle_type or vehicle_type not in VEHICLE_PROFILES:
            vehicle_type = get_default_vehicle_for_quantity(quantity_qtl)

        v_profile = VEHICLE_PROFILES[vehicle_type]
        base_rate_per_km = custom_rate_per_km if custom_rate_per_km is not None else v_profile["base_rate_per_km"]
        loading_per_qtl = custom_loading_per_qtl if custom_loading_per_qtl is not None else v_profile["loading_unloading_per_qtl"]
        max_cap = v_profile["max_capacity_qtl"]

        # Number of vehicle trips required if quantity exceeds single vehicle capacity
        num_trips = max(1, math.ceil(quantity_qtl / max_cap))

        results: List[DecisionResult] = []

        # Find nearest market first
        min_dist = float("inf")
        nearest_mkt_id = None
        distances = {}
        for _, row in candidate_markets_df.iterrows():
            d = haversine_distance_km(farmer_lat, farmer_lon, float(row["latitude"]), float(row["longitude"]))
            distances[row["market_id"]] = d
            if d < min_dist:
                min_dist = d
                nearest_mkt_id = row["market_id"]

        for _, row in candidate_markets_df.iterrows():
            m_id = row["market_id"]
            dist_km = distances[m_id]

            # Price selection
            selected_price = float(row.get(price_basis, row["modal_price"]))
            min_p = float(row["min_price"])
            max_p = float(row["max_price"])
            modal_p = float(row["modal_price"])

            # 1. Gross Revenue
            gross_rev = quantity_qtl * selected_price

            # 2. Transportation cost: distance * rate * trips
            transport_exp = dist_km * base_rate_per_km * num_trips

            # 3. Labor & Handling: farmer loading + APMC unloading/weighing
            mkt_handling = float(row.get("handling_charge_per_qtl", 15.0))
            total_labor_rate = loading_per_qtl + mkt_handling
            labor_exp = quantity_qtl * total_labor_rate

            # 4. APMC Market Cess
            cess_rate = float(row.get("cess_rate", 1.5))
            cess_exp = gross_rev * (cess_rate / 100.0)

            # 5. Total Expenses & Net Profit
            total_expenses = transport_exp + labor_exp + cess_exp
            net_profit = gross_rev - total_expenses
            net_price_per_qtl = net_profit / quantity_qtl if quantity_qtl > 0 else 0
            net_price_per_kg = net_price_per_qtl / 100.0
            expense_ratio = (total_expenses / gross_rev * 100.0) if gross_rev > 0 else 0.0

            results.append(
                DecisionResult(
                    market_id=m_id,
                    market_name=row["market_name"],
                    district=row["district"],
                    distance_km=dist_km,
                    quantity_qtl=quantity_qtl,
                    modal_price=modal_p,
                    min_price=min_p,
                    max_price=max_p,
                    gross_revenue=gross_rev,
                    transport_cost=transport_exp,
                    labor_cost=labor_exp,
                    apmc_cess=cess_exp,
                    total_deductions=total_expenses,
                    net_profit=net_profit,
                    net_price_per_qtl=net_price_per_qtl,
                    net_price_per_kg=net_price_per_kg,
                    expense_ratio_pct=expense_ratio,
                    is_nearest=(m_id == nearest_mkt_id)
                )
            )

        # Rank markets by net profit descending
        results.sort(key=lambda x: x.net_profit, reverse=True)
        for idx, res in enumerate(results, start=1):
            res.rank = idx

        return results

    def generate_recommendation_summary(
        self,
        results: List[DecisionResult],
        commodity_name: str,
        origin_name: str
    ) -> Dict[str, Any]:
        """
        Builds a comprehensive decision summary including profit gain,
        comparison with local market, and strategic takeaways.
        """
        if not results:
            return {"status": "error", "message": "No candidate markets available."}

        best = results[0]
        nearest = next((r for r in results if r.is_nearest), best)

        profit_gain_vs_nearest = best.net_profit - nearest.net_profit
        extra_distance_km = best.distance_km - nearest.distance_km

        is_best_the_nearest = (best.market_id == nearest.market_id)

        # Strategic advice
        if is_best_the_nearest:
            strategy_text = (
                f"Sell locally at **{best.market_name}** ({best.distance_km:.1f} km). "
                f"Even though other markets may post higher nominal prices, the extra transportation "
                f"costs to reach distant markets will diminish your net margin. Your local market yields "
                f"the highest net profit of **₹ {best.net_profit:,.2f}**."
            )
        else:
            strategy_text = (
                f"Transport your produce to **{best.market_name}** ({best.distance_km:.1f} km) instead of "
                f"local market **{nearest.market_name}** ({nearest.distance_km:.1f} km). "
                f"Even after paying an extra ₹ {best.transport_cost - nearest.transport_cost:,.2f} in transport expenses, "
                f"you will earn an extra net profit of **₹ {profit_gain_vs_nearest:,.2f}** "
                f"(+{((profit_gain_vs_nearest / nearest.net_profit) * 100):.1f}% boost)."
            )

        # Break-even threshold for nearest market
        # How much would nearest market need to pay to match the best market's net profit?
        required_nearest_gross = best.net_profit + nearest.transport_cost + nearest.labor_cost
        # required_gross = Q * P - cess% * (Q * P) = Q * P * (1 - cess/100)
        # P = required_gross / (Q * (1 - cess/100))
        cess_multiplier = (1.0 - (nearest.apmc_cess / nearest.gross_revenue if nearest.gross_revenue > 0 else 0.015))
        break_even_price = (required_nearest_gross / (best.quantity_qtl * cess_multiplier)) if best.quantity_qtl > 0 else 0

        return {
            "status": "success",
            "best_market": best,
            "nearest_market": nearest,
            "profit_gain_vs_nearest": profit_gain_vs_nearest,
            "extra_distance_km": extra_distance_km,
            "is_best_the_nearest": is_best_the_nearest,
            "break_even_nearest_price": round(break_even_price, 2),
            "strategy_text": strategy_text,
            "commodity_name": commodity_name,
            "origin_name": origin_name,
            "ranking_table": pd.DataFrame([r.to_dict() for r in results])
        }
