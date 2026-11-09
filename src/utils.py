"""
FarmGate Utility Module
Includes distance calculations, Karnataka geography presets, vehicle models,
and formatting utilities.
"""

import math
from typing import Dict, Any, Tuple, Optional

# Road winding factor: Highway and rural roads are typically ~25% longer than straight-line distance
ROAD_WINDING_FACTOR = 1.25

# Preset Farmer Origin Hubs in Karnataka (Taluk / Farming clusters)
KARNATAKA_FARMER_ORIGINS = {
    "Malur (Kolar District)": {"lat": 13.0038, "lon": 77.9398, "district": "Kolar"},
    "Srinivasapur (Kolar District)": {"lat": 13.3377, "lon": 78.2144, "district": "Kolar"},
    "Bangarapet (Kolar District)": {"lat": 12.9818, "lon": 78.2039, "district": "Kolar"},
    "Chintamani (Chikkaballapur District)": {"lat": 13.4005, "lon": 78.0583, "district": "Chikkaballapur"},
    "Devanahalli (Bengaluru Rural)": {"lat": 13.2483, "lon": 77.7126, "district": "Bengaluru Rural"},
    "Doddaballapura (Bengaluru Rural)": {"lat": 13.2929, "lon": 77.5385, "district": "Bengaluru Rural"},
    "Ramanagara (Ramanagara District)": {"lat": 12.7209, "lon": 77.2799, "district": "Ramanagara"},
    "Channapatna (Ramanagara District)": {"lat": 12.6518, "lon": 77.2089, "district": "Ramanagara"},
    "Maddur (Mandya District)": {"lat": 12.5844, "lon": 77.0450, "district": "Mandya"},
    "Mandya (Mandya District)": {"lat": 12.5238, "lon": 76.8984, "district": "Mandya"},
    "Pandavapura (Mandya District)": {"lat": 12.4965, "lon": 76.6713, "district": "Mandya"},
    "Nanjangud (Mysuru District)": {"lat": 12.1190, "lon": 76.6789, "district": "Mysuru"},
    "Hunsur (Mysuru District)": {"lat": 12.3082, "lon": 76.2894, "district": "Mysuru"},
    "Tiptur (Tumakuru District)": {"lat": 13.2625, "lon": 76.4789, "district": "Tumakuru"},
    "Chiknayakanhalli (Tumakuru District)": {"lat": 13.4183, "lon": 76.6200, "district": "Tumakuru"},
    "Channagiri (Davanagere District)": {"lat": 14.0258, "lon": 75.9272, "district": "Davanagere"},
    "Bhadravati (Shivamogga District)": {"lat": 13.8436, "lon": 75.7034, "district": "Shivamogga"},
    "Haveri (Haveri District)": {"lat": 14.7964, "lon": 75.4024, "district": "Haveri"},
    "Ranebennur (Haveri District)": {"lat": 14.6214, "lon": 75.6268, "district": "Haveri"},
    "Bailhongal (Belagavi District)": {"lat": 15.8143, "lon": 74.8569, "district": "Belagavi"},
    "Gokak (Belagavi District)": {"lat": 16.1683, "lon": 74.8258, "district": "Belagavi"},
    "Alur (Hassan District)": {"lat": 12.9840, "lon": 76.0125, "district": "Hassan"},
    "Channarayapatna (Hassan District)": {"lat": 12.9056, "lon": 76.3897, "district": "Hassan"}
}

# Standard Commercial Goods Vehicles used by Karnataka Farmers
VEHICLE_PROFILES = {
    "Mini Truck / Tata Ace (Chota Hathi)": {
        "max_capacity_qtl": 15.0,  # ~1.5 Tons
        "base_rate_per_km": 16.0,  # ₹ per km
        "loading_unloading_per_qtl": 12.0,
        "description": "Ideal for small harvests (up to 15 Quintals / 30 bags)"
    },
    "Bolero Maxi Truck / Pickup": {
        "max_capacity_qtl": 35.0,  # ~3.5 Tons
        "base_rate_per_km": 24.0,  # ₹ per km
        "loading_unloading_per_qtl": 14.0,
        "description": "Medium loads (15 - 35 Quintals / up to 70 bags)"
    },
    "Canter 407 / 14ft Eicher Truck": {
        "max_capacity_qtl": 75.0,  # ~7.5 Tons
        "base_rate_per_km": 38.0,  # ₹ per km
        "loading_unloading_per_qtl": 15.0,
        "description": "Large wholesale shipments (35 - 75 Quintals)"
    },
    "Heavy 6/10-Wheel Truck": {
        "max_capacity_qtl": 160.0, # ~16 Tons
        "base_rate_per_km": 58.0,  # ₹ per km
        "loading_unloading_per_qtl": 16.0,
        "description": "Inter-district bulk transport (75 - 160 Quintals)"
    }
}


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes great-circle distance between two GPS coordinates using the Haversine formula,
    adjusted by the road winding factor.
    """
    R = 6371.0  # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    straight_line_dist = R * c
    return round(straight_line_dist * ROAD_WINDING_FACTOR, 1)


def get_default_vehicle_for_quantity(quantity_qtl: float) -> str:
    """Selects the most suitable vehicle based on produce quantity."""
    if quantity_qtl <= 15:
        return "Mini Truck / Tata Ace (Chota Hathi)"
    elif quantity_qtl <= 35:
        return "Bolero Maxi Truck / Pickup"
    elif quantity_qtl <= 75:
        return "Canter 407 / 14ft Eicher Truck"
    else:
        return "Heavy 6/10-Wheel Truck"


def format_currency(amount: float) -> str:
    """Formats a number as Indian Rupee (₹)."""
    return f"₹ {amount:,.2f}"


def format_quintal_price(amount: float) -> str:
    """Formats price per quintal with per-kg equivalent."""
    per_kg = amount / 100.0
    return f"₹ {amount:,.0f}/qtl (₹ {per_kg:.2f}/kg)"
