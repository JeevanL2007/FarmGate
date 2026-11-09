"""
FarmGate Live Agmarknet Sync Module
Fetches real-time market price data from government open data APIs (data.gov.in / Agmarknet).
"""

import os
import requests
from typing import Dict, Any, List, Optional
import pandas as pd

# Open Government Data (data.gov.in) Agmarknet Resource Endpoint
DATA_GOV_AGMARKNET_URL = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"


class AgmarknetLiveSync:
    def __init__(self, api_key: Optional[str] = None):
        """
        api_key: Optional personal API key obtained from data.gov.in.
        If no API key is provided, the system uses mock/pre-seeded data to avoid runtime downtime.
        """
        self.api_key = api_key or os.getenv("DATA_GOV_IN_API_KEY", "")

    def fetch_live_karnataka_prices(
        self,
        commodity: str = "Tomato",
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Pulls real-time Agmarknet commodity price entries for Karnataka.
        """
        if not self.api_key:
            return []

        params = {
            "api-key": self.api_key,
            "format": "json",
            "offset": 0,
            "limit": limit,
            "filters[state]": "Karnataka",
            "filters[commodity]": commodity
        }

        try:
            response = requests.get(DATA_GOV_AGMARKNET_URL, params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                records = data.get("records", [])
                return records
            else:
                return []
        except Exception as e:
            print(f"Agmarknet API fetch notice: {e}")
            return []
