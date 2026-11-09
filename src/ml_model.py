"""
FarmGate Machine Learning Price Forecasting Module
Trains regression models on historical APMC crop prices and generates
multi-day future price estimates with confidence intervals and sell/hold signals.
"""

import math
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class PriceForecaster:
    def __init__(self, model_type: str = "RandomForest"):
        """
        model_type: 'RandomForest', 'GradientBoosting', or 'Ridge'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names: List[str] = []
        self.metrics: Dict[str, float] = {}
        self.residual_std: float = 50.0  # Default fallback residual standard deviation

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineers time-series features: lags, moving averages, arrival momentum,
        and cyclical date features.
        """
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # Lags on modal price
        df["lag_1"] = df["modal_price"].shift(1)
        df["lag_2"] = df["modal_price"].shift(2)
        df["lag_3"] = df["modal_price"].shift(3)
        df["lag_7"] = df["modal_price"].shift(7)
        df["lag_14"] = df["modal_price"].shift(14)

        # Rolling statistics
        df["rolling_mean_7"] = df["modal_price"].shift(1).rolling(window=7, min_periods=3).mean()
        df["rolling_std_7"] = df["modal_price"].shift(1).rolling(window=7, min_periods=3).std().fillna(10.0)
        df["rolling_mean_14"] = df["modal_price"].shift(1).rolling(window=14, min_periods=5).mean()

        # Arrivals momentum
        df["arrivals_lag_1"] = df["arrivals_in_qtl"].shift(1)
        df["arrivals_rolling_7"] = df["arrivals_in_qtl"].shift(1).rolling(window=7, min_periods=3).mean()

        # Calendar features
        df["dayofweek"] = df["date"].dt.dayofweek
        df["month"] = df["date"].dt.month
        df["day"] = df["date"].dt.day

        # Cyclical month features
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)

        # Momentum / Rate of Change
        df["pct_change_7"] = (df["lag_1"] - df["lag_7"]) / (df["lag_7"] + 1e-5)

        # Drop rows with NaN caused by lagging
        df = df.dropna().reset_index(drop=True)
        return df

    def train(self, historical_df: pd.DataFrame, test_size_days: int = 30) -> Dict[str, Any]:
        """
        Trains the selected regression model on time-ordered historical records.
        Returns model performance metrics on the out-of-time holdout set.
        """
        if len(historical_df) < 40:
            raise ValueError("Insufficient data to train model. Need at least 40 daily records.")

        processed_df = self._engineer_features(historical_df)
        if len(processed_df) < 20:
            raise ValueError("Data too short after feature lagging.")

        feature_cols = [
            "lag_1", "lag_2", "lag_3", "lag_7", "lag_14",
            "rolling_mean_7", "rolling_std_7", "rolling_mean_14",
            "arrivals_lag_1", "arrivals_rolling_7",
            "dayofweek", "month", "month_sin", "month_cos", "pct_change_7"
        ]
        self.feature_names = feature_cols

        X = processed_df[feature_cols]
        y = processed_df["modal_price"]

        # Time-based train / test split
        split_idx = max(len(processed_df) - test_size_days, int(len(processed_df) * 0.8))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Initialize model
        if self.model_type == "RandomForest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        elif self.model_type == "GradientBoosting":
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
        elif self.model_type == "Ridge":
            self.model = Ridge(alpha=1.0)
        else:
            self.model = LinearRegression()

        self.model.fit(X_train, y_train)

        # Evaluate on test set
        y_pred = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-5))) * 100.0

        residuals = y_test - y_pred
        self.residual_std = float(np.std(residuals)) if len(residuals) > 1 else 30.0

        self.metrics = {
            "MAE": round(float(mae), 2),
            "RMSE": round(float(rmse), 2),
            "R2": round(float(r2), 3),
            "MAPE_pct": round(float(mape), 2),
            "train_samples": len(X_train),
            "test_samples": len(X_test)
        }

        # Feature importances if tree-based
        feature_importance = {}
        if hasattr(self.model, "feature_importances_"):
            for name, imp in zip(feature_cols, self.model.feature_importances_):
                feature_importance[name] = round(float(imp), 4)

        return {
            "metrics": self.metrics,
            "feature_importance": feature_importance,
            "y_test": y_test.values.tolist(),
            "y_pred": y_pred.tolist(),
            "test_dates": processed_df.iloc[split_idx:]["date"].dt.strftime("%Y-%m-%d").tolist()
        }

    def forecast_next_days(
        self,
        historical_df: pd.DataFrame,
        horizon_days: int = 7
    ) -> pd.DataFrame:
        """
        Recursively projects future crop prices for the next horizon_days (e.g. 7-14 days),
        generating projected modal price, lower bound, upper bound, and daily trend.
        """
        if self.model is None:
            raise ValueError("Model must be trained before forecasting.")

        df = historical_df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        forecasts = []
        last_date = df["date"].max()

        # Work on a dynamic sliding buffer
        buffer_df = df[["date", "modal_price", "arrivals_in_qtl"]].copy()

        for step in range(1, horizon_days + 1):
            next_date = last_date + pd.Timedelta(days=step)

            # Extract features from buffer
            recent_prices = buffer_df["modal_price"].values
            recent_arrivals = buffer_df["arrivals_in_qtl"].values

            lag_1 = recent_prices[-1]
            lag_2 = recent_prices[-2] if len(recent_prices) >= 2 else lag_1
            lag_3 = recent_prices[-3] if len(recent_prices) >= 3 else lag_2
            lag_7 = recent_prices[-7] if len(recent_prices) >= 7 else recent_prices[0]
            lag_14 = recent_prices[-14] if len(recent_prices) >= 14 else recent_prices[0]

            window_7 = recent_prices[-7:] if len(recent_prices) >= 7 else recent_prices
            rolling_mean_7 = np.mean(window_7)
            rolling_std_7 = np.std(window_7) if len(window_7) > 1 else 10.0

            window_14 = recent_prices[-14:] if len(recent_prices) >= 14 else recent_prices
            rolling_mean_14 = np.mean(window_14)

            arrivals_lag_1 = recent_arrivals[-1]
            arrivals_rolling_7 = np.mean(recent_arrivals[-7:]) if len(recent_arrivals) >= 7 else arrivals_lag_1

            dayofweek = next_date.dayofweek
            month = next_date.month
            month_sin = np.sin(2 * np.pi * month / 12.0)
            month_cos = np.cos(2 * np.pi * month / 12.0)
            pct_change_7 = (lag_1 - lag_7) / (lag_7 + 1e-5)

            feat_row = pd.DataFrame([{
                "lag_1": lag_1,
                "lag_2": lag_2,
                "lag_3": lag_3,
                "lag_7": lag_7,
                "lag_14": lag_14,
                "rolling_mean_7": rolling_mean_7,
                "rolling_std_7": rolling_std_7,
                "rolling_mean_14": rolling_mean_14,
                "arrivals_lag_1": arrivals_lag_1,
                "arrivals_rolling_7": arrivals_rolling_7,
                "dayofweek": dayofweek,
                "month": month,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "pct_change_7": pct_change_7
            }])[self.feature_names]

            pred_price = float(self.model.predict(feat_row)[0])
            pred_price = max(100.0, pred_price)  # Ensure non-negative crop price

            # Confidence interval grows slightly with horizon step: sqrt(step) * std
            margin = 1.96 * self.residual_std * math.sqrt(1 + 0.1 * step)
            lower_bound = max(50.0, pred_price - margin)
            upper_bound = pred_price + margin

            forecasts.append({
                "Date": next_date.strftime("%Y-%m-%d"),
                "Day": next_date.strftime("%a"),
                "Forecasted Price (₹/qtl)": round(pred_price, 2),
                "Lower CI (₹/qtl)": round(lower_bound, 2),
                "Upper CI (₹/qtl)": round(upper_bound, 2)
            })

            # Append prediction to buffer for next multi-step lag
            new_row = pd.DataFrame([{
                "date": next_date,
                "modal_price": pred_price,
                "arrivals_in_qtl": arrivals_rolling_7
            }])
            buffer_df = pd.concat([buffer_df, new_row], ignore_index=True)

        forecast_df = pd.DataFrame(forecasts)
        return forecast_df

    def generate_selling_timing_signal(
        self,
        current_price: float,
        forecast_df: pd.DataFrame,
        perishable_level: str = "Medium"
    ) -> Dict[str, Any]:
        """
        Analyzes the projected price trajectory to generate a farmer actionable signal:
        e.g. 'SELL IMMEDIATELY' vs 'HOLD FOR 3-5 DAYS'.
        """
        if forecast_df.empty:
            return {"signal": "NEUTRAL", "rationale": "No forecast available."}

        avg_future_price = forecast_df["Forecasted Price (₹/qtl)"].iloc[:4].mean()
        price_diff = avg_future_price - current_price
        pct_change = (price_diff / current_price) * 100.0

        if pct_change >= 6.0:
            if perishable_level == "High":
                signal = "HOLD WITH CAUTION (1-2 DAYS)"
                rationale = (
                    f"Prices are projected to increase by ~{pct_change:.1f}% over the next few days. "
                    f"However, this crop has high perishability, so store in cool conditions or sell within 48 hours."
                )
            else:
                signal = "HOLD FOR HIGHER GAINS"
                rationale = (
                    f"Bullish trend predicted: Market expected to gain ~₹ {price_diff:.1f}/qtl (+{pct_change:.1f}%). "
                    f"Holding harvest for 3-5 days can maximize your returns."
                )
        elif pct_change <= -4.0:
            signal = "SELL IMMEDIATELY"
            rationale = (
                f"Bearish trend detected: Prices are projected to decline by ~₹ {abs(price_diff):.1f}/qtl "
                f"({pct_change:.1f}%) due to expected arrival increases. Sell today to avoid losses."
            )
        else:
            signal = "STABLE / SELL AT WILL"
            rationale = (
                f"Market prices are expected to remain steady within a ±{abs(pct_change):.1f}% band. "
                f"Sell according to your harvesting schedule without urgency."
            )

        return {
            "signal": signal,
            "pct_change": round(pct_change, 1),
            "price_diff": round(price_diff, 1),
            "rationale": rationale,
            "trend": "Bullish" if pct_change > 2 else ("Bearish" if pct_change < -2 else "Sideways")
        }
