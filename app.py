"""
FarmGate – Intelligent Agricultural Market Price Comparison and Smart Selling Decision Support System
A Streamlit web application for Karnataka agricultural markets.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.database import FarmGateDB
from src.decision_engine import SmartSellingEngine
from src.ml_model import PriceForecaster
from src.utils import (
    KARNATAKA_FARMER_ORIGINS,
    VEHICLE_PROFILES,
    get_default_vehicle_for_quantity,
    format_currency,
    format_quintal_price,
    haversine_distance_km
)
from src.data_loader import seed_database_and_generate_data

# Page Configuration
st.set_page_config(
    page_title="FarmGate – Smart Selling Decision Support",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 50%, #388e3c 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        margin-bottom: 6px;
        font-weight: 700;
    }
    .main-header p {
        color: #e8f5e9;
        font-size: 1.05rem;
        margin-bottom: 0;
    }
    .metric-card {
        background-color: #f8fafc;
        border-left: 5px solid #2e7d32;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }
    .profit-badge {
        background-color: #e8f5e9;
        color: #1b5e20;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #81c784;
    }
    .warning-badge {
        background-color: #fff3e0;
        color: #e65100;
        font-weight: bold;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid #ffb74d;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_db():
    db = FarmGateDB()
    # Check if database has data, if not seed it
    markets = db.get_markets()
    if markets.empty:
        seed_database_and_generate_data(db, num_days=180)
    return db


db = get_db()
decision_engine = SmartSellingEngine()

# Sidebar Navigation
st.sidebar.image("https://images.unsplash.com/photo-1595974482597-4b8da8879bc5?auto=format&fit=crop&w=600&q=80", use_container_width=True)
st.sidebar.title("🌾 FarmGate")
st.sidebar.caption("Intelligent Decision Support Platform")

menu_selection = st.sidebar.radio(
    "Navigation Menu",
    [
        "🚚 Smart Selling Recommender",
        "📊 Market Price Comparison",
        "📈 Historical Trends & Analytics",
        "🤖 ML Price Forecasting",
        "🗄️ Database & Price Ingestion",
        "ℹ️ Project Synopsis & Team"
    ]
)

st.sidebar.markdown("---")
# Quick stats in sidebar
markets_df = db.get_markets()
commodities_df = db.get_commodities()
st.sidebar.markdown(f"**Markets Monitored:** {len(markets_df)} APMCs")
st.sidebar.markdown(f"**Commodities Tracked:** {len(commodities_df)} crops")
st.sidebar.info("💡 **Tip**: Change origin and harvest quantity to compare real net earnings after transport.")


# ---------------------------------------------------------
# MODULE 1: SMART SELLING RECOMMENDER
# ---------------------------------------------------------
if menu_selection == "🚚 Smart Selling Recommender":
    st.markdown("""
    <div class="main-header">
        <h1>🚚 Smart Selling Decision Engine</h1>
        <p>Calculates net farmer earnings across Karnataka APMC markets by factoring in transport freight, loading labor, and APMC cess.</p>
    </div>
    """, unsafe_allow_html=True)

    col_in1, col_in2, col_in3 = st.columns([1.2, 1.2, 1.2])

    with col_in1:
        st.subheader("1. Crop & Harvest")
        comm_names = commodities_df["name"].tolist()
        comm_display = [f"{row['name']} ({row['kannada_name']})" for _, row in commodities_df.iterrows()]
        selected_comm_idx = st.selectbox(
            "Select Crop / ಬೆಳೆ",
            range(len(commodities_df)),
            format_func=lambda i: comm_display[i]
        )
        selected_commodity = commodities_df.iloc[selected_comm_idx]
        comm_id = int(selected_commodity["id"])
        comm_name = selected_commodity["name"]

        harvest_qtl = st.number_input(
            "Produce Quantity (in Quintals / ಕ್ವಿಂಟಾಲ್)",
            min_value=1.0,
            max_value=500.0,
            value=25.0,
            step=1.0,
            help="1 Quintal = 100 kg"
        )
        bags_approx = harvest_qtl * (100.0 / selected_commodity["standard_bag_weight_kg"])
        st.caption(f"📦 Approx **{int(bags_approx)} standard bags/crates** ({harvest_qtl * 100:.0f} kg)")

    with col_in2:
        st.subheader("2. Farmer Origin")
        origin_names = list(KARNATAKA_FARMER_ORIGINS.keys())
        selected_origin_name = st.selectbox(
            "Farmer Location / ರೈತರ ಸ್ಥಳ (Taluk)",
            origin_names,
            index=0
        )
        origin_info = KARNATAKA_FARMER_ORIGINS[selected_origin_name]
        farmer_lat = origin_info["lat"]
        farmer_lon = origin_info["lon"]
        st.caption(f"📍 District: **{origin_info['district']}** (Lat: {farmer_lat:.2f}, Lon: {farmer_lon:.2f})")

        price_basis = st.radio(
            "Price Basis for Revenue Calculation",
            ["modal_price", "min_price", "max_price"],
            format_func=lambda x: "Modal Price (Most common trade rate)" if x == "modal_price" else ("Minimum Price (Conservative)" if x == "min_price" else "Maximum Price (Grade A)"),
            horizontal=False
        )

    with col_in3:
        st.subheader("3. Vehicle & Freight Settings")
        default_veh = get_default_vehicle_for_quantity(harvest_qtl)
        veh_keys = list(VEHICLE_PROFILES.keys())
        selected_veh = st.selectbox(
            "Transportation Vehicle",
            veh_keys,
            index=veh_keys.index(default_veh)
        )
        veh_spec = VEHICLE_PROFILES[selected_veh]
        st.caption(f"ℹ️ {veh_spec['description']} (Capacity: {veh_spec['max_capacity_qtl']} qtl)")

        with st.expander("⚙️ Customize Rates (Optional)"):
            custom_rate = st.number_input("Freight Rate (₹ / km)", value=float(veh_spec["base_rate_per_km"]), step=1.0)
            custom_labor = st.number_input("Loading Labor (₹ / quintal)", value=float(veh_spec["loading_unloading_per_qtl"]), step=1.0)

    st.markdown("---")

    # Fetch latest prices for selected commodity
    latest_prices_df = db.get_latest_prices(commodity_id=comm_id)

    if latest_prices_df.empty:
        st.warning("No price data found for the selected commodity. Please seed the database.")
    else:
        results = decision_engine.evaluate_markets(
            candidate_markets_df=latest_prices_df,
            farmer_lat=farmer_lat,
            farmer_lon=farmer_lon,
            quantity_qtl=harvest_qtl,
            vehicle_type=selected_veh,
            custom_rate_per_km=custom_rate,
            custom_loading_per_qtl=custom_labor,
            price_basis=price_basis
        )

        summary = decision_engine.generate_recommendation_summary(
            results=results,
            commodity_name=comm_name,
            origin_name=selected_origin_name
        )

        best = summary["best_market"]
        nearest = summary["nearest_market"]
        profit_delta = summary["profit_gain_vs_nearest"]

        # Recommendation Banner
        st.subheader("🎯 Recommendation Result")
        
        banner_col1, banner_col2, banner_col3, banner_col4 = st.columns([1.5, 1, 1, 1])
        with banner_col1:
            st.metric(
                label="🌟 Recommended Market Destination",
                value=best.market_name,
                delta=f"{best.distance_km:.1f} km away"
            )
        with banner_col2:
            st.metric(
                label="Estimated Net Profit",
                value=format_currency(best.net_profit),
                delta=f"+{format_currency(profit_delta)} vs nearest" if not summary["is_best_the_nearest"] else "Best local option"
            )
        with banner_col3:
            st.metric(
                label="Net Realized Price",
                value=f"₹ {best.net_price_per_qtl:,.1f}/qtl",
                delta=f"₹ {best.net_price_per_kg:.2f}/kg"
            )
        with banner_col4:
            st.metric(
                label="Freight & APMC Deduction",
                value=format_currency(best.total_deductions),
                delta=f"{best.expense_ratio_pct:.1f}% of gross",
                delta_color="inverse"
            )

        # Strategic Guidance Text Card
        st.success(f"💡 **Strategy Advice:** {summary['strategy_text']}")

        if not summary["is_best_the_nearest"]:
            st.info(
                f"⚖️ **Break-even Price Insight**: For your local market (**{nearest.market_name}**) to match this net profit, "
                f"they would need to offer at least **₹ {summary['break_even_nearest_price']:,.2f}/quintal** "
                f"(Current local price: ₹ {nearest.modal_price:,.2f}/qtl)."
            )

        st.markdown("### 📊 Market Comparison & Profit Waterfall")
        tab_chart, tab_table, tab_breakdown = st.tabs(["📈 Profit & Cost Chart", "📋 Complete Market Ranking Table", "🔍 Cost Waterfall Breakdown"])

        ranking_df = summary["ranking_table"]

        with tab_chart:
            # Bar chart comparing Gross vs Deductions vs Net
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                x=ranking_df["Market"],
                y=ranking_df["Net Profit (₹)"],
                name="Net Profit (Take-Home)",
                marker_color="#2e7d32"
            ))
            fig_bar.add_trace(go.Bar(
                x=ranking_df["Market"],
                y=ranking_df["Total Expenses (₹)"],
                name="Total Deductions (Transport + Cess + Labor)",
                marker_color="#d32f2f"
            ))
            fig_bar.update_layout(
                barmode="stack",
                title=f"Net Profit vs Total Deductions for {harvest_qtl} Quintals of {comm_name}",
                xaxis_title="APMC Market",
                yaxis_title="Amount in Indian Rupees (₹)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Bubble Chart: Distance vs Realized Price
            fig_bubble = px.scatter(
                ranking_df,
                x="Distance (km)",
                y="Realized Price (₹/qtl)",
                size="Net Profit (₹)",
                color="Net Profit (₹)",
                hover_name="Market",
                text="Market",
                color_continuous_scale="Greens",
                title=f"Distance vs Realized Price per Quintal (Bubble Size = Net Take-Home Profit)"
            )
            fig_bubble.update_traces(textposition='top center')
            st.plotly_chart(fig_bubble, use_container_width=True)

        with tab_table:
            # Format and present ranking table
            styled_df = ranking_df.copy()
            st.dataframe(styled_df, use_container_width=True)

            csv_data = ranking_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Decision Report (CSV)",
                data=csv_data,
                file_name=f"FarmGate_Decision_{comm_name}_{selected_origin_name.split()[0]}.csv",
                mime="text/csv"
            )

        with tab_breakdown:
            st.markdown(f"#### Expense Breakdown for Recommended: **{best.market_name}**")
            breakdown_cols = st.columns(4)
            with breakdown_cols[0]:
                st.metric("Gross Harvest Revenue", format_currency(best.gross_revenue))
            with breakdown_cols[1]:
                st.metric("Transport Freight", format_currency(best.transport_cost), help=f"{best.distance_km:.1f} km at ₹ {custom_rate}/km")
            with breakdown_cols[2]:
                st.metric("Labor & Hamali", format_currency(best.labor_cost), help=f"{harvest_qtl} qtl handled")
            with breakdown_cols[3]:
                st.metric("APMC Mandi Cess", format_currency(best.apmc_cess), help="Mandatory government market cess")

            # Waterfall chart
            waterfall_fig = go.Figure(go.Waterfall(
                name="Cost Bridge",
                orientation="v",
                measure=["relative", "relative", "relative", "relative", "total"],
                x=["Gross Revenue", "Transport Freight", "Labor & Handling", "APMC Cess", "Net Earnings"],
                textposition="outside",
                text=[f"+₹{best.gross_revenue:,.0f}", f"-₹{best.transport_cost:,.0f}", f"-₹{best.labor_cost:,.0f}", f"-₹{best.apmc_cess:,.0f}", f"₹{best.net_profit:,.0f}"],
                y=[best.gross_revenue, -best.transport_cost, -best.labor_cost, -best.apmc_cess, 0],
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#ef5350"}},
                increasing={"marker": {"color": "#66bb6a"}},
                totals={"marker": {"color": "#2e7d32"}}
            ))
            waterfall_fig.update_layout(title="Revenue to Net Profit Waterfall Breakdown (₹)", showlegend=False)
            st.plotly_chart(waterfall_fig, use_container_width=True)

        # Log this decision
        if st.button("💾 Save Query to Decision History"):
            db.log_farmer_decision(
                origin_name=selected_origin_name,
                commodity_name=comm_name,
                quantity_qtl=harvest_qtl,
                vehicle_type=selected_veh,
                recommended_market=best.market_name,
                recommended_net_profit=best.net_profit,
                alt_market=nearest.market_name if not summary["is_best_the_nearest"] else None,
                net_profit_difference=profit_delta if not summary["is_best_the_nearest"] else None
            )
            st.success("Query logged successfully into database audit history!")


# ---------------------------------------------------------
# MODULE 2: MARKET PRICE COMPARISON
# ---------------------------------------------------------
elif menu_selection == "📊 Market Price Comparison":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Cross-Market Price Comparison</h1>
        <p>Compare minimum, maximum, and modal crop prices across Karnataka APMC agricultural markets.</p>
    </div>
    """, unsafe_allow_html=True)

    filter_col1, filter_col2 = st.columns(2)
    with filter_col1:
        selected_comm_name = st.selectbox(
            "Select Commodity / ಬೆಳೆ",
            commodities_df["name"].tolist()
        )
        comm_row = commodities_df[commodities_df["name"] == selected_comm_name].iloc[0]
        comm_id = int(comm_row["id"])

    with filter_col2:
        available_dates = db.get_available_dates(commodity_id=comm_id)
        selected_date = st.selectbox("Select Date / ದಿನಾಂಕ", available_dates, index=0)

    prices_df = db.get_latest_prices(commodity_id=comm_id, target_date=selected_date)

    if prices_df.empty:
        st.warning(f"No price records found for {selected_comm_name} on {selected_date}.")
    else:
        # Key Summary Metrics
        highest_row = prices_df.loc[prices_df["modal_price"].idxmax()]
        lowest_row = prices_df.loc[prices_df["modal_price"].idxmin()]
        avg_modal = prices_df["modal_price"].mean()
        price_spread = highest_row["modal_price"] - lowest_row["modal_price"]

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Highest Modal Price", f"₹ {highest_row['modal_price']:,.0f}/qtl", f"{highest_row['market_name']}")
        with m_col2:
            st.metric("Lowest Modal Price", f"₹ {lowest_row['modal_price']:,.0f}/qtl", f"{lowest_row['market_name']}")
        with m_col3:
            st.metric("State Average Modal", f"₹ {avg_modal:,.0f}/qtl")
        with m_col4:
            st.metric("Cross-Market Price Spread", f"₹ {price_spread:,.0f}/qtl", f"{((price_spread / avg_modal) * 100):.1f}% variance")

        # Price Comparison Grouped Bar Chart
        st.markdown("### 📊 Min, Modal & Max Price Comparison")
        fig_prices = go.Figure()
        fig_prices.add_trace(go.Bar(
            x=prices_df["market_name"],
            y=prices_df["min_price"],
            name="Min Price",
            marker_color="#90caf9"
        ))
        fig_prices.add_trace(go.Bar(
            x=prices_df["market_name"],
            y=prices_df["modal_price"],
            name="Modal Price (Benchmark)",
            marker_color="#2e7d32"
        ))
        fig_prices.add_trace(go.Bar(
            x=prices_df["market_name"],
            y=prices_df["max_price"],
            name="Max Price",
            marker_color="#ffb74d"
        ))
        fig_prices.update_layout(
            barmode="group",
            title=f"{selected_comm_name} Price Comparison Across Karnataka Markets ({selected_date})",
            xaxis_title="Market",
            yaxis_title="Price (₹ / Quintal)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_prices, use_container_width=True)

        # Tabular View
        st.markdown("### 📋 APMC Price & Arrival Table")
        display_table = prices_df[[
            "market_name", "district", "modal_price", "min_price", "max_price", "arrivals_in_qtl"
        ]].copy()
        display_table.columns = ["APMC Market", "District", "Modal Price (₹/qtl)", "Min Price (₹/qtl)", "Max Price (₹/qtl)", "Daily Arrivals (Quintals)"]
        display_table["Per Kg Modal Rate (₹)"] = (display_table["Modal Price (₹/qtl)"] / 100.0).round(2)
        st.dataframe(display_table, use_container_width=True)


# ---------------------------------------------------------
# MODULE 3: HISTORICAL TRENDS & VISUAL ANALYTICS
# ---------------------------------------------------------
elif menu_selection == "📈 Historical Trends & Analytics":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Historical Price Trends & Market Analytics</h1>
        <p>Analyze commodity price movements, seasonal behavior, and arrival volume dynamics across time.</p>
    </div>
    """, unsafe_allow_html=True)

    col_t1, col_t2, col_t3 = st.columns([1, 1.5, 1])
    with col_t1:
        selected_comm_name = st.selectbox("Select Commodity", commodities_df["name"].tolist())
        comm_id = int(commodities_df[commodities_df["name"] == selected_comm_name].iloc[0]["id"])
    with col_t2:
        selected_markets = st.multiselect(
            "Select Markets to Compare",
            markets_df["name"].tolist(),
            default=markets_df["name"].tolist()[:4]
        )
        selected_market_ids = markets_df[markets_df["name"].isin(selected_markets)]["id"].tolist()
    with col_t3:
        time_horizon = st.selectbox(
            "Time Horizon",
            ["Last 30 Days", "Last 60 Days", "Last 90 Days", "Full 180 Days"],
            index=2
        )
        days_map = {"Last 30 Days": 30, "Last 60 Days": 60, "Last 90 Days": 90, "Full 180 Days": 180}
        cutoff_date = (pd.to_datetime("today") - pd.Timedelta(days=days_map[time_horizon])).strftime("%Y-%m-%d")

    history_df = db.get_historical_prices(
        commodity_id=comm_id,
        market_ids=selected_market_ids,
        start_date=cutoff_date
    )

    if history_df.empty:
        st.warning("No historical records available for the selected filters.")
    else:
        # Time Series Line Chart
        st.markdown("### 📉 Multi-Market Modal Price Movement")
        fig_trend = px.line(
            history_df,
            x="date",
            y="modal_price",
            color="market_name",
            title=f"{selected_comm_name} Price Trend (₹/qtl) – {time_horizon}",
            labels={"modal_price": "Modal Price (₹/qtl)", "date": "Date", "market_name": "Market"}
        )
        fig_trend.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trend, use_container_width=True)

        # Arrival vs Price Relationship
        st.markdown("### ⚖️ Arrival Volume vs Modal Price Dynamics")
        fig_scatter = px.scatter(
            history_df,
            x="arrivals_in_qtl",
            y="modal_price",
            color="market_name",
            title=f"Supply vs Price Elasticity: Arrivals (Quintals) vs Modal Price (₹/qtl)",
            labels={"arrivals_in_qtl": "Arrivals (Quintals)", "modal_price": "Modal Price (₹/qtl)"}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # Price Distribution Boxplot
        st.markdown("### 📦 Market Price Distribution & Volatility Spread")
        fig_box = px.box(
            history_df,
            x="market_name",
            y="modal_price",
            color="market_name",
            points="outliers",
            title=f"Price Dispersion across Selected APMC Markets ({time_horizon})"
        )
        st.plotly_chart(fig_box, use_container_width=True)


# ---------------------------------------------------------
# MODULE 4: ML PRICE FORECASTING & SIGNALS
# ---------------------------------------------------------
elif menu_selection == "🤖 ML Price Forecasting":
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Machine Learning Price Forecasting Engine</h1>
        <p>Predict future crop price trends using regression algorithms trained on historical Agmarknet records, with farmer timing signals.</p>
    </div>
    """, unsafe_allow_html=True)

    ml_col1, ml_col2, ml_col3 = st.columns([1, 1, 1])

    with ml_col1:
        forecast_comm = st.selectbox("Select Crop", commodities_df["name"].tolist())
        comm_row = commodities_df[commodities_df["name"] == forecast_comm].iloc[0]
        comm_id = int(comm_row["id"])
        perish_level = comm_row["perishable_level"]

    with ml_col2:
        forecast_market = st.selectbox("Select Target APMC Market", markets_df["name"].tolist())
        market_id = int(markets_df[markets_df["name"] == forecast_market].iloc[0]["id"])

    with ml_col3:
        model_algorithm = st.selectbox(
            "Regression Algorithm",
            ["RandomForest", "GradientBoosting", "Ridge"],
            format_func=lambda x: "Random Forest Regressor" if x == "RandomForest" else ("Gradient Boosting Regressor" if x == "GradientBoosting" else "Ridge Linear Regressor")
        )
        horizon_days = st.slider("Forecast Horizon (Days)", min_value=3, max_value=14, value=7)

    # Fetch historical data for this market and commodity
    market_history_df = db.get_historical_prices(commodity_id=comm_id, market_ids=[market_id])

    if len(market_history_df) < 40:
        st.warning("Insufficient data available for this market. Need at least 40 records.")
    else:
        with st.spinner("Training regression model and generating multi-day forecast..."):
            forecaster = PriceForecaster(model_type=model_algorithm)
            train_output = forecaster.train(market_history_df, test_size_days=30)
            forecast_df = forecaster.forecast_next_days(market_history_df, horizon_days=horizon_days)

            latest_modal = float(market_history_df["modal_price"].iloc[-1])
            timing_signal = forecaster.generate_selling_timing_signal(
                current_price=latest_modal,
                forecast_df=forecast_df,
                perishable_level=perish_level
            )

        # Farmer Action Signal Banner
        signal_type = timing_signal["signal"]
        if "HOLD" in signal_type:
            st.success(f"### 🟢 SMART SELLING SIGNAL: {signal_type}\n{timing_signal['rationale']}")
        elif "SELL IMMEDIATELY" in signal_type:
            st.error(f"### 🔴 SMART SELLING SIGNAL: {signal_type}\n{timing_signal['rationale']}")
        else:
            st.info(f"### 🟡 SMART SELLING SIGNAL: {signal_type}\n{timing_signal['rationale']}")

        # Model Performance Cards
        st.markdown("### 📐 Model Performance Metrics (Test Set Evaluation)")
        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
        m = train_output["metrics"]
        with metric_col1:
            st.metric("R² Score (Goodness of Fit)", f"{m['R2']:.3f}", "Higher is better")
        with metric_col2:
            st.metric("MAE (Mean Absolute Error)", f"₹ {m['MAE']:.1f}/qtl", "Average prediction error")
        with metric_col3:
            st.metric("RMSE (Root Mean Square)", f"₹ {m['RMSE']:.1f}/qtl")
        with metric_col4:
            st.metric("MAPE (Mean % Error)", f"{m['MAPE_pct']:.2f}%")

        # Forecast Trajectory Chart
        st.markdown("### 🔮 Projected Price Path & Confidence Intervals")
        
        # Prepare plot with recent history + forecast
        recent_hist = market_history_df.iloc[-30:].copy()
        
        fig_fc = go.Figure()

        # Historical curve
        fig_fc.add_trace(go.Scatter(
            x=recent_hist["date"],
            y=recent_hist["modal_price"],
            mode="lines+markers",
            name="Historical Actual Modal Price",
            line=dict(color="#1976d2", width=2.5)
        ))

        # Forecast Line
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Forecasted Price (₹/qtl)"],
            mode="lines+markers",
            name="ML Projected Modal Price",
            line=dict(color="#2e7d32", width=3, dash="dash")
        ))

        # Upper Bound
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Upper CI (₹/qtl)"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            name="Upper 95% Confidence Bound"
        ))

        # Lower Bound with fill
        fig_fc.add_trace(go.Scatter(
            x=forecast_df["Date"],
            y=forecast_df["Lower CI (₹/qtl)"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(46, 125, 50, 0.15)",
            name="95% Confidence Interval"
        ))

        fig_fc.update_layout(
            title=f"{forecast_comm} at {forecast_market}: {horizon_days}-Day ML Price Forecast",
            xaxis_title="Date",
            yaxis_title="Price (₹ / Quintal)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_fc, use_container_width=True)

        # Tabular Forecast & Feature Importance
        tab_fc_table, tab_feat_imp = st.tabs(["📋 Forecast Schedule Table", "🔬 Feature Importance Breakdown"])
        with tab_fc_table:
            st.dataframe(forecast_df, use_container_width=True)

        with tab_feat_imp:
            if train_output["feature_importance"]:
                imp_df = pd.DataFrame([
                    {"Feature": k, "Importance": v}
                    for k, v in train_output["feature_importance"].items()
                ]).sort_values("Importance", ascending=False)
                
                fig_imp = px.bar(
                    imp_df,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Feature Importance in Price Prediction Model",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig_imp.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.info("Feature importance is available for tree-based models (Random Forest, Gradient Boosting).")


# ---------------------------------------------------------
# MODULE 5: DATABASE & PRICE INGESTION
# ---------------------------------------------------------
elif menu_selection == "🗄️ Database & Price Ingestion":
    st.markdown("""
    <div class="main-header">
        <h1>🗄️ Market Database Explorer & Price Ingestion</h1>
        <p>Inspect database records, view decision audit logs, or input new daily APMC trade records.</p>
    </div>
    """, unsafe_allow_html=True)

    db_tab1, db_tab2, db_tab3 = st.tabs(["📝 Add Daily Price Entry", "📜 Farmer Decision History", "💾 Raw Agmarknet Data Explorer"])

    with db_tab1:
        st.subheader("Manual APMC Trade Record Entry")
        with st.form("add_price_form"):
            form_col1, form_col2 = st.columns(2)
            with form_col1:
                in_market = st.selectbox("APMC Market", markets_df["name"].tolist())
                in_comm = st.selectbox("Commodity", commodities_df["name"].tolist())
                in_date = st.date_input("Trade Date", pd.to_datetime("today"))
            with form_col2:
                in_arrivals = st.number_input("Arrivals (Quintals)", min_value=1.0, value=1200.0)
                in_min = st.number_input("Minimum Price (₹/qtl)", min_value=100.0, value=2200.0)
                in_modal = st.number_input("Modal Price (₹/qtl)", min_value=100.0, value=2500.0)
                in_max = st.number_input("Maximum Price (₹/qtl)", min_value=100.0, value=2800.0)

            submitted = st.form_submit_button("Submit Price Entry")
            if submitted:
                m_id = int(markets_df[markets_df["name"] == in_market].iloc[0]["id"])
                c_id = int(commodities_df[commodities_df["name"] == in_comm].iloc[0]["id"])
                db.insert_daily_price(
                    market_id=m_id,
                    commodity_id=c_id,
                    date_str=in_date.strftime("%Y-%m-%d"),
                    arrivals=in_arrivals,
                    min_p=in_min,
                    max_p=in_max,
                    modal_p=in_modal
                )
                st.success(f"Successfully recorded price for {in_comm} at {in_market} on {in_date}!")

    with db_tab2:
        st.subheader("Recent Farmer Queries & Recommender Log")
        decisions_df = db.get_recent_decisions(limit=25)
        if decisions_df.empty:
            st.info("No decisions logged yet. Use the 'Save Query to Decision History' button on the recommender page.")
        else:
            st.dataframe(decisions_df, use_container_width=True)

    with db_tab3:
        st.subheader("APMC Markets Reference")
        st.dataframe(markets_df, use_container_width=True)

        st.subheader("Commodities Reference")
        st.dataframe(commodities_df, use_container_width=True)


# ---------------------------------------------------------
# MODULE 6: PROJECT SYNOPSIS & TEAM CREDITS
# ---------------------------------------------------------
elif menu_selection == "ℹ️ Project Synopsis & Team":
    st.markdown("""
    <div class="main-header">
        <h1>🌾 FarmGate Project Synopsis & Team</h1>
        <p>Intelligent Agricultural Market Price Comparison and Smart Selling Decision Support System</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 🎓 Student Project Team Details
    - **Team Number:** 2
    - **Lab Batch:** C2
    """)

    team_data = [
        {"Sl. No": 1, "Student Name": "Gandikote Subramanya", "USN": "4NI25CS072"},
        {"Sl. No": 2, "Student Name": "Gurukiran Patil", "USN": "4NI25CS077"},
        {"Sl. No": 3, "Student Name": "Girish Patil", "USN": "4NI25CS075"},
        {"Sl. No": 4, "Student Name": "Kariappa C G", "USN": "4NI25CS089"},
        {"Sl. No": 5, "Student Name": "Jeevan L", "USN": "4NI25CS085"},
    ]
    st.table(pd.DataFrame(team_data))

    st.markdown("""
    ### 📖 Project Overview
    FarmGate is a web-based agricultural decision-support platform designed to help farmers make informed decisions about where to sell their agricultural produce. The system provides agricultural market price information, compares crop prices across different markets, considers transportation expenses and estimates potential net earnings.

    It also uses historical market data to analyse price trends and provide machine-learning-based price estimates. By combining market comparison, profit calculation and data analytics, FarmGate aims to help farmers identify better selling opportunities.

    ### ❗ Problem Statement
    Farmers often face difficulties accessing and comparing agricultural commodity prices across different markets. A market offering a higher selling price may not necessarily provide better profits because of transportation expenses.

    Additionally, limited access to historical price analysis makes it difficult for farmers to understand market trends before selling their produce. Therefore, a platform is needed to consolidate market price information, calculate estimated net earnings after transportation costs and provide data-driven selling recommendations.

    ### 🎯 Objectives
    1. To provide agricultural commodity price information from different markets.
    2. To compare crop prices across multiple agricultural markets.
    3. To calculate estimated transportation expenses and net earnings.
    4. To recommend potentially more profitable markets based on crop prices and transportation costs.
    5. To analyse and visualise historical crop price trends.
    6. To implement a machine-learning model for estimating possible future price trends.
    7. To develop a simple and interactive platform for farmers.

    ### 🛠️ Technologies and Tools Used
    - **Programming Language:** Python
    - **Frontend:** Streamlit
    - **Data Processing:** Pandas, NumPy
    - **Machine Learning:** Scikit-learn (Random Forest, Gradient Boosting, Ridge Regression)
    - **Database:** SQLite
    - **Data Visualisation:** Plotly
    - **Version Control:** Git & GitHub
    - **CI/CD:** GitHub Actions
    - **Containerisation:** Docker
    - **Testing:** Pytest
    """)
