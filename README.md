# FarmGate 🌾
### Intelligent Agricultural Market Price Comparison and Smart Selling Decision Support System

[![CI/CD Pipeline](https://github.com/JeevanL2007/FarmGate/actions/workflows/ci.yml/badge.svg)](https://github.com/JeevanL2007/FarmGate/actions)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Frontend: Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Backend: FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Database: SQLite](https://img.shields.io/badge/Database-SQLite-003B57.svg)](https://www.sqlite.org/)
[![Container: Docker](https://img.shields.io/badge/Container-Docker-2496ED.svg)](https://www.docker.com/)

---

## 🌐 Live Web Application
🔗 **Deployed URL:** [https://farmgate-karnataka.streamlit.app/](https://farmgate-karnataka.streamlit.app/)

---

## 👥 Student Project Details
- **Team Number:** 2
- **Lab Batch:** C2
- **Institution:** Department of Computer Science & Engineering

| Sl. No | Student Name | USN | Role / Contributions |
| :---: | :--- | :--- | :--- |
| 1 | **Gandikote Subramanya** | 4NI25CS072 | Decision Engine & Mathematical Formulation |
| 2 | **Gurukiran Patil** | 4NI25CS077 | Machine Learning Models & Time-Series Forecasting |
| 3 | **Girish Patil** | 4NI25CS075 | Database Schema, Data Processing & Analysis |
| 4 | **Kariappa C G** | 4NI25CS089 | Streamlit Frontend & Data Visualization |
| 5 | **Jeevan L** | 4NI25CS085 | REST API Backend, Docker Containerization & CI/CD |

---

## 📌 1. Project Overview & Problem Statement

### The Agricultural Dilemma
In Karnataka, smallholder farmers often face difficulty deciding where and when to sell harvested produce. While a metropolitan APMC (such as Yeshwanthpur or Kalasipalya in Bengaluru) might quote a higher headline modal price than a local mandi (such as Kolar or Chikkaballapur), reaching distant markets incurs significant variable transportation expenses:
- Commercial vehicle freight rates per kilometer.
- Multi-trip logistics when harvest volume exceeds vehicle capacity.
- Loading and unloading labor (hamali) charges.
- Statutory APMC mandi cess (typically 1.5% in Karnataka).

Without accurate net-earning calculations, farmers frequently transport produce farther only to realize a **lower net take-home profit** than if they had sold locally.

### The FarmGate Solution
**FarmGate** is an intelligent agricultural decision-support platform designed for Karnataka's agrarian ecosystem. It provides:
1 **Smart Selling Recommender:** Deducts freight, labor, and cess from gross revenues to rank candidate APMCs by actual net earnings.
2 **Break-Even Price Sensitivity:** Computes the minimum price the local market must pay to match a distant market's profit.
3 **Cross-Market Price Comparison:** Compares minimum, maximum, and modal price spreads across 10 major Karnataka APMC markets.
4 **Historical Trend Analysis:** Multi-market time series charts, price volatility spreads, and arrival volume elasticity.
5 **Machine Learning Price Forecasting:** Regression models predicting 7- to 14-day price trajectories with illustrative uncertainty bands and demo timing signals (**HOLD** / **SELL** / **STABLE**).
6 **Dual-Interface Architecture:** User-friendly **Streamlit web dashboard** for farmers and an asynchronous **FastAPI REST API** with interactive Swagger documentation.

---

## 🏗️ 2. System Architecture

```text
+-----------------------------------------------------------------------------------+
|                                  USER CLIENTS                                     |
|     Web Browser / Farmers                 External Apps / Mobile Clients          |
+-----------------------------------------------------------------------------------+
             |                                             |
             v                                             v
+-------------------------------+             +-------------------------------------+
|   Streamlit Web Application   |             |       FastAPI REST API Backend      |
|           (app.py)            |             |               (api.py)              |
|  - Smart Selling Recommender  |             |  - POST /api/recommend              |
|  - Cross-Market Price Compare |             |  - GET  /api/prices/latest          |
|  - Historical Trend Analytics |             |  - GET  /api/forecast               |
|  - ML Price Forecasting       |             |  - GET  /api/markets, /commodities  |
|  - Interactive Plotly Charts  |             |  - Interactive Docs: /docs          |
+-------------------------------+             +-------------------------------------+
             |                                             |
             +----------------------+----------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                            CORE APPLICATION DOMAIN                                |
|  +------------------------+  +------------------------+  +---------------------+  |
|  | Smart Selling Engine   |  | ML Forecasting Engine  |  | Routing & Freight   |  |
|  | (decision_engine.py)   |  | (ml_model.py)          |  | (utils.py)          |  |
|  | - Net Earnings Calc.  |  | - Random Forest        |  | - Haversine Formula |  |
|  | - Market Ranker        |  | - Gradient Boosting    |  | - Road Winding 1.25x|  |
|  | - Break-Even Analysis  |  | - Ridge Regression     |  | - Vehicle Profiles  |  |
|  +------------------------+  +------------------------+  +---------------------+  |
+-----------------------------------------------------------------------------------+
                                    |
                                    v
+-----------------------------------------------------------------------------------+
|                              DATA & PERSISTENCE LAYER                             |
|  +------------------------------------+  +-------------------------------------+  |
|  |        SQLite Database Layer       |  |       Government Data Connector     |  |
|  |             (database.py)          |  |             (live_sync.py)          |  |
|  | - markets, commodities             |  | - data.gov.in Agmarknet API format  |  |
|  | - daily_prices (12,740+ records)   |  | - Resource: 9ef84268-d588-465a      |  |
|  | - farmer_decisions (Audit Trail)   |  | - Official field mappings           |  |
|  +------------------------------------+  +-------------------------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 🧮 3. Mathematical Formulation (Smart Selling Engine)

For a farmer located at origin $O$ with quantity $Q$ quintals of commodity $C$, and candidate APMC market $M_i$:

1. **Road Distance Calculation (Haversine with Winding Adjustment):**
   $$D_{\text{straight}} = 2 R \arcsin \left( \sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)} \right)$$
   $$D_i = D_{\text{straight}} \times 1.25 \quad (\text{adjusted for Karnataka highway/rural road winding})$$

2. **Trip & Vehicle Logistics:**
   $$N_{\text{trips}} = \max\left(1, \left\lceil \frac{Q}{\text{Capacity}_{\text{vehicle}}} \right\rceil\right)$$

3. **Total Transport & Handling Deductions:**
   $$\text{Cost}_{\text{transport}} = D_i \times \text{Rate}_{\text{per\_km}} \times N_{\text{trips}}$$
   $$\text{Cost}_{\text{labor}} = Q \times (\text{Rate}_{\text{loading}} + \text{HandlingCharge}_{M_i})$$
   $$\text{Cost}_{\text{cess}} = (Q \times P_{\text{modal}, i}) \times \left(\frac{\text{CessRate}_{M_i}}{100}\right)$$
   $$\text{Total Deductions} = \text{Cost}_{\text{transport}} + \text{Cost}_{\text{labor}} + \text{Cost}_{\text{cess}}$$

4. **Net Farmer Take-Home Earnings (labelled "Estimated net selling proceeds" in the UI):**
   $$\text{Gross Revenue} = Q \times P_{\text{modal}, i}$$
   $$\text{Net Profit} = \text{Gross Revenue} - \text{Total Deductions}$$
   $$\text{Realized Rate per Quintal} = \frac{\text{Net Profit}}{Q}$$

5. **Local Break-Even Price Threshold:**
   $$\text{BreakEvenPrice}_{\text{nearest}} = \frac{\text{NetProfit}_{\text{best}} + \text{Cost}_{\text{transport, nearest}} + \text{Cost}_{\text{labor, nearest}}}{Q \times (1 - \text{CessRate}/100)}$$

---

## 📊 4. Data Sources & Real vs. Simulated Disclosure

Honest academic transparency regarding data handling in FarmGate:

| Data Layer | Source & Methodology | Current Status |
| :--- | :--- | :--- |
| **Baseline Benchmark Dataset** | Mathematically calibrated time series based on historical Karnataka APMC patterns (seasonal sinusoidal macro trends, arrival-to-price elasticity, day-of-week surges, and inter-mandi price multipliers). | **Active in Demo Mode.** Used for offline presentations and evaluation demos to ensure 100% uptime and resilience against government network timeouts. |
| **Daily Rolling Auto-Updater** | Mean-reverting random walk expanding the calendar series up to the current date so that evaluation dates match current time. | **Active.** Clearly marked in UI as `🟡 DEMO / SIMULATED`. |
| **AGMARKNET Government Connector** | Direct integration module (`src/live_sync.py`) configured for Open Government Data India (`api.data.gov.in`, Resource ID: `9ef84268-d588-465a-a308-a864a43d0070`). | **Configured.** Requires a free personal API key from `data.gov.in`. In absence of key, gracefully defaults to labelled demo data. |
| **Manual APMC Field Entry** | Daily entry form in UI under *Database & Price Ingestion* tab. | **Disabled in deployed demo** to protect benchmark data integrity. The form and benchmark explorer remain visible for reference. |

> [!NOTE]
> All simulated prices are explicitly marked with **`🟡 DEMO / SIMULATED`** badges across the user interface, tables, and charts to prevent misrepresenting synthetic data as live government feeds.

---

## 🤖 5. Machine Learning Methodology

### Models Implemented
- **Random Forest Regressor:** 100 estimators, max depth 8 (non-linear seasonal and arrival interactions).
- **Gradient Boosting Regressor:** 100 estimators, learning rate 0.08, max depth 5 (sequential residual learning).
- **Ridge Linear Regression:** $\ell_2$-regularized linear model providing a robust benchmark baseline.

### Feature Engineering Pipeline
- **Price Lag Features:** $t-1$ (yesterday's modal price), $t-2$, $t-3$, $t-7$ (same day previous week), $t-14$.
- **Rolling Window Statistics:** 7-day moving average, 7-day standard deviation (volatility proxy), 14-day moving average.
- **Supply / Arrival Dynamics:** 1-day lag arrival volume, 7-day rolling arrival average (capturing supply elasticity).
- **Calendar & Cyclical Signals:** Day of week (0=Mon to 6=Sun), month, day of month, plus cyclical trigonometric encodings:
  $$\text{Month}_{\sin} = \sin\left(\frac{2\pi \times \text{Month}}{12}\right), \quad \text{Month}_{\cos} = \cos\left(\frac{2\pi \times \text{Month}}{12}\right)$$
- **Price Momentum:** 7-day percentage rate of change $\frac{P_{t-1} - P_{t-7}}{P_{t-7}}$.

### Evaluation & Validation
- **Train/Test Split:** Temporal out-of-time evaluation (last 30 calendar days held out for validation; no future data leakage).
- **Evaluation Metrics:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Coefficient of Determination ($R^2$), and Mean Absolute Percentage Error (MAPE).
- **Illustrative Uncertainty Bounds:** Multi-step recursive forecasting with widening illustrative uncertainty bands (not formal statistical confidence intervals):
  $$\text{Illustrative Bound} = \hat{y}_t \pm 1.96 \cdot \sigma_{\text{residual}} \sqrt{1 + 0.1 \cdot t}$$
- **Demo Decision Signals** (labels prefixed with DEMO SIGNAL in the app):
  - 🟢 **HOLD FOR HIGHER GAINS:** Projected price increase $> 6\%$.
  - 🔴 **SELL IMMEDIATELY:** Projected price decline $> 4\%$.
  - 🟡 **STABLE / SELL AT WILL:** Signed change between latest price and near-term forecast average within $\pm 4\%$.

---

## 💻 6. Technologies Used

- **Programming Language:** Python 3.11 / 3.12 / 3.13
- **Frontend Dashboard:** Streamlit, Altair
- **REST API Backend:** FastAPI, Pydantic, Uvicorn
- **Data Science & ML:** Pandas, NumPy, Scikit-learn, SciPy
- **Data Visualization:** Plotly Express, Plotly Graph Objects
- **Database:** SQLite3
- **Testing:** Pytest
- **Containerization:** Docker, Docker Compose
- **CI/CD:** GitHub Actions (Multi-python matrix test + Docker build)

---

## 🚀 7. Installation and Execution

### Prerequisites
- Python 3.11+ installed.
- Git installed.

### Step 1: Clone Repository
```bash
git clone https://github.com/JeevanL2007/FarmGate.git
cd FarmGate
```

### Step 2: Set Up Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at **`http://localhost:8501`**.

### Step 5: (Optional) Run the FastAPI REST API Backend
In a separate terminal:
```bash
python api.py
```
Open **`http://localhost:8000/docs`** for interactive Swagger documentation to test endpoints (`/api/recommend`, `/api/forecast`, `/api/prices/latest`).

---

## 🐳 8. Docker Utilization

The application is containerized with a production-grade multi-stage Docker setup.

### Run with Docker Compose:
```bash
docker-compose up --build
```
Access the application at **`http://localhost:8501`**.

### Run with Standalone Docker:
```bash
docker build -t farmgate:latest .
docker run -p 8501:8501 farmgate:latest
```

---

## ⚙️ 9. GitHub Actions CI/CD Pipeline

The repository includes an automated CI/CD pipeline defined in [`.github/workflows/ci.yml`](.github/workflows/ci.yml) triggered on every `push` and `pull_request` to `main`:

1. **Job 1: Automated Unit Testing (`test`)**
   - Tests across Python 3.11 and 3.12.
   - Installs dependencies from `requirements.txt`.
   - Seeds baseline database schema and records.
   - Executes full Pytest suite.
2. **Job 2: Container Image Build (`docker-build`)**
   - Depends on `test` job passing.
   - Validates that `Dockerfile` builds cleanly without warnings or missing files.

---

## 🧪 10. Automated Testing & Verified Results

Run the automated test suite locally:
```bash
python -m pytest -v
```

### Verified Test Suite (9 Passed):
```text
tests/test_database.py::test_db_initialization PASSED                    [ 11%]
tests/test_database.py::test_insert_and_get_markets PASSED               [ 22%]
tests/test_database.py::test_log_farmer_decision PASSED                  [ 33%]
tests/test_decision_engine.py::test_haversine_distance PASSED            [ 44%]
tests/test_decision_engine.py::test_smart_decision_ranking PASSED        [ 55%]
tests/test_decision_engine.py::test_decision_summary_generation PASSED   [ 66%]
tests/test_ml_model.py::test_price_forecaster_training PASSED            [ 77%]
tests/test_ml_model.py::test_multi_day_forecasting PASSED                [ 88%]
tests/test_ml_model.py::test_selling_signal PASSED                       [100%]
======================== 9 passed in 14.68s ========================
```

---

## 📁 11. Project Directory Structure

```text
FarmGate/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated CI/CD pipeline (Test + Docker build)
├── data/
│   ├── agmarknet_historical_data.csv  # Calibrated baseline price dataset export
│   └── farmgate.db              # SQLite Database (markets, commodities, daily_prices)
├── src/
│   ├── __init__.py
│   ├── database.py              # SQLite connection, schema, and query helpers
│   ├── data_loader.py           # Baseline seeder & rolling date auto-updater
│   ├── decision_engine.py       # Smart Selling Decision & Freight Optimization Engine
│   ├── live_sync.py             # AGMARKNET data.gov.in API connector
│   ├── ml_model.py              # ML Price Forecasting & Timing Signal Generator
│   └── utils.py                 # Haversine distance, Vehicle profiles, Formatters
├── tests/
│   ├── __init__.py
│   ├── test_database.py         # Database CRUD & audit log unit tests
│   ├── test_decision_engine.py  # Freight, cess, and ranking math tests
│   └── test_ml_model.py         # ML training, forecasting, and signal tests
├── app.py                       # Main Streamlit Web Application
├── api.py                       # FastAPI REST API Backend with Swagger UI
├── Dockerfile                   # Docker container definition
├── docker-compose.yml           # Docker Compose orchestration
├── pytest.ini                   # Pytest testpath and pythonpath configuration
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

---

## ⚠️ 12. Project Limitations & Future Scope

### Current Limitations
1. **Demo Data Only in Deployed App:** The deployed Streamlit application runs on simulated benchmark data. The `src/live_sync.py` module is an experimental reference connector for `data.gov.in` and is not used to serve live prices in the current deployment.
2. **Fixed Mandi Cess Assumption:** APMC cess is modeled at a uniform 1.5% statutory rate; certain specialized commodities may have localized fee exemptions.
3. **Mandi Holiday Reporting:** APMCs do not conduct auctions on Sundays and state holidays, resulting in physical arrival gaps.
4. **Single-Commodity Freight:** Current freight logic models a single crop shipment; multi-commodity mixed truckloads are not yet simulated.

### Future Scope
1. **SMS & WhatsApp Dispatch:** Direct integration with Twilio / Gupshup for dispatching daily selling recommendations to farmers via vernacular SMS.
2. **Cold Storage Economic Modeling:** Factoring in the per-quintal cold storage electricity and warehouse rent when evaluating 14-day hold decisions.
3. **Farmer Shared Logistics Pooling:** An aggregation feature allowing nearby farmers in the same taluk to share a Bolero or Canter truck to reduce transport costs.

---

## 👥 Authors
- **Team 2 (Batch C2)**
- Department of Computer Science & Engineering
- Project: FarmGate – Intelligent Agricultural Market Price Comparison and Smart Selling Decision Support System
