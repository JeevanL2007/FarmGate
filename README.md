# FarmGate 🌾
### Intelligent Agricultural Market Price Comparison and Smart Selling Decision Support System

[![CI Pipeline](https://github.com/team2/farmgate/actions/workflows/ci.yml/badge.svg)](https://github.com/team2/farmgate)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Framework: Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 👥 Student Project Details
- **Team Number:** 2
- **Lab Batch:** C2

| Sl. No | Student Name | USN |
| :---: | :--- | :--- |
| 1 | **Gandikote Subramanya** | 4NI25CS072 |
| 2 | **Gurukiran Patil** | 4NI25CS077 |
| 3 | **Girish Patil** | 4NI25CS075 |
| 4 | **Kariappa C G** | 4NI25CS089 |
| 5 | **Jeevan L** | 4NI25CS085 |

---

## 📌 1. Project Overview & Motivation
Farmers often face difficulty deciding where and when to sell their harvested produce. While one agricultural market (APMC) might quote a higher modal price than the local town mandi, the increased distance often incurs significant fuel, vehicle freight, loading/unloading hamali charges, and mandi cess. Without precise calculations, farmers may travel farther only to realize lower take-home profits.

**FarmGate** addresses this critical challenge by providing an intelligent, data-driven decision-support platform tailored for Karnataka's agricultural ecosystem (focusing on major APMCs such as Kolar, Yeshwanthpur, Kalasipalya, Mysuru, Hubballi, Belagavi, Chikkaballapur, Hassan, Shivamogga, and Davanagere).

### Key Pillars:
1. **Multi-Market Price Comparison:** Aggregates and compares minimum, maximum, and modal prices across Karnataka APMCs.
2. **Smart Freight & Net Earning Engine:** Calculates true net profit by deducting road freight, labor/hamali charges, and APMC cess from gross revenues, ranking candidate markets from highest to lowest net returns.
3. **Break-Even & Sensitivity Analysis:** Tells the farmer the exact price threshold their local market would need to match the profitability of a distant market.
4. **Machine Learning Price Forecasting:** Employs regression algorithms (Random Forest, Gradient Boosting, Ridge) trained on historical time-series data to predict 7-14 day price trajectories and output actionable timing signals (**SELL NOW** vs **HOLD FOR 3-5 DAYS**).
5. **Interactive Visualizations:** Interactive Plotly charts showing cost-revenue waterfalls, price-arrival elasticities, and confidence interval bands.

---

## 🏗️ 2. System Architecture

```text
+-------------------------------------------------------------+
|                     FarmGate Streamlit UI                   |
|   (Interactive Dashboard, Bilingual Hints, Metric Cards)    |
+-------------------------------------------------------------+
       |                           |                       |
       v                           v                       v
+------------------+     +-------------------+    +--------------------+
|  Smart Selling   |     | Cross-Market      |    |  ML Price Trend    |
|  Decision Engine |     | Price Comparison  |    |  Forecasting Engine|
+------------------+     +-------------------+    +--------------------+
       |                           |                       |
       +---------------------------+-----------------------+
                                   |
                                   v
                      +--------------------------+
                      |   SQLite Database Layer  |
                      |  - markets               |
                      |  - commodities           |
                      |  - daily_prices (12k+)   |
                      |  - farmer_decisions      |
                      +--------------------------+
                                   |
                      +--------------------------+
                      | Data Ingestion / Seeder  |
                      | (Agmarknet Standards)    |
                      +--------------------------+
```

---

## 🧮 3. Smart Selling Mathematical Formulation

For a farmer at location $O$ with quantity $Q$ (quintals) of commodity $C$:

$$\text{Gross Revenue } (GR) = Q \times P_{\text{modal}}$$

$$\text{Distance } (D) = \text{Haversine}(O, \text{APMC}) \times 1.25$$

$$\text{Trips } (N) = \left\lceil \frac{Q}{\text{Capacity}_{\text{veh}}} \right\rceil$$

$$\text{Transport Expense } (TE) = D \times \text{Rate}_{\text{km}} \times N$$

$$\text{Labor / Hamali } (LE) = Q \times (\text{LoadingRate} + \text{MandiHandling})$$

$$\text{APMC Cess } (CE) = GR \times \left( \frac{\text{CessRate}}{100} \right)$$

$$\text{Net Profit } (NP) = GR - (TE + LE + CE)$$

$$\text{Realized Rate per Quintal} = \frac{NP}{Q}$$

$$\text{Realized Rate per Kg} = \frac{\text{Realized Rate per Quintal}}{100}$$

---

## 💻 4. Technologies Used
- **Programming Language:** Python 3.11+
- **Frontend / Dashboard:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn (RandomForestRegressor, GradientBoostingRegressor, Ridge)
- **Database:** SQLite
- **Data Visualisation:** Plotly Express & Plotly Graph Objects
- **Containerisation:** Docker & Docker Compose
- **Testing:** Pytest (100% passing test suite across database, decision engine, ML forecaster)
- **CI/CD:** GitHub Actions workflow

---

## 🚀 5. Installation and Setup

### Prerequisites
- Python 3.10+ (tested on Python 3.11, 3.12, and 3.13)
- Git

### Step 1: Clone Repository
```bash
git clone https://github.com/team2/farmgate.git
cd farmgate
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Seed Database with Historical Agmarknet Data
```bash
python -m src.data_loader
```
*(Populates SQLite database with 10+ Karnataka APMCs, 7 commodities, and 12,000+ daily price/arrival records).*

### Step 5: Run Pytest Test Suite
```bash
python -m pytest
```

### Step 6: Launch Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🐳 6. Running with Docker

### Using Docker Compose:
```bash
docker-compose up --build
```
Access the application at `http://localhost:8501`.

### Using Standalone Docker Build:
```bash
docker build -t farmgate-app .
docker run -p 8501:8501 farmgate-app
```

---

## 📁 7. Project Directory Structure

```text
farmgate/
├── .github/
│   └── workflows/
│       └── ci.yml               # GitHub Actions Automated CI Pipeline
├── data/
│   ├── agmarknet_historical_data.csv  # Agmarknet historical dataset export
│   └── farmgate.db              # SQLite Database
├── src/
│   ├── __init__.py
│   ├── database.py              # SQLite Schema, Queries & Audit Logging
│   ├── data_loader.py           # Agmarknet Data Generator & DB Seeder
│   ├── decision_engine.py       # Smart Selling Decision & Profit Optimization Engine
│   ├── ml_model.py              # Scikit-learn Price Forecasting & Timing Signals
│   └── utils.py                 # Haversine distance, Vehicle profiles, Formatters
├── tests/
│   ├── __init__.py
│   ├── test_database.py         # DB unit tests
│   ├── test_decision_engine.py  # Decision engine mathematical validation
│   └── test_ml_model.py         # ML training, forecasting, and signal tests
├── app.py                       # Main Streamlit Web Application
├── Dockerfile                   # Docker container configuration
├── docker-compose.yml           # Multi-container orchestration
├── pytest.ini                   # Pytest configuration
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## 👥 Authors
- **Team 2 (Batch C2)**
- Project: FarmGate – Intelligent Agricultural Market Price Comparison and Smart Selling Decision Support System
