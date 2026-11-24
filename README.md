# Proposal 2 — Pharmacy Sales & Demand Intelligence

## Run locally
1. Install Python 3.10 or newer.
2. Extract this folder.
3. Open Command Prompt inside this folder.
4. Run `python -m pip install -r requirements.txt`.
5. Run `python -m streamlit run app.py`.

Or double-click `RUN_APP.bat` on Windows.

## Included files
- `app.py` — sales analysis and product-level forecasting dashboard.
- `data/02_pharmacy_medicine_sales_synthetic.csv` — sample transaction-line dataset.
- `requirements.txt` — Python dependencies.
- `.streamlit/config.toml` — theme settings.

## Main features
- Sales, units, products and revenue KPIs.
- Monthly revenue, category mix, and top products.
- Product movement and recorded stock view.
- Select a product and forecast 1–6 months ahead.
- Holdout MAE and RMSE and downloadable filtered records.

## Important methodology note
The forecast is a baseline time-index model: Random Forest when at least 12 monthly observations are available, otherwise Linear Regression. It uses a chronological holdout segment for error evaluation and then refits on the full selected product history for the future estimate. It is not a substitute for a carefully validated seasonal forecasting study.

The included dataset is synthetic. Do not report it as actual pharmacy transactions or actual demand. Get written permission and use anonymized transaction and inventory records for actual research.
