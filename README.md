# Experience-Salary: EDA, Model Training, and Production API

This project takes the `Experience-Salary.csv` dataset and provides:

- Exploratory data analysis (EDA) outputs
- A trained regression model for salary prediction
- A production-style FastAPI service for predictions

## 1. Dataset understanding

- Feature: `exp(in months)`
- Target: `salary(in thousands)`
- Unit note: model predicts salary in **thousands**

The scripts internally rename these to:

- `experience_months`
- `salary_thousands`

## 2. Run EDA

```bash
python scripts/eda.py
```

Outputs in `reports/`:

- `eda_summary.json`
- `experience_distribution.png`
- `salary_distribution.png`
- `experience_salary_scatter.png`
- `correlation_heatmap.png`

## 3. Train model

```bash
python scripts/train.py
```

What this does:

- Automatically cleans invalid target rows (`salary_thousands < 0`)
- Train/test split (80/20)
- Trains two candidates:
  - Linear Regression
  - Random Forest Regressor
- Selects best by lowest RMSE on test set
- Creates a new model version on every run

Outputs:

- `artifacts/salary_model.joblib`
- `artifacts/model_metadata.json`
- `artifacts/latest_model.json`
- `artifacts/model_registry.json`
- `artifacts/model_versions/v001/` (and future versions like `v002`, `v003`, ...)
- `reports/model_report.json`

## 4. Run prediction API (production-style)

```bash
uvicorn app.main:app --reload
```

Open docs:

- `http://127.0.0.1:8000/docs`

Health check:

- `GET /health`

Prediction endpoint:

- `POST /predict`
- Example body:

```json
{
  "experience_months": 24
}
```

Example response:

```json
{
  "predicted_salary_thousands": 25.4,
  "predicted_salary": 25400,
  "unit": "predicted_salary is in full currency units, predicted_salary_thousands is in thousands"
}
```

## 5. Run Streamlit app (interactive UI)

```bash
streamlit run streamlit_app.py
```

What you get:

- Input experience in months
- Instant salary prediction
- Model version, leaderboard, and cleaning report in the UI

## 6. Impact tips to make most out of this dataset

- Convert months to years in dashboards for business readability.
- Add prediction intervals to show uncertainty, not only point estimates.
- Retrain periodically and monitor RMSE drift.
- Collect more features (education, location, role, company size) to improve accuracy.
