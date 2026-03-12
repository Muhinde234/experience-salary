import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path("Experience-Salary.csv")
ARTIFACT_DIR = Path("artifacts")
REPORT_DIR = Path("reports")



def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }



def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH).rename(
        columns={
            "exp(in months)": "experience_months",
            "salary(in thousands)": "salary_thousands",
        }
    )

    X = df[["experience_months"]]
    y = df["salary_thousands"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    candidates = {
        "linear_regression": Pipeline(
            steps=[("scaler", StandardScaler()), ("model", LinearRegression())]
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            min_samples_leaf=2,
        ),
    }

    leaderboard = {}
    trained_models = {}

    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        leaderboard[name] = metrics(y_test.to_numpy(), preds)
        trained_models[name] = model

    best_model_name = min(leaderboard, key=lambda m: leaderboard[m]["rmse"])
    best_model = trained_models[best_model_name]

    joblib.dump(best_model, ARTIFACT_DIR / "salary_model.joblib")

    metadata = {
        "feature_columns": ["experience_months"],
        "target_column": "salary_thousands",
        "best_model": best_model_name,
        "leaderboard": leaderboard,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "note": "Salary is predicted in thousands.",
    }

    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(REPORT_DIR / "model_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete. Best model: {best_model_name}")
    print(json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    main()
