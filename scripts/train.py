import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_utils import clean_salary_data, load_raw_data

DATA_PATH = Path("Experience-Salary.csv")
ARTIFACT_DIR = Path("artifacts")
MODEL_VERSIONS_DIR = ARTIFACT_DIR / "model_versions"
REPORT_DIR = Path("reports")
REGISTRY_PATH = ARTIFACT_DIR / "model_registry.json"
LATEST_PATH = ARTIFACT_DIR / "latest_model.json"



def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {"versions": []}
    return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))


def next_version(registry: dict) -> int:
    return len(registry.get("versions", [])) + 1



def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df_raw = load_raw_data(DATA_PATH)
    df, cleaning_report = clean_salary_data(df_raw)

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

    registry = load_registry()
    version_number = next_version(registry)
    version_id = f"v{version_number:03d}"
    trained_at = datetime.now(UTC).isoformat()
    version_dir = MODEL_VERSIONS_DIR / version_id
    version_dir.mkdir(parents=True, exist_ok=True)
    model_path = version_dir / "salary_model.joblib"
    metadata_path = version_dir / "model_metadata.json"

    joblib.dump(best_model, model_path)

    metadata = {
        "model_version": version_id,
        "trained_at_utc": trained_at,
        "feature_columns": ["experience_months"],
        "target_column": "salary_thousands",
        "best_model": best_model_name,
        "leaderboard": leaderboard,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "cleaning_report": cleaning_report,
        "note": "Salary is predicted in thousands.",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

 
    joblib.dump(best_model, ARTIFACT_DIR / "salary_model.joblib")
    with open(ARTIFACT_DIR / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    registry.setdefault("versions", []).append(
        {
            "version": version_id,
            "trained_at_utc": trained_at,
            "model_path": str(model_path).replace("\\", "/"),
            "metadata_path": str(metadata_path).replace("\\", "/"),
            "best_model": best_model_name,
            "rmse": leaderboard[best_model_name]["rmse"],
            "rows_removed_during_cleaning": cleaning_report["rows_removed"],
        }
    )

    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    latest = {
        "version": version_id,
        "model_path": str(model_path).replace("\\", "/"),
        "metadata_path": str(metadata_path).replace("\\", "/"),
    }
    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(latest, f, indent=2)

    with open(REPORT_DIR / "model_report.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Training complete. Best model: {best_model_name} ({version_id})")
    print(json.dumps(leaderboard, indent=2))


if __name__ == "__main__":
    main()
