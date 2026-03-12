from pathlib import Path

import pandas as pd

DATA_PATH = Path("Experience-Salary.csv")



def load_raw_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_csv(data_path).rename(
        columns={
            "exp(in months)": "experience_months",
            "salary(in thousands)": "salary_thousands",
        }
    )



def clean_salary_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_rows = int(df.shape[0])
    negative_salary_mask = df["salary_thousands"] < 0
    negative_salary_count = int(negative_salary_mask.sum())

    cleaned = df.loc[~negative_salary_mask].copy()
    cleaned_rows = int(cleaned.shape[0])

    cleaning_report = {
        "original_rows": original_rows,
        "cleaned_rows": cleaned_rows,
        "rows_removed": original_rows - cleaned_rows,
        "negative_salary_count": negative_salary_count,
        "rule": "Removed rows where salary_thousands < 0",
    }

    return cleaned, cleaning_report
