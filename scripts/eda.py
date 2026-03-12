import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DATA_PATH = Path("Experience-Salary.csv")
REPORT_DIR = Path("reports")



def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    df = df.rename(
        columns={
            "exp(in months)": "experience_months",
            "salary(in thousands)": "salary_thousands",
        }
    )

    summary = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values": df.isna().sum().to_dict(),
        "describe": df.describe().to_dict(),
        "correlation": float(df["experience_months"].corr(df["salary_thousands"])),
        "negative_salary_count": int((df["salary_thousands"] < 0).sum()),
    }

    with open(REPORT_DIR / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(10, 5))
    sns.histplot(df["experience_months"], bins=30, kde=True, color="#1f77b4")
    plt.title("Distribution of Experience (Months)")
    plt.xlabel("Experience (months)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "experience_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    sns.histplot(df["salary_thousands"], bins=30, kde=True, color="#2ca02c")
    plt.title("Distribution of Salary (Thousands)")
    plt.xlabel("Salary (thousands)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "salary_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="experience_months", y="salary_thousands", alpha=0.7)
    sns.regplot(
        data=df,
        x="experience_months",
        y="salary_thousands",
        scatter=False,
        color="red",
    )
    plt.title("Experience vs Salary")
    plt.xlabel("Experience (months)")
    plt.ylabel("Salary (thousands)")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "experience_salary_scatter.png", dpi=160)
    plt.close()

    corr = df[["experience_months", "salary_thousands"]].corr()
    plt.figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="Blues", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "correlation_heatmap.png", dpi=160)
    plt.close()

    print("EDA complete. Files saved in reports/")


if __name__ == "__main__":
    main()
