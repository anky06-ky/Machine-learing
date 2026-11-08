"""Tạo 210 khách mô phỏng THAY THẾ; không phải CSV gốc của PDF."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def create_data():
    rng = np.random.default_rng(8)
    groups = [(35, 25, 7.5, 80), (35, 43, 30, 20), (60, 40, 17.5, 50),
              (40, 45, 8, 19), (40, 33, 30, 81)]
    rows = []
    for n, age, income, spending in groups:
        rows.append(pd.DataFrame({"tuoi": np.clip(np.rint(rng.normal(age, 6, n)), 18, 65).astype(int),
            "thu_nhap": np.round(np.clip(rng.normal(income, 2.2, n), 3, 40), 1),
            "diem_chi_tieu": np.clip(np.rint(rng.normal(spending, 7, n)), 1, 100).astype(int)}))
    df = pd.concat(rows, ignore_index=True)
    return df.iloc[rng.permutation(len(df))].reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghi-de", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / "data" / "khach_hang.csv"
    path.parent.mkdir(exist_ok=True)
    if path.exists() and not args.ghi_de:
        raise SystemExit("CSV da ton tai; dung --ghi-de de tao lai.")
    create_data().to_csv(path, index=False)
    print("DA TAO DU LIEU MO PHONG THAY THE, seed=8:", path)
