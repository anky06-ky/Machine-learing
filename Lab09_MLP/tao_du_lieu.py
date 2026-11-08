"""Tạo 400 điểm bốn góc THAY THẾ; không phải xor.csv gốc của PDF."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def create_data():
    rng = np.random.default_rng(9)
    X = rng.uniform(-2, 2, (400, 2))
    y = (X[:, 0]*X[:, 1] > 0).astype(int)
    flip = rng.choice(len(X), 24, replace=False)
    y[flip] = 1-y[flip]
    return pd.DataFrame({"x1": np.round(X[:, 0], 4), "x2": np.round(X[:, 1], 4), "nhan": y})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghi-de", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / "data" / "xor.csv"
    path.parent.mkdir(exist_ok=True)
    if path.exists() and not args.ghi_de:
        raise SystemExit("CSV da ton tai; dung --ghi-de de tao lai.")
    create_data().to_csv(path, index=False)
    print("DA TAO DU LIEU MO PHONG THAY THE, seed=9:", path)
