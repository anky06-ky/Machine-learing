"""Đường dẫn và các thao tác dùng chung; chạy được từ bất kỳ thư mục nào."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)


def read_data(name):
    return pd.read_csv(ROOT / "data" / name)


def split_data(name="sinh_vien_bo_hoc.csv", target="bo_hoc", columns=None):
    df = read_data(name)
    X = df[columns] if columns else df.drop(columns=target)
    return train_test_split(X, df[target], test_size=0.25,
                            random_state=17, stratify=df[target])


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIGURES / name, dpi=150, bbox_inches="tight")
    plt.close()
    print("Da luu:", "figures/" + name)
