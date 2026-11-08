# -*- coding: utf-8 -*-
"""Bài 3: logistic, tanh, relu trên cùng mạng (100,100), alpha=1."""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "hai_vang_trang.csv")
X = df.drop(columns="nhan")
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
results = {}
print("Activation    tap hoc   tap kiem tra")
for activation in ["logistic", "tanh", "relu"]:
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100, 100), alpha=1.,
                           activation=activation, solver="lbfgs", max_iter=20000, random_state=0)).fit(X_tr, y_tr)
    results[activation] = model.score(X_te, y_te)
    print(f"{activation:10s} {model.score(X_tr, y_tr):9.4f} {results[activation]:14.4f}", flush=True)
best = max(results.values())
print("Cao nhat tren test minh hoa:", [k for k, v in results.items() if v == best])
print("Ket luan chi cho bo du lieu va seed nay; de chon activation cho trien khai can CV tren tap hoc.")
