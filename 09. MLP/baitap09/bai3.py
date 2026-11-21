# -*- coding: utf-8 -*-
"""Bài tập 3: so các hàm kích hoạt trên bộ hai vầng trăng."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Cùng mạng như mục 4: hai lớp ẩn 100 nơ-ron, alpha=1.0, chỉ đổi activation
print("  activation   tap hoc  tap kiem tra  so vong")
ket_qua = {}
for ham in ["logistic", "tanh", "relu"]:
    m = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100, 100), alpha=1.0,
                      activation=ham, solver="lbfgs", max_iter=20000,
                      random_state=0),
    ).fit(X_tr, y_tr)
    a_hoc = accuracy_score(y_tr, m.predict(X_tr))
    a_kt = accuracy_score(y_te, m.predict(X_te))
    ket_qua[ham] = a_kt
    print(f"  {ham:10s}   {a_hoc:.4f}   {a_kt:.4f}        {m[-1].n_iter_}")
tot = max(ket_qua.values())
print("Tap kiem tra cao nhat:", [h for h, v in ket_qua.items() if v == tot])
