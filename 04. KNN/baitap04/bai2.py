# -*- coding: utf-8 -*-
"""Bài tập 2: khoảng cách Manhattan so với Euclid trên bộ hai vầng trăng."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def chon_k(metric):
    """Chon k le tu 1 den 51 bang kiem dinh cheo 5 phan tren tap hoc."""
    diem = {}
    for k in range(1, 52, 2):
        mh = KNeighborsClassifier(n_neighbors=k, metric=metric)
        diem[k] = cross_val_score(mh, X_tr, y_tr, cv=5).mean()
    k_tot = max(diem, key=diem.get)
    mh = KNeighborsClassifier(n_neighbors=k_tot, metric=metric).fit(X_tr, y_tr)
    return k_tot, diem[k_tot], accuracy_score(y_te, mh.predict(X_te))


print("Khoang cach   k tot nhat   diem CV   accuracy tap kiem tra")
ket_qua = {}
for ten, metric in [("Euclid   ", "euclidean"), ("Manhattan", "manhattan")]:
    k_tot, d_cv, acc = chon_k(metric)
    ket_qua[metric] = acc
    print(f"{ten}     {k_tot:7d}     {d_cv:.4f}    {acc:.4f}")

chenh = abs(ket_qua["manhattan"] - ket_qua["euclidean"])
print()
print(f"Nhan xet: accuracy chenh nhau {chenh:.4f}, doi cach do khoang cach "
      "gan nhu khong lam ket qua khac di nhieu tren bo nay.")
