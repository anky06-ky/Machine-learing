# -*- coding: utf-8 -*-
"""Bài tập 1: tự viết KNN không dùng thư viện, so với KNeighborsClassifier."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def knn_du_doan(X_hoc, y_hoc, x_moi, k):
    """Tra ve nhan du doan cho mot mau moi bang cach bo phieu k lang gieng."""
    khoang = np.sqrt(((X_hoc - x_moi) ** 2).sum(axis=1))
    gan = np.argsort(khoang)[:k]
    nhan, dem = np.unique(y_hoc[gan], return_counts=True)
    return nhan[np.argmax(dem)]


df = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

k = 13
X_hoc = X_tr.to_numpy(dtype=float)
y_hoc = y_tr.to_numpy()
du_doan_tay = np.array([knn_du_doan(X_hoc, y_hoc, x, k)
                        for x in X_te.to_numpy(dtype=float)])
acc_tay = accuracy_score(y_te, du_doan_tay)

mh = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
acc_thu_vien = accuracy_score(y_te, mh.predict(X_te))

print(f"Accuracy KNN tu viet, k = {k}   : {acc_tay:.4f}")
print(f"Accuracy KNeighborsClassifier : {acc_thu_vien:.4f}")
print("Hai con so", "bang nhau" if acc_tay == acc_thu_vien else "KHAC nhau")
