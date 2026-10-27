# -*- coding: utf-8 -*-
"""Bước 4: đặt KNN cạnh hai mô hình đã học trên dữ liệu có biên cong."""

import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")


def tim_duong_dan_du_lieu(ten_tap_tin):
    cac_duong_dan = [
        Path(__file__).resolve().parent / "data" / ten_tap_tin,
        Path("data") / ten_tap_tin,
        Path(__file__).resolve().parent.parent / "data" / ten_tap_tin,
        Path(__file__).resolve().parent / ten_tap_tin,
        Path(ten_tap_tin),
    ]
    for p in cac_duong_dan:
        if p.exists():
            return p
    return Path("data") / ten_tap_tin


data_path = tim_duong_dan_du_lieu("hai_vang_trang.csv")
df = pd.read_csv(data_path)
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

cac_mo_hinh = [
    ("Hoi quy logistic", LogisticRegression()),
    ("Naive Bayes     ", GaussianNB()),
    ("KNN, k = 13     ", KNeighborsClassifier(n_neighbors=13)),
]

print("Accuracy tren tap kiem tra, bo hai vang trang")
for ten, mh in cac_mo_hinh:
    mh.fit(X_tr, y_tr)
    print(f"  {ten}: {accuracy_score(y_te, mh.predict(X_te)):.4f}")
