# -*- coding: utf-8 -*-
"""Bài 5: dò ngưỡng từ 0.05 đến 0.95 để tìm F1 lớn nhất."""

print("BAI 5: Dang huan luyen va do nguong...", flush=True)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "data" / "sinh_vien.csv"
df = pd.read_csv(DATA_PATH)
X = df[["gio_on"]]
y = df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)
xac_suat = mo_hinh.predict_proba(X_test)[:, 1]

ket_qua = []
print("Nguong     F1")
for nguong in np.round(np.arange(0.05, 1.0, 0.05), 2):
    y_pred = (xac_suat >= nguong).astype(int)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    ket_qua.append((nguong, f1))
    print(f" {nguong:.2f}    {f1:.4f}")

f1_cao_nhat = max(f1 for nguong, f1 in ket_qua)
# Nhiều ngưỡng có thể cho cùng nhãn và cùng F1, nên liệt kê mọi ngưỡng đồng hạng.
nguong_dong_hang = [
    nguong for nguong, f1 in ket_qua if np.isclose(f1, f1_cao_nhat)
]
nguong_tot_nhat = nguong_dong_hang[0]

print()
print(f"F1 cao nhat: {f1_cao_nhat:.4f}")
print("Cac nguong dat F1 cao nhat:", ", ".join(f"{t:.2f}" for t in nguong_dong_hang))
print(f"Chon nguong: {nguong_tot_nhat:.2f} (lay nguong nho nhat khi dong hang).")

if any(np.isclose(t, 0.5) for t in nguong_dong_hang):
    print("Nhan xet: Nguong 0.50 nam trong cac nguong cho F1 cao nhat.")
else:
    print("Nhan xet: Nguong tot nhat theo F1 khac 0.50 tren tap kiem tra nay.")
