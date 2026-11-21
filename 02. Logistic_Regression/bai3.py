# -*- coding: utf-8 -*-
"""Bài 3: dự đoán từ số giờ ôn bằng mô hình một biến."""

print("BAI 3: Dang khop mo hinh...", flush=True)

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

DATA_PATH = Path(__file__).resolve().parent / "data" / "sinh_vien.csv"
df = pd.read_csv(DATA_PATH)
X = df[["gio_on"]]
y = df["qua_mon"]

# Bài này dùng toàn bộ dữ liệu, giống mục 4 của tài liệu.
mo_hinh = LogisticRegression().fit(X, y)
w = float(mo_hinh.coef_[0][0])
b = float(mo_hinh.intercept_[0])


def du_doan(gio):
    """Tính z, bọc sigmoid, rồi áp ngưỡng 0.5 để lấy nhãn."""
    z = w * gio + b
    xac_suat = 1 / (1 + np.exp(-z))
    nhan = int(xac_suat >= 0.5)
    print(f"{gio:8.2f} {z:10.6f} {xac_suat:14.6f} {nhan:6d}")


print(f"w = {w:.6f}, b = {b:.6f}")
print("  Gio on          z   Xac suat qua   Nhan")
for gio in [3, 8, 12.89, 18, 26]:
    du_doan(gio)

print()
print(f"Moc xac suat dung 0.5: -b / w = {-b / w:.8f} gio.")
print("Giai thich: 12.89 la gia tri lam tron cua -b / w, nen z gan 0")
print("va sigmoid(z) gan 0.5. Nhan duoc tinh tu xac suat chua lam tron.")
