# -*- coding: utf-8 -*-
"""Bài 6: tăng đúng một dự đoán giá nhà thêm 5 tỷ."""

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "gia_nha.csv")
X_tr, X_te, y_tr, y_te = train_test_split(df[["dien_tich"]], df.gia, test_size=0.2, random_state=42)
pred = LinearRegression().fit(X_tr, y_tr).predict(X_te)
bad = pred.copy(); bad[0] += 5
results = []
for name, values in [("Ban dau", pred), ("Mot can +5 ty", bad)]:
    scores = np.array([mean_absolute_error(y_te, values), np.sqrt(mean_squared_error(y_te, values)), r2_score(y_te, values)])
    results.append(scores)
    print(f"{name:15s} MAE={scores[0]:.4f} ty, RMSE={scores[1]:.4f} ty, R2={scores[2]:.4f}")
print("Thay doi MAE, RMSE, R2:", results[1]-results[0])
print("Tang tuong doi MAE, RMSE (%):", 100*(results[1][:2]/results[0][:2]-1))
print("RMSE binh phuong sai so truoc khi lay trung binh nen phat nang du doan hong.")
print("Chi so khong cung thang do: chi so sanh truc tiep MAE va RMSE (ty dong); R2 khong don vi.")
