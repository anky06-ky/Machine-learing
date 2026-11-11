# -*- coding: utf-8 -*-
"""Bài tập 6: MAE, RMSE và R2 khi có một lần đoán hỏng 5 tỷ."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Giống bài 1: dự đoán giá theo diện tích, chia 80/20 với random_state=42
df = pd.read_csv(DATA_DIR / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)

du = LinearRegression().fit(X_tr, y_tr).predict(X_te)
du_hong = du.copy()
du_hong[0] += 5  # đúng một căn bị đoán cao thêm 5 tỷ


def ba_con_so(y_that, y_doan):
    mae = mean_absolute_error(y_that, y_doan)
    rmse = np.sqrt(mean_squared_error(y_that, y_doan))
    return mae, rmse, r2_score(y_that, y_doan)


truoc = ba_con_so(y_te, du)
sau = ba_con_so(y_te, du_hong)
print(f"Tap kiem tra co {len(y_te)} can")
print("                      MAE (ty)  RMSE (ty)      R2")
print(f"  Ban dau            {truoc[0]:8.4f}  {truoc[1]:9.4f}  {truoc[2]:7.4f}")
print(f"  Mot can +5 ty      {sau[0]:8.4f}  {sau[1]:9.4f}  {sau[2]:7.4f}")
print(f"  Tang them          {sau[0] - truoc[0]:8.4f}  {sau[1] - truoc[1]:9.4f}"
      f"  {sau[2] - truoc[2]:7.4f}")
print(f"  Gap bao nhieu lan  {sau[0] / truoc[0]:8.2f}  {sau[1] / truoc[1]:9.2f}")
print()
print("Nhan xet:")
print(f"- MAE chi tang them khoang 5 / {len(y_te)} ty, vi sai so 5 ty duoc chia deu"
      f" cho {len(y_te)} can.")
print("- RMSE binh phuong sai so truoc khi lay trung binh: 5 ty thanh 25,")
print("  lan at cac sai so nho khac, nen RMSE tang manh hon MAE.")
print("- R2 giam manh vi tong binh phuong sai so tang vot; R2 khong co don vi,")
print("  nen so muc thay doi cua MAE va RMSE (cung don vi ty dong) voi nhau.")
