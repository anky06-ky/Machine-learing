# -*- coding: utf-8 -*-
"""Bài tập 5: chọn k cho KNN hồi quy trên dữ liệu giá nhà."""

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)

# Diem tra ve mang dau am nen doi dau lai de doc thanh MAE
print("Kiem dinh cheo 5 phan tren tap hoc (MAE, ty dong)")
mae_cv = {}
for k in range(1, 21):
    diem = cross_val_score(KNeighborsRegressor(n_neighbors=k), X_tr, y_tr,
                           cv=5, scoring="neg_mean_absolute_error")
    mae_cv[k] = -diem.mean()
    print(f"  k = {k:2d}   MAE trung binh {mae_cv[k]:.4f}")
k_tot = min(mae_cv, key=mae_cv.get)
print()

knn = KNeighborsRegressor(n_neighbors=k_tot).fit(X_tr, y_tr)
duong = LinearRegression().fit(X_tr, y_tr)
mae_knn = mean_absolute_error(y_te, knn.predict(X_te))
mae_duong = mean_absolute_error(y_te, duong.predict(X_te))

print(f"k tot nhat = {k_tot}")
print("MAE tren tap kiem tra (ty dong)")
print(f"  KNN, k = {k_tot}: {mae_knn:.4f}")
print(f"  Duong thang : {mae_duong:.4f}")
print()
if mae_knn < mae_duong:
    print("Nhan xet: KNN voi k tot nhat da vuot duoc duong thang.")
else:
    print("Nhan xet: KNN voi k tot nhat van khong vuot duoc duong thang, "
          "vi gia nha tang gan nhu thang theo dien tich.")
