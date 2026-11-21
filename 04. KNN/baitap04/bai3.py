# -*- coding: utf-8 -*-
"""Bài tập 3: đổi đơn vị cột chi_tieu từ nghìn đồng sang triệu đồng."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "sinh_vien_mo_rong.csv")
y = df["qua_mon"]
ba_cot = ["gio_on", "diem_giua_ky", "chi_tieu"]

X_nghin = df[ba_cot]
X_trieu = df[ba_cot].copy()
X_trieu["chi_tieu"] = X_trieu["chi_tieu"] / 1000

# Cung cach chia voi muc 3 de so sanh voi 0.5333
idx_tr, idx_te = train_test_split(
    df.index, test_size=0.25, random_state=17, stratify=y
)


def cham(mo_hinh, X):
    mo_hinh.fit(X.loc[idx_tr], y.loc[idx_tr])
    return accuracy_score(y.loc[idx_te], mo_hinh.predict(X.loc[idx_te]))


def knn_tho():
    return KNeighborsClassifier(n_neighbors=7)


def knn_chuan():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(7))


print("Accuracy tren tap kiem tra, k = 7")
print(f"  Khong chuan hoa, chi_tieu nghin dong : {cham(knn_tho(), X_nghin):.4f}")
print(f"  Khong chuan hoa, chi_tieu trieu dong : {cham(knn_tho(), X_trieu):.4f}")
print(f"  Co chuan hoa,    chi_tieu nghin dong : {cham(knn_chuan(), X_nghin):.4f}")
print(f"  Co chuan hoa,    chi_tieu trieu dong : {cham(knn_chuan(), X_trieu):.4f}")
print()
print("Giai thich: KNN chi dua vao khoang cach. Doi sang trieu dong thi cot "
      "chi_tieu nho di 1000 lan, khong con nuot khoang cach nua, nen ket qua "
      "tu 0.5333 tang len 0.8667 chi vi doi don vi. Khi da chuan hoa, moi cot "
      "duoc do bang do lech chuan cua chinh no, nen doi don vi khong con anh "
      "huong, hai dong co chuan hoa cho cung ket qua.")
