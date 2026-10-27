# -*- coding: utf-8 -*-
"""Bước 2: đo hậu quả của việc quên chuẩn hoá trên tập kiểm tra."""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


data_path = tim_duong_dan_du_lieu("sinh_vien_mo_rong.csv")
df = pd.read_csv(data_path)
y = df["qua_mon"]

X3 = df[["gio_on", "diem_giua_ky", "chi_tieu"]]
X_tr, X_te, y_tr, y_te = train_test_split(
    X3, y, test_size=0.25, random_state=17, stratify=y
)


def cham(mo_hinh, cot):
    mo_hinh.fit(X_tr[cot], y_tr)
    return accuracy_score(y_te, mo_hinh.predict(X_te[cot]))


def knn_chuan():
    """KNN co chuan hoa, gop lam mot khoi bang pipeline."""
    return make_pipeline(StandardScaler(), KNeighborsClassifier(7))


ba_cot = ["gio_on", "diem_giua_ky", "chi_tieu"]
hai_cot = ["gio_on", "diem_giua_ky"]

a = cham(KNeighborsClassifier(n_neighbors=7), ba_cot)
b = cham(knn_chuan(), ba_cot)
c = cham(knn_chuan(), hai_cot)

print(f"Doan bua ca lop qua          : {y_te.mean():.4f}")
print(f"Ba cot, khong chuan hoa      : {a:.4f}")
print(f"Ba cot, co chuan hoa         : {b:.4f}")
print(f"Bo cot chi_tieu, co chuan hoa: {c:.4f}")
