# -*- coding: utf-8 -*-
"""Bước 3: chọn số láng giềng k bằng kiểm định chéo."""

import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
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

# Hai cột x1 và x2 có thang đo gần nhau, độ lệch chuẩn chỉ chênh khoảng
# 1.6 lần, nên ở bộ này không chuẩn hoá cũng được.
print("Accuracy tren tap hoc va tap kiem tra theo k")
print("  k   tap hoc  tap kiem tra")
for k in [1, 5, 13, 31, 101]:
    mh = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)
    a_hoc = accuracy_score(y_tr, mh.predict(X_tr))
    a_kt = accuracy_score(y_te, mh.predict(X_te))
    print(f"{k:3d}    {a_hoc:.4f}        {a_kt:.4f}")
print()

# Chọn k chỉ dựa trên tập học, không nhìn tập kiểm tra
print("Kiem dinh cheo 5 phan tren tap hoc:")
diem = {}
for k in range(1, 52, 2):
    diem[k] = cross_val_score(KNeighborsClassifier(n_neighbors=k),
                              X_tr, y_tr, cv=5).mean()
k_tot = max(diem, key=diem.get)
for k in [1, 5, 9, 13, 17, 25, 51]:
    dau = " <- tot nhat" if k == k_tot else ""
    print(f"  k = {k:2d}   diem trung binh {diem[k]:.4f}{dau}")
print()

# Chọn xong mới mở tập kiểm tra, và chỉ mở một lần
mh = KNeighborsClassifier(n_neighbors=k_tot).fit(X_tr, y_tr)
print(f"k tot nhat = {k_tot}")
acc = accuracy_score(y_te, mh.predict(X_te))
print(f"Accuracy tren tap kiem tra = {acc:.4f}")
