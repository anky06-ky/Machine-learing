# -*- coding: utf-8 -*-
"""Bước 5: KNN cho bài toán hồi quy, trên dữ liệu giá nhà của bài 1."""

import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

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


data_path = tim_duong_dan_du_lieu("gia_nha.csv")
df = pd.read_csv(data_path)
X = df[["dien_tich"]]
y = df["gia"]

# Cùng cách chia với bài 1 để so sánh công bằng
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)

knn = KNeighborsRegressor(n_neighbors=5).fit(X_tr, y_tr)
duong = LinearRegression().fit(X_tr, y_tr)

print("MAE tren tap kiem tra (ty dong)")
mae_knn = mean_absolute_error(y_te, knn.predict(X_te))
mae_duong = mean_absolute_error(y_te, duong.predict(X_te))
print(f"  KNN, k = 5 : {mae_knn:.4f}")
print(f"  Duong thang : {mae_duong:.4f}")
print()

can_moi = pd.DataFrame({"dien_tich": [60.0, 100.0, 150.0, 250.0]})
print("Du doan cho bon can moi (ty dong)")
print("  dien tich    KNN  duong thang")
for dt, a, b in zip(can_moi["dien_tich"], knn.predict(can_moi),
                    duong.predict(can_moi)):
    print(f"     {dt:6.0f} m2 {a:6.2f}      {b:6.2f}")
print()

# Năm láng giềng của căn 250 m2 chính là những căn rộng nhất đã thấy
_, chi_so = knn.kneighbors(pd.DataFrame({"dien_tich": [250.0]}))
lg = X_tr.iloc[chi_so[0]]["dien_tich"].tolist()
print("Nam lang gieng cua can 250 m2:", ", ".join(f"{v:.1f}" for v in lg))
