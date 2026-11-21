# -*- coding: utf-8 -*-
"""Bài tập 6: rừng ngẫu nhiên cho bài toán hồi quy trên dữ liệu giá nhà."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

df = pd.read_csv(Path(__file__).resolve().parent.parent / "data" / "gia_nha.csv")
X = df[["dien_tich"]]
y = df["gia"]

# Cùng cách chia với mục 8
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42)

rung = RandomForestRegressor(n_estimators=200, random_state=0)
rung.fit(X_tr, y_tr)
cay = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X_tr, y_tr)
duong = LinearRegression().fit(X_tr, y_tr)

print("MAE tren tap kiem tra (ty dong)")
print(f"  Rung 200 cay: {mean_absolute_error(y_te, rung.predict(X_te)):.4f}")
print(f"  Cay sau 3   : {mean_absolute_error(y_te, cay.predict(X_te)):.4f}")
print(f"  Duong thang : {mean_absolute_error(y_te, duong.predict(X_te)):.4f}")
print()

can_250 = pd.DataFrame({"dien_tich": [250.0]})
print("Du doan gia can 250 m2 (ty dong)")
print(f"  Rung 200 cay: {rung.predict(can_250)[0]:.2f}")
print(f"  Cay sau 3   : {cay.predict(can_250)[0]:.2f}")
print(f"  Duong thang : {duong.predict(can_250)[0]:.2f}")
print()

# Mỗi cây trong rừng đoán bao nhiêu cho căn 250 m2
tung_cay = [c.predict(can_250.to_numpy())[0] for c in rung.estimators_]
print("Du doan cua tung cay trong rung cho can 250 m2")
print(f"  thap nhat {min(tung_cay):.2f}, cao nhat {max(tung_cay):.2f}, "
      f"trung binh {sum(tung_cay) / len(tung_cay):.2f}")
print(f"Gia cao nhat trong tap hoc: {y_tr.max():.2f}")
print(f"Dien tich lon nhat trong tap hoc: {X_tr['dien_tich'].max():.1f} m2")
print()

print("Nhan xet: rung khong ngoai suy duoc.")
print("Moi cay chia theo cac nguong dien tich da thay trong tap hoc. Can 250 m2")
print("lon hon moi nguong nen roi vao la ngoai cung ben phai, va la do chi")
print("dua ra trung binh gia cua may can rong nhat tap hoc. Rung chi lay")
print("trung binh cac du doan do, nen khong vuot qua muc gia da thay,")
print("trong khi duong thang van tiep tuc tang theo dien tich.")
