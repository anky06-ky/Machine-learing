# -*- coding: utf-8 -*-
"""Bài tập 3: đổi biến đầu vào sang tuổi nhà, tìm w và b bằng công thức."""

from pathlib import Path

import numpy as np
import pandas as pd

# Đường dẫn tính theo vị trí tệp mã, nên chạy từ thư mục nào cũng được.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gia_nha.csv"
df = pd.read_csv(DATA_PATH)

# Chỉ đổi đúng dòng chọn cột cho biến x so với b3_cong_thuc.py
x = df["tuoi_nha"].to_numpy()
y = df["gia"].to_numpy()

x_tb = x.mean()
y_tb = y.mean()

tu_so = ((x - x_tb) * (y - y_tb)).sum()
mau_so = ((x - x_tb) ** 2).sum()

w = tu_so / mau_so
b = y_tb - w * x_tb

print(f"He so goc  w = {w:.6f}")
print(f"He so chan b = {b:.6f}")
print()

# Hệ số tương quan để trả lời câu hỏi thêm: đường thẳng bám dữ liệu tới đâu.
y_du_doan = w * x + b
r = np.corrcoef(x, y)[0, 1]
r2 = 1 - ((y - y_du_doan) ** 2).sum() / ((y - y_tb) ** 2).sum()
print(f"He so tuong quan giua tuoi_nha va gia: r = {r:.4f}")
print(f"R2 tren 60 can: {r2:.4f} (voi dien_tich la khoang 0.93)")
print()

print(f"Nhan xet: w = {w:.6f} la so am. w la muc thay doi cua gia khi tuoi nha "
      "tang them mot nam, nen nha cang cu thi gia du doan cang giam.")
print("Tuy vay r va R2 deu rat nho, cac diem phan tan manh quanh duong thang. "
      "Dau cua w dung nhung chua du de ket luan hai dai luong co quan he chat.")
