# -*- coding: utf-8 -*-
"""Bài tập 1: lọc ra nhóm căn hộ lớn (diện tích trên 100 m2)."""

from pathlib import Path

import pandas as pd

# Đường dẫn tính theo vị trí tệp mã, nên chạy từ thư mục nào cũng được.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gia_nha.csv"
df = pd.read_csv(DATA_PATH)

# Lọc bằng điều kiện đúng sai (boolean mask).
nhom_lon = df[df["dien_tich"] > 100]

so_can = len(nhom_lon)
gia_trung_binh = nhom_lon["gia"].mean()

print("So can co dien tich lon hon 100 m2:", so_can)
print(f"Gia trung binh cua nhom do: {gia_trung_binh:.4f} ty dong")
