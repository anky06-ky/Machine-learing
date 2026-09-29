# -*- coding: utf-8 -*-
"""Bài 1: thống kê tỷ lệ qua môn theo điểm giữa kỳ."""

from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "sinh_vien.csv"
df = pd.read_csv(DATA_PATH)

# Mỗi nhóm vẫn là một bảng, lọc bằng điều kiện trên điểm giữa kỳ.
nhom_tu_7 = df[df["diem_giua_ky"] >= 7]
nhom_duoi_7 = df[df["diem_giua_ky"] < 7]

so_ban_tu_7 = len(nhom_tu_7)
# Trung bình một cột gồm 0 và 1 chính là tỷ lệ các giá trị 1.
ty_le_tu_7 = nhom_tu_7["qua_mon"].mean()
ty_le_duoi_7 = nhom_duoi_7["qua_mon"].mean()

print("BAI 1: THONG KE THEO NHOM")
print(f"So ban co diem giua ky tu 7 tro len: {so_ban_tu_7}")
print(f"Ty le qua mon cua nhom >= 7: {ty_le_tu_7:.4f} ({ty_le_tu_7:.2%})")
print(f"Ty le qua mon cua nhom < 7:  {ty_le_duoi_7:.4f} ({ty_le_duoi_7:.2%})")
print()

if ty_le_tu_7 > ty_le_duoi_7:
    print("Nhan xet: Nhom diem giua ky tu 7 co ty le qua mon cao hon,")
    print("nen diem giua ky giup phan biet hai nhom trong bo du lieu nay.")
else:
    print("Nhan xet: Nhom diem giua ky tu 7 khong co ty le qua mon cao hon")
    print("nhom con lai trong bo du lieu nay.")
