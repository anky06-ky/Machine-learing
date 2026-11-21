# -*- coding: utf-8 -*-
"""Bài tập 2: vẽ biểu đồ phân tán giữa số phòng và giá."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

THU_MUC = Path(__file__).resolve().parent
df = pd.read_csv(THU_MUC.parent / "data" / "gia_nha.csv")

plt.figure(figsize=(8, 5))
plt.scatter(df["so_phong"], df["gia"], alpha=0.75)
plt.xlabel("So phong")
plt.ylabel("Gia (ty dong)")
plt.title("Gia can ho theo so phong")
plt.xticks(sorted(df["so_phong"].unique()))

# Lưu hình trước khi gọi show, vì show đóng hình lại và lưu sau sẽ ra tệp trắng.
plt.savefig(THU_MUC / "bai2.png", dpi=150, bbox_inches="tight")
plt.show()

# Giá trung bình của từng nhóm số phòng để làm căn cứ nhận xét.
print("Gia trung binh theo so phong (ty dong):")
print(df.groupby("so_phong")["gia"].mean().round(2))
print()
print("Nhan xet: so phong cang nhieu thi gia cang cao, cac diem di len tu trai "
      "sang phai. Tuy vay moi nhom so phong van trai ra nhieu muc gia khac nhau.")
