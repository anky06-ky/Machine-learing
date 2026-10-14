# -*- coding: utf-8 -*-
"""Bước 2: đọc kho tin nhắn và biến văn bản thành bảng số."""

import sys
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent / "data" / "tin_nhan.csv"
df = pd.read_csv(DATA_PATH)

print("Kich thuoc bang:", df.shape)
print()
print("So tin theo nhan (1 la rac, 0 la thuong):")
print(df["la_rac"].value_counts())
print()
print("Ty le tin rac:", round(df["la_rac"].mean(), 4))
print()

print("Ba tin nhan dau tien:")
for i in range(3):
    print(f" [{df.la_rac[i]}] {df.noi_dung[i]}")
print()

# token_pattern mặc định của scikit-learn bỏ mất từ một ký tự,
# nên ta chỉ định tách từ bằng khoảng trắng cho đơn giản.
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])

print("Kich thuoc bang dem tu:", X.shape)
print("So tu trong tu dien:", len(vec.vocabulary_))
print()

# Đếm tổng số lần xuất hiện của từng từ, tách theo hai lớp
dem = pd.DataFrame(X.toarray(), columns=vec.get_feature_names_out())
dem["la_rac"] = df["la_rac"]
tong = dem.groupby("la_rac").sum().T
tong.columns = ["thuong", "rac"]

print("Muoi tu xuat hien nhieu nhat trong tin rac:")
print(tong.sort_values("rac", ascending=False).head(10))
print()
print("Muoi tu xuat hien nhieu nhat trong tin thuong:")
print(tong.sort_values("thuong", ascending=False).head(10))
