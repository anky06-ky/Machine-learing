# -*- coding: utf-8 -*-
"""Bài tập 4: tự viết bộ lọc Naive Bayes, không dùng thư viện học máy."""

import math
from collections import Counter
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data_path = Path(__file__).resolve().parent.parent / "data" / "tin_nhan.csv"
df = pd.read_csv(data_path)

# Cùng random_state và stratify nên chia giống hệt khi chia ma trận đếm
df_train, df_test = train_test_split(
    df, test_size=0.25, random_state=17, stratify=df["la_rac"]
)

# Từ điển lấy từ cả kho, giống CountVectorizer ở d3
tu_dien = sorted({t for cau in df["noi_dung"] for t in cau.split()})
V = len(tu_dien)
alpha = 1.0

# Tự đếm từ theo từng lớp trên tập học
dem = {0: Counter(), 1: Counter()}
tong = {0: 0, 1: 0}
so_tin = {0: 0, 1: 0}
for cau, nhan in zip(df_train["noi_dung"], df_train["la_rac"]):
    so_tin[nhan] += 1
    for t in cau.split():
        dem[nhan][t] += 1
        tong[nhan] += 1

# Loga của xác suất tiên nghiệm và của từng từ (đã làm trơn Laplace)
log_tien_nghiem = {k: math.log(so_tin[k] / len(df_train)) for k in (0, 1)}
log_tu = {k: {t: math.log((dem[k][t] + alpha) / (tong[k] + alpha * V))
              for t in tu_dien} for k in (0, 1)}


def diem(cau, nhan):
    """Tong loga: log P(lop) + tong log P(tu | lop)."""
    d = log_tien_nghiem[nhan]
    for t in cau.split():
        if t in log_tu[nhan]:
            d += log_tu[nhan][t]
    return d


du_doan = [1 if diem(cau, 1) > diem(cau, 0) else 0
           for cau in df_test["noi_dung"]]
acc_tu_viet = accuracy_score(df_test["la_rac"], du_doan)

# Mô hình thư viện trên cùng cách chia
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = MultinomialNB(alpha=alpha).fit(X_train, y_train)
acc_thu_vien = accuracy_score(y_test, mo_hinh.predict(X_test))

print(f"Accuracy bo loc tu viet : {acc_tu_viet:.4f}")
print(f"Accuracy MultinomialNB  : {acc_thu_vien:.4f}")
print()
print("Nhan xet: hai con so trung nhau vi cung dem tu, cung lam tron")
print("Laplace voi alpha = 1 va cung so sanh tong loga cua hai lop.")
print("Dung loga thay vi tich de tranh tran duoi khi tin nhan dai.")
