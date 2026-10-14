# -*- coding: utf-8 -*-
"""Bước 5: Naive Bayes cho số liệu liên tục, đặt cạnh hồi quy logistic."""

import sys
from pathlib import Path
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Đảm bảo hiển thị tiếng Việt trên terminal Windows không bị lỗi encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

DATA_PATH = Path(__file__).resolve().parent / "data" / "sinh_vien.csv"
df = pd.read_csv(DATA_PATH)
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Hai mô hình, cùng dữ liệu, cùng cách chia, nên so sánh mới công bằng
for ten, mo_hinh in [("Naive Bayes    ", GaussianNB()),
                     ("Hoi quy logistic", LogisticRegression())]:
    mo_hinh.fit(X_train, y_train)
    acc = accuracy_score(y_test, mo_hinh.predict(X_test))
    print(f"{ten}: accuracy = {acc:.4f}")
print()

gnb = GaussianNB().fit(X_train, y_train)

# theta_ la trung binh, var_ la phuong sai, moi dong mot lop
print("Nhung gi Naive Bayes hoc duoc")
print(f" {'':16s} {'gio_on':>10s} {'diem_giua_ky':>14s}")
for i, ten_lop in enumerate(["Lop 0 (rot) ", "Lop 1 (qua) "]):
    tb = gnb.theta_[i]
    dl = gnb.var_[i] ** 0.5
    print(f" {ten_lop} trung binh {tb[0]:10.2f} {tb[1]:14.2f}")
    print(f" {'':12s} do lech    {dl[0]:10.2f} {dl[1]:14.2f}")
print()

ty_le = ", ".join(f"{float(v):.4f}" for v in gnb.class_prior_)
print(f"Ty le tung lop trong tap hoc: {ty_le}")
print()

# Dự đoán cho vài sinh viên giả định
ban_moi = pd.DataFrame({"gio_on": [5, 13, 20, 28],
                        "diem_giua_ky": [4.0, 6.0, 7.0, 9.0]})
p = gnb.predict_proba(ban_moi)[:, 1]
print("Du doan cho bon ban moi:")
print(f" {'gio on':>8s} {'diem GK':>9s} {'P(qua mon)':>12s}")
for (_, r), pi in zip(ban_moi.iterrows(), p):
    print(f" {r.gio_on:8.0f} {r.diem_giua_ky:9.1f} {pi:12.4f}")
