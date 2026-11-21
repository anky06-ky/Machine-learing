# -*- coding: utf-8 -*-
"""Bài tập 3: đổi tham số làm trơn alpha."""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data_path = Path(__file__).resolve().parent.parent / "data" / "tin_nhan.csv"
df = pd.read_csv(data_path)
vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df["noi_dung"])
y = df["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print(f"{'alpha':>8s} {'accuracy':>9s} {'precision':>10s} {'recall':>8s} {'chenh_lech_max':>15s}")
for alpha in [0.001, 0.1, 1.0, 10.0, 100.0, 1000.0]:
    mo_hinh = MultinomialNB(alpha=alpha).fit(X_train, y_train)
    y_pred = mo_hinh.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    pre = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    # chênh lệch loga lớn nhất giữa hai lớp ở mọi từ: càng nhỏ thì hai lớp càng giống nhau
    cl = abs(mo_hinh.feature_log_prob_[1] - mo_hinh.feature_log_prob_[0]).max()
    ghi_chu = "  <- them vao de xem alpha rat lon" if alpha == 1000.0 else ""
    print(f"{alpha:8g} {acc:9.4f} {pre:10.4f} {rec:8.4f} {cl:15.4f}{ghi_chu}")

print()
print("Nhan xet: alpha qua lon keo moi xac suat tu lai gan nhau (deu xap xi 1/V),")
print("cot chenh_lech_max giam dan, cac tu khong con phan biet duoc hai lop.")
print("Mo hinh gan nhu chi dua vao tien nghiem (lop thuong chiem 60 phan tram),")
print("nen o alpha = 1000 no doan toan bo la thuong: recall ve 0, precision mat nghia.")
