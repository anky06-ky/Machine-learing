# -*- coding: utf-8 -*-
"""Bài tập 6: lớp hiếm và xác suất tiên nghiệm."""

from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

data_path = Path(__file__).resolve().parent.parent / "data" / "tin_nhan.csv"
df = pd.read_csv(data_path)

# Giữ toàn bộ 180 tin thường, chỉ lấy 20 tin rác đầu tiên
thuong = df[df["la_rac"] == 0]
rac = df[df["la_rac"] == 1].head(20)
df_lech = pd.concat([thuong, rac], ignore_index=True)

print("So tin theo nhan trong bo moi:")
print(df_lech["la_rac"].value_counts().to_string())
print()

vec = CountVectorizer(token_pattern=r"(?u)\S+")
X = vec.fit_transform(df_lech["noi_dung"])
y = df_lech["la_rac"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = MultinomialNB(alpha=1.0).fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

acc = accuracy_score(y_test, y_pred)
rec = recall_score(y_test, y_pred, pos_label=1)
ty_le_thuong = (y_test == 0).mean()

print(f"Accuracy       = {acc:.4f}")
print(f"Recall lop rac = {rec:.4f}")
print(f"Doan toan 'thuong' cung dat accuracy = {ty_le_thuong:.4f}")
print()
print("Nhan xet: accuracy cao nhung khong phan anh dung chat luong bo loc.")
print("Lop rac chi con 10 phan tram nen tien nghiem cua lop thuong rat lon,")
print("mo hinh de thien ve doan 'thuong'. Chi doan toan 'thuong' cung dat 0.9000,")
print("nen accuracy 0.98 khong noi len bo loc bat duoc bao nhieu tin rac.")
print("Phai xem recall rieng lop rac: tap kiem tra chi co 5 tin rac, bo loc de lot 1 tin.")
