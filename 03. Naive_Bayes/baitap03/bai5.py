# -*- coding: utf-8 -*-
"""Bài tập 5: chọn sai biến thể Naive Bayes."""

from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

data_path = Path(__file__).resolve().parent.parent / "data" / "sinh_vien.csv"
df = pd.read_csv(data_path)
X = df[["gio_on", "diem_giua_ky"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

for ten, mo_hinh in [("MultinomialNB", MultinomialNB()),
                     ("GaussianNB   ", GaussianNB())]:
    mo_hinh.fit(X_train, y_train)
    acc = accuracy_score(y_test, mo_hinh.predict(X_test))
    print(f"{ten}: accuracy = {acc:.4f}")

print()
print("Nhan xet: chuong trinh KHONG bao loi, van in ra mot con so accuracy.")
print("Nhung sai ve nguyen tac: MultinomialNB danh cho du lieu DEM (so lan")
print("xuat hien, khong am). Gio on va diem giua ky la so thuc lien tuc,")
print("khong the dem duoc, nen mo hinh coi chung nhu so dem va tinh xac suat")
print("khong co y nghia. Day la loi im lang: con so van ra nhung khong dang tin.")
