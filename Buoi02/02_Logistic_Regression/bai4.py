# -*- coding: utf-8 -*-
"""Bài 4: tự tính bốn thước đo chỉ từ TP, TN, FP, FN."""

print("BAI 4: Dang huan luyen va du doan...", flush=True)

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "data" / "sinh_vien.csv"
df = pd.read_csv(DATA_PATH)
X = df[["gio_on"]]
y = df["qua_mon"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)
y_pred = mo_hinh.predict(X_test)

# Hàng là nhãn thật, cột là nhãn dự đoán: [[TN, FP], [FN, TP]].
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

# Tự tính từ bốn ô của ma trận; không dùng hàm chấm điểm có sẵn.
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp)
recall = tp / (tp + fn)
f1 = 2 * tp / (2 * tp + fp + fn)

print(f"So sinh vien de hoc: {len(X_train)}, de kiem tra: {len(X_test)}")
print(f"TN = {tn}, FP = {fp}, FN = {fn}, TP = {tp}")
print()
print("Thuoc do    Tu tinh    Tai lieu")
# Cột tài liệu lấy từ kết quả tham chiếu trong Lab02, không tính bằng thư viện.
for ten, gia_tri, tham_chieu in [
    ("Accuracy", accuracy, 0.8667),
    ("Precision", precision, 0.8500),
    ("Recall", recall, 0.9444),
    ("F1", f1, 0.8947),
]:
    ket_luan = "KHOP" if f"{gia_tri:.4f}" == f"{tham_chieu:.4f}" else "KHAC"
    print(f"{ten:10s}  {gia_tri:.4f}     {tham_chieu:.4f}  {ket_luan}")

print()
print("Giai thich cac cong thuc (lop duong la qua mon):")
print(f"Accuracy = (TP + TN) / tong = ({tp} + {tn}) / {tp + tn + fp + fn}.")
print("  Do ty le du doan dung cua ca hai lop.")
print(f"Precision = TP / (TP + FP) = {tp} / ({tp} + {fp}).")
print("  Trong cac ban duoc doan qua, bao nhieu ban that su qua?")
print(f"Recall = TP / (TP + FN) = {tp} / ({tp} + {fn}).")
print("  Trong cac ban that su qua, mo hinh tim duoc bao nhieu ban?")
print(f"F1 = 2*TP / (2*TP + FP + FN) = {2 * tp} / {2 * tp + fp + fn}.")
print("  F1 la trung binh dieu hoa cua precision va recall.")
