# -*- coding: utf-8 -*-
"""Bài 6: đổi lớp dương sang rớt môn và so sánh precision, recall."""

print("BAI 6: Dang cham diem cho hai cach chon lop duong...", flush=True)

from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "data" / "sinh_vien.csv"
df = pd.read_csv(DATA_PATH)
X = df[["gio_on"]]
y = df["qua_mon"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mo_hinh = LogisticRegression().fit(X_train, y_train)
# Dùng cùng một bộ dự đoán cho cả hai lần chấm điểm.
y_pred = mo_hinh.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

print("Lop duong        Precision    Recall")
for nhan, ten in [(1, "Qua mon (1)"), (0, "Rot mon (0)")]:
    precision = precision_score(y_test, y_pred, pos_label=nhan, zero_division=0)
    recall = recall_score(y_test, y_pred, pos_label=nhan, zero_division=0)
    print(f"{ten:16s} {precision:.4f}       {recall:.4f}")

print()
print(f"Ma tran ban dau (lop duong = 1): TN={tn}, FP={fp}, FN={fn}, TP={tp}.")
print("Doi lop duong sang 0 thi vai tro cua bon o thay doi:")
print(f"TP_moi = TN_cu = {tn}, FP_moi = FN_cu = {fn},")
print(f"FN_moi = FP_cu = {fp}, TN_moi = TP_cu = {tp}.")
print()
print(f"Precision lop 0 = {tn} / ({tn} + {fn}) = {tn / (tn + fn):.4f}.")
print("  Trong cac ban duoc doan rot, bao nhieu ban that su rot?")
print(f"Recall lop 0 = {tn} / ({tn} + {fp}) = {tn / (tn + fp):.4f}.")
print("  Trong cac ban that su rot, mo hinh tim duoc bao nhieu ban?")
print()
print("Giai thich: Chon lop duong khac lam doi nhom duoc dem va mau so,")
print("nen precision va recall khac nhau du mo hinh, nhan du doan va du lieu")
print("giu nguyen; accuracy van la ty le du doan dung chung cua ca hai lop.")
