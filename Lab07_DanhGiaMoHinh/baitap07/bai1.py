# -*- coding: utf-8 -*-
"""Bài tập 1: tự viết hàm đánh giá chỉ bằng numpy."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def danh_gia(y_that, y_doan):
    """Trả về accuracy, precision, recall, F1 của lớp dương (nhãn 1).

    Khi mẫu số bằng 0 thì thước đo đó không xác định; hàm quy ước trả 0,
    giống tham số zero_division=0 của scikit-learn.
    """
    y_that = np.asarray(y_that)
    y_doan = np.asarray(y_doan)
    if y_that.shape != y_doan.shape or y_that.size == 0:
        raise ValueError("Hai mang nhan phai cung kich thuoc va khong rong")

    # Bốn ô của ma trận nhầm lẫn
    tp = np.sum((y_that == 1) & (y_doan == 1))
    fp = np.sum((y_that == 0) & (y_doan == 1))
    fn = np.sum((y_that == 1) & (y_doan == 0))
    tn = np.sum((y_that == 0) & (y_doan == 0))

    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall > 0 else 0.0)
    return float(accuracy), float(precision), float(recall), float(f1)


df = pd.read_csv(DATA_DIR / "sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = {
    "Hoi quy logistic": make_pipeline(StandardScaler(), LogisticRegression()),
    "Mo hinh luoi": DummyClassifier(strategy="most_frequent"),
}

print("                   accuracy  precision  recall      F1")
for ten, mh in mo_hinh.items():
    du = mh.fit(X_tr, y_tr).predict(X_te)
    tu_tinh = danh_gia(y_te, du)
    thu_vien = (accuracy_score(y_te, du),
                precision_score(y_te, du, zero_division=0),
                recall_score(y_te, du, zero_division=0),
                f1_score(y_te, du, zero_division=0))
    print(ten)
    print("  ham danh_gia   " + "".join(f"{v:10.4f}" for v in tu_tinh))
    print("  scikit-learn   " + "".join(f"{v:10.4f}" for v in thu_vien))
    print("  khop nhau:", bool(np.allclose(tu_tinh, thu_vien)))
    tp = int(((du == 1) & (y_te == 1)).sum())
    fp = int(((du == 1) & (y_te == 0)).sum())
    print(f"  TP = {tp}, FP = {fp}, so ban bi bao = TP + FP = {tp + fp}")
print()

print("Nhan xet:")
print("- Mo hinh luoi luon doan lop 0, nen TP + FP = 0.")
print("  Precision = TP / (TP + FP) = 0 / 0, khong tinh duoc.")
print("- Ham danh_gia kiem tra mau so truoc khi chia: neu bang 0 thi tra 0,")
print("  giong zero_division=0 cua scikit-learn. Neu goi precision_score")
print("  khong co zero_division, thu vien bao UndefinedMetricWarning va cung tra 0.")
print("- Accuracy 0.9000 cua mo hinh luoi chi la muc san; recall 0 cho thay")
print("  no khong tim ra ban bo hoc nao.")
