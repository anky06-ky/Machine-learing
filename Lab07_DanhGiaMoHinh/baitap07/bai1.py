"""Bài 1: bốn thước đo bằng NumPy, kể cả trường hợp không báo lớp 1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def danh_gia(y_that, y_doan):
    """Trả dict; quy ước precision/recall/F1 bằng 0 khi mẫu số bằng 0."""
    that, doan = np.asarray(y_that), np.asarray(y_doan)
    if that.ndim != 1 or that.shape != doan.shape or that.size == 0:
        raise ValueError("Hai mang phai mot chieu, cung kich thuoc va khong rong")
    if not np.isin(that, [0, 1]).all() or not np.isin(doan, [0, 1]).all():
        raise ValueError("Chi chap nhan nhan nhi phan 0, 1")
    tp = np.sum((that == 1) & (doan == 1))
    fp = np.sum((that == 0) & (doan == 1))
    fn = np.sum((that == 1) & (doan == 0))
    precision = float(tp/(tp+fp)) if tp+fp else 0.0
    recall = float(tp/(tp+fn)) if tp+fn else 0.0
    return {"accuracy": float(np.mean(that == doan)), "precision": precision,
            "recall": recall, "f1": 2*precision*recall/(precision+recall) if precision+recall else 0.0}


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = split_data()
    for name, model in [("Logistic", make_pipeline(StandardScaler(), LogisticRegression())),
                        ("Luoi", DummyClassifier(strategy="most_frequent"))]:
        du = model.fit(X_tr, y_tr).predict(X_te)
        manual = danh_gia(y_te, du)
        library = [accuracy_score(y_te, du), precision_score(y_te, du, zero_division=0),
                   recall_score(y_te, du, zero_division=0), f1_score(y_te, du, zero_division=0)]
        np.testing.assert_allclose(list(manual.values()), library)
        print(name, manual, "- khop scikit-learn")
    print("Mo hinh luoi khong bao lop 1: TP+FP=0, precision khong xac dinh.")
    print("Quy uoc tra 0, tuong duong zero_division=0; khong phai precision do duoc.")
