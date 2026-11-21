# -*- coding: utf-8 -*-
"""Bài tập 5: cân bằng lớp cho rừng ngẫu nhiên và SVM."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "sinh_vien_bo_hoc.csv")
X = df.drop(columns="bo_hoc")
y = df["bo_hoc"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)


def tao_mo_hinh(ten, can_bang):
    cw = "balanced" if can_bang else None
    if ten == "Rung 200 cay":
        return RandomForestClassifier(200, random_state=0, class_weight=cw)
    # SVM luôn cần chuẩn hoá, như bài 6
    return make_pipeline(StandardScaler(), SVC(class_weight=cw))


print("Tap kiem tra                  so bao  precision  recall      F1")
thay_doi = {}
for ten in ["Rung 200 cay", "SVM RBF"]:
    ket_qua = []
    for can_bang in [False, True]:
        du = tao_mo_hinh(ten, can_bang).fit(X_tr, y_tr).predict(X_te)
        d = (precision_score(y_te, du, zero_division=0),
             recall_score(y_te, du), f1_score(y_te, du))
        ket_qua.append(d)
        nhan = "can bang lop" if can_bang else "binh thuong"
        print(f"  {ten:12s} {nhan:13s} {du.sum():5d}    {d[0]:.4f}"
              f"    {d[1]:.4f}  {d[2]:.4f}")
    truoc, sau = ket_qua
    thay_doi[ten] = [s - t for t, s in zip(truoc, sau)]
print()

print("Thay doi sau khi can bang (sau - truoc):")
print("                precision   recall       F1")
for ten, d in thay_doi.items():
    print(f"  {ten:12s}" + "".join(f"{v:+11.4f}" for v in d))
nhieu_nhat = max(thay_doi, key=lambda t: sum(abs(v) for v in thay_doi[t]))
print("Mo hinh thay doi nhieu nhat:", nhieu_nhat)
