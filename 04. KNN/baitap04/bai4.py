# -*- coding: utf-8 -*-
"""Bài tập 4: láng giềng có trọng số theo khoảng cách."""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("Accuracy theo k, bo phieu thuong va bo phieu theo khoang cach")
print("    k            tap hoc                 tap kiem tra")
print("           uniform  distance        uniform  distance")
for k in [1, 13, 101]:
    ket_qua = []
    for trong_so in ["uniform", "distance"]:
        mh = KNeighborsClassifier(n_neighbors=k, weights=trong_so)
        mh.fit(X_tr, y_tr)
        ket_qua.append((accuracy_score(y_tr, mh.predict(X_tr)),
                        accuracy_score(y_te, mh.predict(X_te))))
    (hoc_u, kt_u), (hoc_d, kt_d) = ket_qua
    print(f"{k:5d}      {hoc_u:.4f}   {hoc_d:.4f}         "
          f"{kt_u:.4f}   {kt_d:.4f}")

print()
print("Nhan xet: voi k = 101, trong so theo khoang cach giup mo hinh bot "
      "cau tha (kiem tra tang tu 0.8533 len 0.9067) vi lang gieng gan co "
      "phieu nang hon lang gieng xa, nen bien khong con bi 101 diem xa keo "
      "thang. Diem tren tap hoc luon 1.0000 voi distance vi moi diem co "
      "khoang cach 0 toi chinh no, con so nay khong co y nghia.")
