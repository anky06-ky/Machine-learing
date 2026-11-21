# -*- coding: utf-8 -*-
"""Bài tập 3: Kernel đa thức."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

data_path = Path(__file__).resolve().parent.parent / "data" / "hai_vang_trang.csv"
df = pd.read_csv(data_path)
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

luoi = {"svc__C": [0.1, 1, 10, 100]}

print("Ket qua do luoi voi Kernel Da thuc (Poly):")
ket_qua = []
for d in [2, 3, 4]:
    pipe = make_pipeline(StandardScaler(), SVC(kernel="poly", degree=d))
    tim = GridSearchCV(pipe, luoi, cv=5)
    tim.fit(X_tr, y_tr)
    acc_te = tim.score(X_te, y_te)
    print(f"Degree = {d:d}: Cap tot nhat = {tim.best_params_}, "
          f"CV score = {tim.best_score_:.4f}, Test accuracy = {acc_te:.4f}")
    ket_qua.append((d, tim.best_score_, acc_te))

# Chọn bậc bằng điểm CV trên tập học, rồi so với RBF của mục 5
bac_tot, cv_tot, test_tot = max(ket_qua, key=lambda kq: kq[1])
rbf = GridSearchCV(
    SVC(kernel="rbf"),
    {"C": [0.1, 1, 10, 100], "gamma": [0.01, 0.1, 1, 10]},
    cv=5,
).fit(X_tr, y_tr)
test_rbf = rbf.score(X_te, y_te)
print()
print(f"Bac tot nhat theo CV: {bac_tot}, CV = {cv_tot:.4f}")
print(f"RBF muc 5: CV = {rbf.best_score_:.4f}, Test accuracy = {test_rbf:.4f}")
print("Nhan xet: Kernel da thuc "
      + ("vuot" if test_tot > test_rbf else "ngang" if test_tot == test_rbf else "chua vuot")
      + " RBF tren tap kiem tra nay.")
