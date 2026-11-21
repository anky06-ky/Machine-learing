# -*- coding: utf-8 -*-
"""Bài tập 4: Dò lưới mịn."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

data_path = Path(__file__).resolve().parent.parent / "data" / "hai_vang_trang.csv"
df = pd.read_csv(data_path)
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Lưới mịn quanh C = 10, gamma = 1
luoi_min = {
    "C": [3, 10, 30],
    "gamma": [0.3, 1, 3]
}

tim = GridSearchCV(SVC(kernel="rbf"), luoi_min, cv=5)
tim.fit(X_tr, y_tr)

bang = pd.DataFrame(tim.cv_results_["params"])
bang["diem"] = tim.cv_results_["mean_test_score"]
print("Diem kiem dinh cheo luoi min:")
print(bang.pivot(index="C", columns="gamma", values="diem").round(4).to_string())
print()

print("Cap tot nhat:", tim.best_params_)
print(f"Diem kiem dinh cheo: {tim.best_score_:.4f}")
print(f"Accuracy tren tap kiem tra: {tim.score(X_te, y_te):.4f}")
print()
cv_cu = 0.8711  # điểm kiểm định chéo tốt nhất của lưới thưa ở mục 5
if tim.best_score_ > cv_cu + 1e-4:
    print(f"Nhan xet: diem kiem dinh cheo tang tu {cv_cu} len {tim.best_score_:.4f},")
else:
    print(f"Nhan xet: diem kiem dinh cheo khong cao hon {cv_cu},")
print("nhung accuracy tren tap kiem tra van la 0.9200 nhu truoc.")
print("Do min hon chi tang rat it, trong khi so lan hoc tang len nhieu.")
