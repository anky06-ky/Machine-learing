# -*- coding: utf-8 -*-
"""Bài tập 5: dò lưới độ lớn mạng và alpha bằng GridSearchCV."""

import pandas as pd
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

mo_hinh = make_pipeline(
    StandardScaler(),
    MLPClassifier(max_iter=1000, random_state=0),
)
luoi = {
    "mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
    "mlpclassifier__alpha": [0.0001, 0.01, 1],
}
# Kiểm định chéo 5 phần, chỉ trên tập học; scaler nằm trong pipeline
dl = GridSearchCV(mo_hinh, luoi, cv=5).fit(X_tr, y_tr)

bang = pd.DataFrame(dl.cv_results_)
bang = bang.rename(columns={
    "param_mlpclassifier__hidden_layer_sizes": "lop_an",
    "param_mlpclassifier__alpha": "alpha",
    "mean_test_score": "kiem_dinh_cheo",
    "rank_test_score": "hang",
})[["lop_an", "alpha", "kiem_dinh_cheo", "hang"]]
print(bang.sort_values("hang").round(4).to_string(index=False))
print()

tot = dl.best_params_
print("Cap tot nhat:", tot["mlpclassifier__hidden_layer_sizes"],
      "alpha =", tot["mlpclassifier__alpha"],
      f"| kiem dinh cheo {dl.best_score_:.4f}")

# Chọn xong mới chấm tập kiểm tra, một lần
acc_mlp = accuracy_score(y_te, dl.predict(X_te))
acc_svm = accuracy_score(y_te, SVC().fit(X_tr, y_tr).predict(X_te))
print(f"Tap kiem tra: MLP tot nhat {acc_mlp:.4f}, SVM kernel RBF {acc_svm:.4f}")
print("MLP vuot SVM:", "co" if acc_mlp > acc_svm else "khong")
