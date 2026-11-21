# -*- coding: utf-8 -*-
"""Bài tập 1: tính tay f(x) và khoảng cách tới ranh giới."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                  cluster_std=0.9, random_state=0)
y = 1 - y
svm = SVC(kernel="linear", C=1000).fit(X, y)
w = svm.coef_[0]
b = svm.intercept_[0]
do_dai_w = np.sqrt(w[0] ** 2 + w[1] ** 2)

print(f"w1 = {w[0]:.3f}, w2 = {w[1]:.3f}, b = {b:.3f}")
print(f"Do rong le: {2 / do_dai_w:.4f}; nua do rong le: {1 / do_dai_w:.4f}")
print()

diem_can_tinh = [
    ("Diem moi (3, 3)", np.array([3.0, 3.0])),
    ("Diem moi (5, 4.5)", np.array([5.0, 4.5])),
]
diem_can_tinh.extend((f"Vecto ho tro {i}", X[i]) for i in svm.support_)
for ten, diem in diem_can_tinh:
    # Tự tính bằng nhân và cộng; không gọi decision_function
    f = w[0] * diem[0] + w[1] * diem[1] + b
    khoang_cach = abs(f) / do_dai_w
    print(f"{ten}: f(x) = {f:+.4f}, khoang cach = {khoang_cach:.4f}")

print()
print("Giai thich: vecto ho tro nam tren hai mep f(x) = +1 hoac -1.")
print("Vi vay khoang cach la 1 / ||w||, bang nua do rong le 2 / ||w||.")
print("Chenh lech rat nho la do sai so toi uu so cua thu vien.")
