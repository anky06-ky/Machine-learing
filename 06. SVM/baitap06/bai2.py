# -*- coding: utf-8 -*-
"""Bài tập 2: Thêm một điểm."""

import numpy as np
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

X, y = make_blobs(n_samples=24, centers=[[2, 2], [6, 6]],
                  cluster_std=0.9, random_state=0)
y = 1 - y

svm_goc = SVC(kernel="linear", C=1000).fit(X, y)
w_goc, b_goc = svm_goc.coef_[0], svm_goc.intercept_[0]
print(f"Ban dau       : w1 = {w_goc[0]:.3f}, w2 = {w_goc[1]:.3f}, b = {b_goc:.3f}")

# Lần 1: Thêm điểm (-1, -1) thuộc lớp 1
X_lan1 = np.vstack([X, [-1.0, -1.0]])
y_lan1 = np.append(y, 1)
svm1 = SVC(kernel="linear", C=1000).fit(X_lan1, y_lan1)
w1, b1 = svm1.coef_[0], svm1.intercept_[0]
print(f"Them (-1, -1) : w1 = {w1[0]:.3f}, w2 = {w1[1]:.3f}, b = {b1:.3f}")

# Lần 2: Thêm điểm (4.5, 4.0) thuộc lớp 1 (bỏ điểm (-1, -1))
X_lan2 = np.vstack([X, [4.5, 4.0]])
y_lan2 = np.append(y, 1)
svm2 = SVC(kernel="linear", C=1000).fit(X_lan2, y_lan2)
w2, b2 = svm2.coef_[0], svm2.intercept_[0]
print(f"Them (4.5, 4) : w1 = {w2[0]:.3f}, w2 = {w2[1]:.3f}, b = {b2:.3f}")

# Nhận xét:
# - Điểm (-1, -1) nằm rất sâu bên trong vùng lớp 1 (|f(x)| >> 1), không là SV nên w, b không đổi.
# - Điểm (4.5, 4.0) nằm lấn vào lề/vùng ranh giới, trở thành vectơ hỗ trợ mới làm thay đổi lề và ranh giới.
print()
print("Nhan xet:")
print("  (-1, -1) nam xa le ve phia lop 1, khong la vecto ho tro; ranh gioi gan nhu khong doi.")
print("  (4.5, 4) vi pham le cu, tro thanh vecto ho tro; ranh gioi thay doi.")
