# -*- coding: utf-8 -*-
"""Bài tập 1: tự dự đoán bằng trọng số mạng đã học."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "xor.csv")
X = df[["x1", "x2"]]
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

# Học lại mạng 8 nơ-ron ẩn như mục 3
mlp = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs",
                    max_iter=5000, random_state=0).fit(X_tr, y_tr)

W1, W2 = mlp.coefs_       # (2, 8) và (8, 1)
b1, b2 = mlp.intercepts_  # (8,) và (1,)
print("Kich thuoc: W1", W1.shape, " b1", b1.shape,
      " W2", W2.shape, " b2", b2.shape)
print("Ham kich hoat lop an:", mlp.activation,
      "| lop ra:", mlp.out_activation_)
print()

# Tự tính lan truyền tiến cho 5 điểm đầu của tập kiểm tra
X5 = X_te.iloc[:5].to_numpy()
Z1 = X5 @ W1 + b1                # tổng có trọng số của 8 nơ-ron ẩn
H = np.maximum(0, Z1)            # ReLU
z2 = H @ W2 + b2                 # tổng có trọng số của nơ-ron ra
p_tu_tinh = 1 / (1 + np.exp(-z2[:, 0]))  # sigmoid
p_thu_vien = mlp.predict_proba(X_te.iloc[:5])[:, 1]

print("      x1       x2   nhan   tu tinh          predict_proba    chenh lech")
for i in range(5):
    print(f"  {X5[i, 0]:7.4f}  {X5[i, 1]:7.4f}   {y_te.iloc[i]}    "
          f"{p_tu_tinh[i]:.12f}   {p_thu_vien[i]:.12f}   "
          f"{abs(p_tu_tinh[i] - p_thu_vien[i]):.1e}")
print("Khop toi 12 chu so thap phan:",
      bool(np.allclose(p_tu_tinh, p_thu_vien, rtol=0, atol=1e-12)))
print()

# Diễn giải điểm đầu tiên: nơ-ron ẩn nào đang bật
print("Diem dau tien, gia tri 8 no-ron an sau ReLU:")
print("  ", np.round(H[0], 4))
print(f"  so no-ron bat (> 0): {int((H[0] > 0).sum())} tren 8;"
      f" z cua lop ra = {z2[0, 0]:.4f} -> p = {p_tu_tinh[0]:.4f}")
print()
print("Nhan xet:")
print("- Moi cot cua W1 la trong so vao mot no-ron an, b1 cong theo cot.")
print("- Lop an dung ReLU, lop ra dung sigmoid (logistic) vi bai toan hai lop.")
print("- Cot thu hai cua predict_proba la xac suat nhan 1, dung bang sigmoid(z).")
print("- Mang chi la cac phep nhan ma tran, cong va ham kich hoat; predict_proba")
print("  lam dung cac phep tinh do nen hai ket qua trung nhau.")
