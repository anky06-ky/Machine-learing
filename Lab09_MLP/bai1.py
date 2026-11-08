# -*- coding: utf-8 -*-
"""Bài 1: tự forward bằng coefs_ và intercepts_, không gọi predict cho phần tính tay."""

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "xor.csv")
X = df.drop(columns="nhan")
y = df["nhan"]
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)
mlp = MLPClassifier(hidden_layer_sizes=(8,), solver="lbfgs", max_iter=5000, random_state=0).fit(X_tr, y_tr)
points = X_te.iloc[:5]
W1, W2 = mlp.coefs_
b1, b2 = mlp.intercepts_
H = np.maximum(0, points.to_numpy()@W1+b1)
z = H@W2+b2
manual = 1/(1+np.exp(-np.clip(z, -700, 700)))
library = mlp.predict_proba(points)[:, 1]
print("H = ReLU(X @ W1 + b1), p = sigmoid(H @ W2 + b2)")
print("Shape X, W1, W2:", points.shape, W1.shape, W2.shape)
for i, (a, b) in enumerate(zip(manual.ravel(), library)):
    print(f"Diem {i}: numpy={a:.12f}, sklearn={b:.12f}, lech={abs(a-b):.3e}")
np.testing.assert_allclose(manual.ravel(), library, rtol=1e-12, atol=1e-12)
print("Khop toi sai so so thuc; cot thu hai la xac suat lop", mlp.classes_[1])
print("Moi cot W1 la trong so cua mot no-ron an; bias duoc cong theo tung cot.")
