"""Bài 1: tự forward bằng coefs_ và intercepts_, không gọi predict cho phần tính tay."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
import numpy as np
from sklearn.neural_network import MLPClassifier

X_tr, X_te, y_tr, y_te = split_data("xor.csv", "nhan")
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
