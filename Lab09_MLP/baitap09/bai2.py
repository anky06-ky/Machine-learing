# -*- coding: utf-8 -*-
"""Bài tập 2: kiểm chứng một mạng XOR khác bằng numpy.

    h1 = max(0, x1 - x2)
    h2 = max(0, x2 - x1)
    y  = h1 + h2
"""

import numpy as np


def relu(z):
    return np.maximum(0, z)


# Viết dưới dạng ma trận như m1_lan_truyen_tien.py
W1 = np.array([[1.0, -1.0],   # cột 1: h1 = x1 - x2
               [-1.0, 1.0]])  # cột 2: h2 = x2 - x1
b1 = np.array([0.0, 0.0])
W2 = np.array([1.0, 1.0])
b2 = 0.0

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
H = relu(X @ W1 + b1)
Y = H @ W2 + b2
xor = np.logical_xor(X[:, 0], X[:, 1]).astype(int)

print("  x1  x2   h1  h2   y   XOR")
for x, h, yy, xx in zip(X, H, Y, xor):
    print(f"  {x[0]:.0f}   {x[1]:.0f}    {h[0]:.0f}   {h[1]:.0f}"
          f"    {yy:.0f}   {xx}")
print("Mang tinh dung XOR:", bool(np.array_equal(Y, xor)))
print()

# Nếu bỏ ReLU, mạng chỉ còn là một phép tuyến tính
Y_khong_relu = (X @ W1 + b1) @ W2 + b2
print("Neu bo ReLU, y =", Y_khong_relu)
print()
print("Giai thich:")
print("- h1 'bat' khi x1 > x2, h2 'bat' khi x2 > x1; ReLU chan phan am ve 0,")
print("  nen moi luc nhieu nhat chi mot no-ron an khac 0.")
print("- y = h1 + h2 = max(0, x1 - x2) + max(0, x2 - x1) = |x1 - x2|.")
print("- Hai bit giong nhau (00, 11): hieu bang 0, y = 0.")
print("  Hai bit khac nhau (01, 10): hieu bang 1 hoac -1, y = 1. Dung bang XOR.")
print("- Khong co ReLU thi (x1 - x2) + (x2 - x1) = 0 voi moi dau vao:")
print("  chinh ham kich hoat phi tuyen giup mang lam duoc dieu mot")
print("  duong thang khong lam duoc.")
