# -*- coding: utf-8 -*-
"""Bài 2: một mạng XOR khác, hai ReLU tính hai hướng của hiệu."""

import numpy as np

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
h1 = np.maximum(0, X[:, 0]-X[:, 1])
h2 = np.maximum(0, X[:, 1]-X[:, 0])
y = h1+h2
print("x1 x2 h1 h2 y")
print(np.column_stack([X, h1, h2, y]))
np.testing.assert_array_equal(y, np.logical_xor(X[:, 0], X[:, 1]).astype(int))
print("h1+h2 = |x1-x2|: hai bit giong nhau cho 0, khac nhau cho 1, dung bang XOR.")
print("Khong co ReLU, (x1-x2)+(x2-x1) luon bang 0; phi tuyen la phan quyet dinh.")
