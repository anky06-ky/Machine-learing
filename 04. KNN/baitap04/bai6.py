# -*- coding: utf-8 -*-
"""Bài tập 6: đo thời gian .fit và .predict của KNN theo cỡ dữ liệu."""

import time

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

rng = np.random.default_rng(0)
X_moi = rng.random((1000, 2))

print("Thoi gian KNN voi 2 dac trung, du doan 1000 mau moi")
print("   so mau    fit (giay)   predict (giay)")
for n in [1000, 10000, 100000]:
    X = rng.random((n, 2))
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    mh = KNeighborsClassifier(n_neighbors=5)

    t0 = time.perf_counter()
    mh.fit(X, y)
    t_fit = time.perf_counter() - t0

    t0 = time.perf_counter()
    mh.predict(X_moi)
    t_du_doan = time.perf_counter() - t0

    print(f"{n:9d}    {t_fit:10.4f}    {t_du_doan:12.4f}")

print()
print("Giai thich: .fit chi cat du lieu vao bo nho nen gan nhu tuc thi, con "
      ".predict phai so mau moi voi du lieu da luu nen cang nhieu mau cang "
      "cham. Do la ly do KNN duoc goi la mo hinh hoc luoi.")
