# -*- coding: utf-8 -*-
"""Bài tập 3: so khởi tạo ngẫu nhiên và K-Means++, mỗi lần chỉ n_init=1."""

from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "khach_hang.csv")
Xs = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])

ket = {}
for init in ["random", "k-means++"]:
    ket[init] = [round(KMeans(n_clusters=5, init=init, n_init=1,
                              random_state=rs).fit(Xs).inertia_, 1)
                 for rs in range(50)]

print("50 lan chay, random_state tu 0 toi 49, n_init=1")
print("  init        dat 23.0   te hon   inertia xau nhat")
for init, ds in ket.items():
    dat = ds.count(23.0)
    print(f"  {init:10s}  {dat:4d}/50   {50 - dat:4d}/50   {max(ds):8.1f}")
print()
for init, ds in ket.items():
    dem = sorted(Counter(ds).items())
    print(f"Phan bo inertia, {init}:")
    print("  " + ", ".join(f"{v} x{n}" for v, n in dem))
