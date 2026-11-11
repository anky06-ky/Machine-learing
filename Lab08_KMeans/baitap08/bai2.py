# -*- coding: utf-8 -*-
"""Bài tập 2: phân cụm trên cả ba cột tuổi, thu nhập, điểm chi tiêu."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

df = pd.read_csv(DATA_DIR / "khach_hang.csv")
cot3 = ["tuoi", "thu_nhap", "diem_chi_tieu"]
sc3 = StandardScaler().fit(df[cot3])
X3 = sc3.transform(df[cot3])

print("Ba cot, co chuan hoa")
print("   K  silhouette")
diem = {}
for k in range(2, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X3)
    diem[k] = silhouette_score(X3, km.labels_)
    print(f"  {k:2d}  {diem[k]:.3f}")
k_tot = max(diem, key=diem.get)
print(f"K tot nhat theo silhouette: {k_tot}"
      f" ({'van la 5' if k_tot == 5 else 'khong con la 5'})")
print()


def bang_tam(sc, km, cot):
    """Tâm cụm đổi về đơn vị gốc, kèm số khách, xếp theo thu nhập."""
    tam = pd.DataFrame(sc.inverse_transform(km.cluster_centers_), columns=cot)
    tam["so_khach"] = np.bincount(km.labels_)
    return tam.sort_values("thu_nhap").round(1)


km3 = KMeans(n_clusters=k_tot, n_init=10, random_state=0).fit(X3)
print(f"Tam cum ba cot voi K = {k_tot}, don vi goc, xep theo thu nhap:")
print(bang_tam(sc3, km3, cot3).to_string())
print()

# Mục 3: chỉ hai cột thu nhập và điểm chi tiêu, tuổi chỉ để mô tả
cot2 = ["thu_nhap", "diem_chi_tieu"]
sc2 = StandardScaler().fit(df[cot2])
km2 = KMeans(n_clusters=5, n_init=10, random_state=0).fit(sc2.transform(df[cot2]))
tam2 = bang_tam(sc2, km2, cot2)
tam2.insert(0, "tuoi_tb", df.groupby(km2.labels_)["tuoi"].mean().round(1))
print("Tam cum muc 3 (hai cot), tuoi trung binh tinh sau:")
print(tam2[["tuoi_tb", "thu_nhap", "diem_chi_tieu", "so_khach"]].to_string())
print()

# Hai cách chia có gom cùng những khách vào cùng nhóm không
bang = pd.crosstab(pd.Series(km2.labels_, name="cum_2_cot"),
                   pd.Series(km3.labels_, name="cum_3_cot"))
print("Bang cheo cum hai cot (hang) va cum ba cot (cot):")
print(bang.to_string())
