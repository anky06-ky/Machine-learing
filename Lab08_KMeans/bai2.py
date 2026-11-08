# -*- coding: utf-8 -*-
"""Bài 2: thêm tuổi, dò K bằng silhouette và đối chiếu tâm hai chiều."""

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "khach_hang.csv")
scaler = StandardScaler().fit(df)
X = scaler.transform(df)
models, scores = {}, {}
for k in range(2, 11):
    models[k] = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    scores[k] = silhouette_score(X, models[k].labels_)
    print(f"K={k:2d}, silhouette={scores[k]:.4f}")
best = max(scores, key=scores.get)
print("K tot nhat:", best, "; con bang 5:", best == 5)
centers = pd.DataFrame(scaler.inverse_transform(models[best].cluster_centers_), columns=df.columns)
print("Tam ba chieu, don vi goc:\n", centers.sort_values("thu_nhap").round(2).to_string(index=False))
cols = ["thu_nhap", "diem_chi_tieu"]
sc2 = StandardScaler().fit(df[cols])
km2 = KMeans(n_clusters=5, n_init=10, random_state=0).fit(sc2.transform(df[cols]))
print("Tam hai chieu ban dau:\n", pd.DataFrame(sc2.inverse_transform(km2.cluster_centers_), columns=cols).sort_values("thu_nhap").round(2).to_string(index=False))
print("Them tuoi doi khoang cach va cach gom nhom; so sanh toa do tam, khong doi chieu bang so cum.")
