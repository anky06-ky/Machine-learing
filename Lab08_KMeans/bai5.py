# -*- coding: utf-8 -*-
"""Bài 5: DBSCAN, đếm cụm không gồm -1; ARI dùng nhãn chỉ để đánh giá."""

from pathlib import Path
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "hai_vang_trang.csv")
X = df[["x1", "x2"]]
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
base = adjusted_rand_score(df.nhan, km)
print(f"KMeans: ARI={base:.4f}")
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
axes[0].scatter(X.x1, X.x2, c=km, s=15, cmap="tab10"); axes[0].set_title(f"KMeans ARI={base:.3f}")
for ax, eps in zip(axes[1:], [0.2, 0.3, 0.4]):
    labels = DBSCAN(eps=eps, min_samples=5).fit_predict(X)
    n_clusters = len(set(labels)-{-1})
    noise = int((labels == -1).sum())
    ari = adjusted_rand_score(df.nhan, labels)
    print(f"eps={eps}: {n_clusters} cum, {noise} diem nhieu, ARI={ari:.4f}, tot hon KMeans={ari>base}")
    ax.scatter(X.x1, X.x2, c=labels, s=15, cmap="tab10"); ax.set_title(f"eps={eps}, ARI={ari:.3f}")
plt.tight_layout()
plt.savefig(BASE_DIR / "dbscan_hai_vang_trang.png", dpi=150, bbox_inches="tight")
plt.close()
print("Da luu: dbscan_hai_vang_trang.png")
print("eps nho: de tach vun va co nhieu; eps lon: co the noi hai vang trang thanh mot cum.")
