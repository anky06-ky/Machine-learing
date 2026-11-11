# -*- coding: utf-8 -*-
"""Bài tập 5: DBSCAN trên hai vầng trăng, so với K-Means."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import adjusted_rand_score

THU_MUC = Path(__file__).resolve().parent
DATA_DIR = THU_MUC.parent / "data"

tr = pd.read_csv(DATA_DIR / "hai_vang_trang.csv")
X = tr[["x1", "x2"]]
y = tr["nhan"].to_numpy()


def ty_le_khop(nhan):
    """Tỷ lệ khớp nhãn thật khi có đúng hai cụm; điểm nhiễu tính là sai."""
    cum = sorted(set(nhan) - {-1})
    if len(cum) != 2:
        return None
    khop = np.mean(np.where(nhan == cum[0], 0, np.where(nhan == cum[1], 1, -1)) == y)
    nguoc = np.mean(np.where(nhan == cum[0], 1, np.where(nhan == cum[1], 0, -1)) == y)
    return max(khop, nguoc)


ket_qua = {}
km = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(X)
ket_qua["K-Means, K = 2"] = km
for e in [0.2, 0.3, 0.4]:
    ket_qua[f"DBSCAN, eps = {e}"] = DBSCAN(eps=e, min_samples=5).fit_predict(X)

print(f"Hai vang trang: {len(X)} diem")
print("  Cach chia          so cum  diem nhieu  ty le khop     ARI")
for ten, nhan in ket_qua.items():
    so_cum = len(set(nhan) - {-1})
    nhieu = int((nhan == -1).sum())
    khop = ty_le_khop(nhan)
    khop_s = f"{khop:.4f}" if khop is not None else "   -  "
    print(f"  {ten:17s}  {so_cum:6d}  {nhieu:10d}  {khop_s:>10s}"
          f"  {adjusted_rand_score(y, nhan):6.4f}")
print("(ty le khop chi tinh khi dung hai cum; ARI = 1 la trung khop nhan that)")
print()

# Thử thêm eps nhỏ hơn đề bài để thấy rõ vai trò của eps
print("Thu them eps nho hon, min_samples=5:")
for e in [0.1, 0.15]:
    nhan = DBSCAN(eps=e, min_samples=5).fit_predict(X)
    print(f"  eps = {e:4}: {len(set(nhan) - {-1}):2d} cum,"
          f" {int((nhan == -1).sum()):3d} diem nhieu")

fig, truc = plt.subplots(1, 4, figsize=(16, 4))
for ax, (ten, nhan) in zip(truc, ket_qua.items()):
    ax.scatter(X.x1, X.x2, c=nhan, cmap="tab10", s=12)
    if (nhan == -1).any():
        ax.scatter(X.x1[nhan == -1], X.x2[nhan == -1], c="black", marker="x",
                   s=25, label="nhieu")
        ax.legend(loc="lower left")
    ax.set_title(ten)
plt.tight_layout()
plt.savefig(THU_MUC / "dbscan_hai_vang_trang.png", dpi=120)
plt.close()
print("Da luu hinh:", (THU_MUC / "dbscan_hai_vang_trang.png")
      .relative_to(THU_MUC.parent))
