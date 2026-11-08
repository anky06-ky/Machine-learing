"""Mục 4: khuỷu tay và silhouette, không tính silhouette khi K=1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import read_data, plt, savefig
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

Xs = StandardScaler().fit_transform(read_data("khach_hang.csv")[["thu_nhap", "diem_chi_tieu"]])
inertias, silhouettes = [], []
print(" K    inertia   giam so voi K-1   silhouette")
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(Xs)
    drop = inertias[-1] - km.inertia_ if inertias else float("nan")
    sil = silhouette_score(Xs, km.labels_) if k > 1 else float("nan")
    inertias.append(km.inertia_); silhouettes.append(sil)
    print(f"{k:2d} {km.inertia_:10.2f} {drop:16.2f} {sil:12.4f}")
best = max(range(2, 11), key=lambda k: silhouettes[k-1])
print("K co silhouette cao nhat:", best)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(range(1, 11), inertias, "o-"); axes[0].set_ylabel("Inertia")
axes[1].plot(range(2, 11), silhouettes[1:], "o-"); axes[1].set_ylabel("Silhouette")
for ax in axes:
    ax.set_xlabel("K"); ax.axvline(best, color="red", linestyle="--")
savefig("chon_k.png")
print("Inertia luon giam khi K tang; khong chon K chi vi inertia nho nhat.")
