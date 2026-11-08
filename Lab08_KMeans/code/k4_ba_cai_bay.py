"""Mục 5: thang đo, cực tiểu cục bộ và cụm không tròn."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import read_data, plt, savefig
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = read_data("khach_hang.csv")
X = df[["thu_nhap", "diem_chi_tieu"]]
Xs = StandardScaler().fit_transform(X)
X_nghin = X.copy(); X_nghin["thu_nhap"] *= 1000
raw = KMeans(n_clusters=5, n_init=10, random_state=0).fit(X_nghin)
centers = raw.cluster_centers_.copy(); centers[:, 0] /= 1000
print("Bay 1: tam khi thu nhap ghi bang nghin dong, doi lai trieu de doc:")
print(centers[np.argsort(centers[:, 0])].round(1))
inertias = [KMeans(n_clusters=5, init="random", n_init=1, random_state=i).fit(Xs).inertia_ for i in range(20)]
print("Bay 2: inertia 20 lan khoi tao:", np.round(sorted(inertias), 1))
good = KMeans(n_clusters=5, n_init=10, random_state=0).fit(Xs)
print(f"n_init=10, inertia={good.inertia_:.2f}")
moon = read_data("hai_vang_trang.csv")
labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(moon[["x1", "x2"]])
match = (labels == moon.nhan).mean()
print(f"Bay 3: ty le khop nhan (co xet hoan vi)={max(match, 1-match):.4f}")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, lab, title in zip(axes, [raw.labels_, good.labels_], ["Khong chuan hoa", "Co chuan hoa"]):
    ax.scatter(X.thu_nhap, X.diem_chi_tieu, c=lab, cmap="tab10", s=22)
    ax.set(xlabel="Thu nhap (trieu)", ylabel="Diem chi tieu", title=title)
savefig("bay_thang_do.png")
print("K-Means gan theo tam gan nhat, khong bam duoc duong cong cua hai vang trang.")
