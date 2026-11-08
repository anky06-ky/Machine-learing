"""Bài 3: 50 seed cho mỗi cách khởi tạo, luôn n_init=1."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import read_data, plt, savefig
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(read_data("khach_hang.csv")[["thu_nhap", "diem_chi_tieu"]])
results = {}
for init in ["random", "k-means++"]:
    results[init] = np.array([KMeans(n_clusters=5, init=init, n_init=1, random_state=seed).fit(X).inertia_ for seed in range(50)])
best_observed = min(v.min() for v in results.values())
print(f"Inertia tot nhat quan sat duoc: {best_observed:.6f} (khong khang dinh toi uu toan cuc)")
for init, values in results.items():
    pdf_hits = int(np.sum(np.round(values, 1) == 23.0))
    hits = int(np.isclose(values, best_observed, atol=1e-6, rtol=0).sum())
    print(f"{init:10s}: dat 23.0 (lam tron 1 so) {pdf_hits}/50; dat muc tot nhat {hits}/50; "
          f"min={values.min():.4f}, max={values.max():.4f}, mean={values.mean():.4f}")
    plt.plot(range(50), values, ".", label=init)
winner = max(results, key=lambda name: np.isclose(results[name], best_observed, atol=1e-6, rtol=0).sum())
print("On dinh hon theo tan suat dat muc tot nhat:", winner)
print("K-means++ uu tien tam xa nhau; van nen dung n_init>1 vi khong bao dam toi uu.")
plt.xlabel("random_state"); plt.ylabel("Inertia"); plt.legend()
savefig("so_sanh_khoi_tao.png")
