"""Bài 1: tính a, b, silhouette từng điểm chỉ bằng NumPy."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import read_data, plt, savefig
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, silhouette_samples


def silhouette_mot_diem(X, nhan, i):
    X, nhan = np.asarray(X), np.asarray(nhan)
    groups = np.unique(nhan)
    if not 2 <= len(groups) < len(X):
        raise ValueError("Silhouette can tu 2 den n-1 cum")
    distances = np.linalg.norm(X-X[i], axis=1)
    same = nhan == nhan[i]
    same[i] = False  # Không tính khoảng cách tới chính điểm đang xét.
    b = min(distances[nhan == g].mean() for g in groups if g != nhan[i])
    if not same.any():
        return 0.0, float(b), 0.0  # Quy ước cụm một điểm có s=0.
    a = distances[same].mean()
    s = (b-a)/max(a, b) if max(a, b) > 0 else 0.0
    return float(a), float(b), float(s)


if __name__ == "__main__":
    df = read_data("khach_hang.csv")
    X = StandardScaler().fit_transform(df[["thu_nhap", "diem_chi_tieu"]])
    labels = KMeans(n_clusters=5, n_init=10, random_state=0).fit_predict(X)
    values = np.array([silhouette_mot_diem(X, labels, i) for i in range(len(X))])
    library = silhouette_samples(X, labels)
    np.testing.assert_allclose(values[:, 2], library, atol=1e-12)
    print(f"Tu tinh trung binh={values[:, 2].mean():.10f}; sklearn={silhouette_score(X, labels):.10f}")
    i = int(np.argmin(values[:, 2]))
    print(f"Khach co s nho nhat: index={i} (dong CSV={i+2}), cum={labels[i]}")
    print(df.iloc[i].to_string())
    print("a, b, s:", values[i])
    plt.scatter(df.thu_nhap, df.diem_chi_tieu, c=labels, cmap="tab10", s=24)
    plt.scatter([df.iloc[i].thu_nhap], [df.iloc[i].diem_chi_tieu], s=200, marker="*", color="red", label=f"Khach {i}")
    plt.xlabel("Thu nhap (trieu)"); plt.ylabel("Diem chi tieu"); plt.legend()
    savefig("silhouette_khach_yeu_nhat.png")
    print("Khach s thap nam gan ranh gioi, hoac xa nhom cua minh so voi nhom ben canh; xem dau sao.")
