"""Mục 2: thuật toán Lloyd trên bảy điểm A-G, kiểm tra với sklearn."""
import numpy as np
from sklearn.cluster import KMeans

P = np.array([[1., 1.], [1.5, 2.], [3., 4.], [5., 7.],
              [3.5, 5.], [4.5, 5.], [3.5, 4.5]])
TEN = list("ABCDEFG")


def k_means(X, tam, so_vong_toi_da=10):
    """Giữ nguyên tâm của cụm rỗng, luôn tính lại nhãn theo tâm trả về."""
    X, tam = np.asarray(X, dtype=float), np.array(tam, dtype=float, copy=True)
    for vong in range(1, so_vong_toi_da + 1):
        kc2 = ((X[:, None, :] - tam[None, :, :]) ** 2).sum(axis=2)
        nhan = kc2.argmin(axis=1)
        print(f"Vong {vong}: tam = {np.round(tam, 2).tolist()}")
        for k in range(len(tam)):
            names = [TEN[i] if len(X) == len(TEN) else str(i) for i in np.flatnonzero(nhan == k)]
            print(f"  cum {k+1}: {', '.join(names)}")
        tam_moi = tam.copy()
        for k in range(len(tam)):
            if np.any(nhan == k):
                tam_moi[k] = X[nhan == k].mean(axis=0)
        if np.allclose(tam_moi, tam):
            print("  Tam khong doi nua, dung.")
            tam = tam_moi
            break
        tam = tam_moi
    nhan = ((X[:, None, :] - tam[None, :, :]) ** 2).sum(axis=2).argmin(axis=1)
    return tam, nhan


if __name__ == "__main__":
    tam, nhan = k_means(P, P[[0, 1]])
    inertia = ((P-tam[nhan])**2).sum()
    km = KMeans(n_clusters=2, init=P[[0, 1]], n_init=1, random_state=0).fit(P)
    print(f"Inertia tu tinh: {inertia:.4f}; scikit-learn: {km.inertia_:.4f}")
    print("Tam scikit-learn:", km.cluster_centers_)
    np.testing.assert_allclose(inertia, km.inertia_)
    np.testing.assert_allclose(tam, km.cluster_centers_)
