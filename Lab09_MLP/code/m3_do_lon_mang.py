"""Mục 4: mạng nhỏ, học vẹt, điều chuẩn; chọn alpha bằng CV tập học."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def mang(lop_an, alpha):
    return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=lop_an, alpha=alpha,
                         solver="lbfgs", max_iter=20000, random_state=0))


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = split_data("hai_vang_trang.csv", "nhan")
    print("Lop an       alpha     tap hoc   tap kiem tra (minh hoa)")
    for hidden, alpha in [((2,), 1e-4), ((100, 100), 1e-5), ((100, 100), 1.0)]:
        model = mang(hidden, alpha).fit(X_tr, y_tr)
        print(f"{str(hidden):12s} {alpha:7g} {model.score(X_tr, y_tr):10.4f} {model.score(X_te, y_te):14.4f}")
    scores = {}
    for alpha in [1e-5, 1e-3, 0.1, 1., 3., 10.]:
        scores[alpha] = cross_val_score(mang((100, 100), alpha), X_tr, y_tr, cv=5).mean()
        print(f"alpha={alpha:7g}, CV={scores[alpha]:.4f}", flush=True)
    best = max(scores, key=scores.get)
    model = mang((100, 100), best).fit(X_tr, y_tr)
    print(f"Chon bang CV: alpha={best}, test={model.score(X_te, y_te):.4f}")
    print("Tap hoc cao nhung test thap la dau hieu hoc vet; khong chon alpha tu bang test minh hoa.")
