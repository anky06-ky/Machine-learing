"""Mục 6: chữ số, trọng số, loss, early stopping và ba mô hình đối chiếu."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import plt, savefig
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=0)).fit(X_tr, y_tr)
mlp = model[-1]
n_params = sum(w.size for w in mlp.coefs_) + sum(b.size for b in mlp.intercepts_)
print("Kich thuoc ma tran trong so:", [w.shape for w in mlp.coefs_])
print("Tong so tham so:", n_params)
print(f"So vong hoc: {mlp.n_iter_}; loss vong 1={mlp.loss_curve_[0]:.4f}, "
      f"vong 10={mlp.loss_curve_[9]:.4f}, cuoi={mlp.loss_curve_[-1]:.4f}")
assert n_params == 7510
du = model.predict(X_te)
wrong = np.flatnonzero(du != y_te)
print(f"MLP test accuracy={model.score(X_te, y_te):.4f}; sai {len(wrong)}/{len(y_te)}")
for i in wrong:
    print(f"anh {i:3d}: that={y_te[i]}, doan={du[i]}")
early = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                       early_stopping=True, random_state=0)).fit(X_tr, y_tr)
print(f"Dung som: {early[-1].n_iter_} vong, test={early.score(X_te, y_te):.4f}")
for name, other in [("Logistic", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                    ("KNN k=5", KNeighborsClassifier()), ("SVM RBF", SVC())]:
    print(f"{name}: {other.fit(X_tr, y_tr).score(X_te, y_te):.4f}")
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(mlp.loss_curve_); axes[0].set(xlabel="Vong hoc", ylabel="Training loss")
axes[1].plot(early[-1].validation_scores_); axes[1].set(xlabel="Vong hoc", ylabel="Validation accuracy")
savefig("chu_so_loss_dung_som.png")
if len(wrong):
    count = min(12, len(wrong))
    fig, axes = plt.subplots(1, count, figsize=(max(4, 1.5*count), 2.5), squeeze=False)
    for ax, i in zip(axes[0], wrong[:count]):
        ax.imshow(X_te[i].reshape(8, 8), cmap="gray_r"); ax.set_title(f"{y_te[i]} -> {du[i]}"); ax.axis("off")
    savefig("chu_so_doan_sai.png")
print("Dung som co the dung qua som; MLP khong luon thang KNN hay SVM tren bo du lieu nho.")
