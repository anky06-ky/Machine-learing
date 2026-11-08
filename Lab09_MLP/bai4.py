# -*- coding: utf-8 -*-
"""Bài 4: bốn tốc độ học Adam trên chữ số, giữ nguyên mọi tham số khác."""

from pathlib import Path
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
print("learning_rate   so vong    loss cuoi    test accuracy")
for rate in [0.0001, 0.001, 0.01, 0.1]:
    model = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                           learning_rate_init=rate, random_state=0)).fit(X_tr, y_tr)
    mlp = model[-1]
    print(f"{rate:13g} {mlp.n_iter_:9d} {mlp.loss_curve_[-1]:12.6f} {model.score(X_te, y_te):16.4f}", flush=True)
    plt.plot(range(1, len(mlp.loss_curve_)+1), mlp.loss_curve_, label=str(rate))
plt.xlabel("Vong hoc"); plt.ylabel("Training loss"); plt.legend(title="learning_rate_init")
plt.tight_layout()
plt.savefig(BASE_DIR / "toc_do_hoc.png", dpi=150, bbox_inches="tight")
plt.close()
print("Da luu: toc_do_hoc.png")
print("Toc do qua nho: giam loss cham, co the het max_iter. Qua lon: de dao dong/nhay qua vung tot.")
print("Doi chieu duong loss va accuracy thuc te; toc do lon khong nhat thiet that bai o moi seed.")
