# -*- coding: utf-8 -*-
"""Bài tập 4: ảnh hưởng của tốc độ học learning_rate_init."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

THU_MUC = Path(__file__).resolve().parent

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, random_state=17, stratify=y
)

print("  learning_rate_init  so vong  mat mat cuoi  tap kiem tra")
plt.figure(figsize=(8, 5))
for toc_do in [0.0001, 0.001, 0.01, 0.1]:
    mang = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000,
                      learning_rate_init=toc_do, random_state=0),
    ).fit(X_tr, y_tr)
    mlp = mang[-1]
    acc = accuracy_score(y_te, mang.predict(X_te))
    print(f"  {toc_do:<18}  {mlp.n_iter_:7d}  {mlp.loss_curve_[-1]:12.4f}"
          f"  {acc:.4f}")
    plt.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_,
             label=f"learning_rate_init = {toc_do}")

plt.yscale("log")
plt.xlabel("Vong hoc")
plt.ylabel("Mat mat tren tap hoc (thang log)")
plt.title("Duong mat mat voi bon toc do hoc")
plt.legend()
plt.tight_layout()
plt.savefig(THU_MUC / "toc_do_hoc.png", dpi=150)
plt.close()
print("Da luu hinh:", (THU_MUC / "toc_do_hoc.png").relative_to(THU_MUC.parent))
