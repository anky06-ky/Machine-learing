"""Bài 4: tự vẽ hai đường ROC và nhận xét từ kết quả thực tế."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data, plt, savefig
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = split_data()
curves, aucs = {}, {}
for name, model in [("Logistic", make_pipeline(StandardScaler(), LogisticRegression())),
                    ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=0))]:
    prob = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, prob)
    curves[name] = (fpr, tpr)
    aucs[name] = roc_auc_score(y_te, prob)
    plt.plot(fpr, tpr, label=f"{name}: AUC={aucs[name]:.4f}")
    print(f"{name}: AUC={aucs[name]:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Ngau nhien")
plt.xlabel("FPR"); plt.ylabel("TPR / Recall"); plt.legend()
savefig("roc.png")
# So hai đường ROC dạng bậc thang tại các FPR chung, xử lý cả đoạn đứng.
grid = np.unique(np.concatenate([v[0] for v in curves.values()]))
heights = [np.array([tpr[fpr <= x].max() for x in grid]) for fpr, tpr in curves.values()]
diff = heights[0] - heights[1]
print("AUC lon hon (tong the):", max(aucs, key=aucs.get))
print("Hai duong doi thu tu cao/thap:", bool(np.any(diff > 1e-12) and np.any(diff < -1e-12)))
print("AUC lon hon khong dong nghia tot hon tai moi nguong; xem cac doan giao tren hinh.")
