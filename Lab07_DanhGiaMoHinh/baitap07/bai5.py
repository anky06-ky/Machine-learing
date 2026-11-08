"""Bài 5: class_weight cho rừng và SVM; so độ thay đổi của recall."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

X_tr, X_te, y_tr, y_te = split_data()
changes = {}
print("Mo hinh   class_weight  precision  recall    F1")
for name in ["Rung", "SVM"]:
    results = []
    for weight in [None, "balanced"]:
        model = (RandomForestClassifier(n_estimators=200, random_state=0, class_weight=weight)
                 if name == "Rung" else make_pipeline(StandardScaler(), SVC(class_weight=weight)))
        du = model.fit(X_tr, y_tr).predict(X_te)
        scores = [precision_score(y_te, du, zero_division=0), recall_score(y_te, du), f1_score(y_te, du)]
        results.append(scores)
        print(f"{name:8s} {str(weight):12s}", "  ".join(f"{x:.4f}" for x in scores))
    delta = [b-a for a, b in zip(*results)]
    changes[name] = abs(delta[1])
    print("Thay doi precision, recall, F1:", [round(x, 4) for x in delta])
print("Thay doi recall nhieu nhat:", max(changes, key=changes.get))
print("Cac mo hinh phan ung khac nhau; can bang lop khong bao dam F1 tang.")
