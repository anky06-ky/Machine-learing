"""Mục 6: ROC, AUC và sáu mô hình trên cùng một cách chia."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score, recall_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = split_data()
models = {"Logistic": make_pipeline(StandardScaler(), LogisticRegression()),
          "Naive Bayes": GaussianNB(),
          "KNN, k=15": make_pipeline(StandardScaler(), KNeighborsClassifier(15)),
          "Cay sau 4": DecisionTreeClassifier(max_depth=4, random_state=0),
          "Rung 200 cay": RandomForestClassifier(n_estimators=200, random_state=0),
          "SVM RBF": make_pipeline(StandardScaler(), SVC(probability=True, random_state=0))}
print("Mo hinh          AUC     recall     F1 (nguong 0.5)")
for ten, model in models.items():
    prob = model.fit(X_tr, y_tr).predict_proba(X_te)[:, 1]
    du = prob >= 0.5
    print(f"{ten:15s} {roc_auc_score(y_te, prob):.4f}   {recall_score(y_te, du):.4f}   {f1_score(y_te, du, zero_division=0):.4f}")
    if ten == "Logistic":
        fpr, tpr, thresholds = roc_curve(y_te, prob)
        print(f"ROC logistic co {len(fpr)} diem; 6 diem dau:", list(zip(fpr, tpr))[:6])
print("AUC do kha nang xep hang; recall va F1 con phu thuoc nguong.")
