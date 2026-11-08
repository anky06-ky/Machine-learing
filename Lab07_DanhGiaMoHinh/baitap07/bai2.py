"""Bài 2: đọc báo cáo SVM cho 10 chữ số."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
du = SVC().fit(X_tr, y_tr).predict(X_te)
print(classification_report(y_te, du, digits=4, zero_division=0))
report = classification_report(y_te, du, output_dict=True, zero_division=0)
for metric in ["recall", "precision"]:
    lowest = min(report[str(d)][metric] for d in range(10))
    digits = [d for d in range(10) if np.isclose(report[str(d)][metric], lowest)]
    print(f"{metric} thap nhat: chu so {digits}, gia tri {lowest:.4f}")
print("Support cac chu so gan bang nhau, nen macro va weighted gan bang nhau.")
