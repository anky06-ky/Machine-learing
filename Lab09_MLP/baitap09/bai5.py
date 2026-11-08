"""Bài 5: GridSearchCV 4 kiến trúc x 3 alpha; SVM trên cùng test."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ROOT
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_digits(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=17, stratify=y)
pipeline = make_pipeline(StandardScaler(), MLPClassifier(max_iter=1000, random_state=0))
params = {"mlpclassifier__hidden_layer_sizes": [(10,), (50,), (100,), (50, 50)],
          "mlpclassifier__alpha": [0.0001, 0.01, 1.]}
search = GridSearchCV(pipeline, params, cv=5, scoring="accuracy", n_jobs=1).fit(X_tr, y_tr)
table = pd.DataFrame(search.cv_results_)[["param_mlpclassifier__hidden_layer_sizes", "param_mlpclassifier__alpha", "mean_test_score", "std_test_score", "rank_test_score"]]
table = table.rename(columns={"param_mlpclassifier__hidden_layer_sizes": "lop_an",
                              "param_mlpclassifier__alpha": "alpha", "mean_test_score": "accuracy_cv",
                              "std_test_score": "do_lech_chuan", "rank_test_score": "hang"})
print(table.sort_values("hang").to_string(index=False))
print("Tham so tot nhat:", search.best_params_)
print(f"CV accuracy={search.best_score_:.4f}")
mlp_score = search.score(X_te, y_te)
svm_score = SVC().fit(X_tr, y_tr).score(X_te, y_te)
print(f"Test: MLP={mlp_score:.4f}; SVM RBF={svm_score:.4f}; MLP vuot SVM={mlp_score>svm_score}")
result_dir = ROOT / "ket_qua"; result_dir.mkdir(exist_ok=True)
table.to_csv(result_dir / "grid_search.csv", index=False)
print("Scaler nam trong pipeline nen moi fold chi chuan hoa bang tap hoc cua fold do.")
