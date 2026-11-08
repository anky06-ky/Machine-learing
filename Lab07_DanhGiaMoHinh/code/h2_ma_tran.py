"""Mục 3: tự đếm TP, FP, FN, TN rồi đối chiếu thư viện."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import split_data, savefig, plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_tr, X_te, y_tr, y_te = split_data()
du = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr).predict(X_te)
that = y_te.to_numpy()
tp = np.sum((that == 1) & (du == 1))
fp = np.sum((that == 0) & (du == 1))
fn = np.sum((that == 1) & (du == 0))
tn = np.sum((that == 0) & (du == 0))
matrix = confusion_matrix(that, du, labels=[0, 1])
np.testing.assert_array_equal(matrix, [[tn, fp], [fn, tp]])
print(f"Tu dem: TP={tp}, FP={fp}, FN={fn}, TN={tn}; tong={len(that)}")
print("confusion_matrix:\n", matrix)
print("Thu tu TN, FP, FN, TP:", matrix.ravel())
print(f"Accuracy: {(tp+tn)/len(that):.4f}; so ban bi bao: {tp+fp}")
ConfusionMatrixDisplay(matrix, display_labels=["Hoc tiep", "Bo hoc"]).plot(cmap="Blues")
savefig("ma_tran_nham_lan.png")
