"""Tái tạo đúng make_moons trong Bài06 của repo giảng viên.

Nguồn: AnhNguyenVLU/Machine_Learning_VLU, 06. SVM/SVM.ipynb,
commit 4461e7ff473f7d34a6a2f2ec2473f808012a4972.
Notebook tạo dữ liệu bằng sklearn, không cung cấp CSV để tải trực tiếp.
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.datasets import make_moons


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghi-de", action="store_true", help="Cho phep cap nhat hai_vang_trang.csv")
    args = parser.parse_args()
    target = Path(__file__).resolve().parent/"data/hai_vang_trang.csv"
    if target.exists() and not args.ghi_de:
        raise SystemExit("CSV da ton tai. Dung --ghi-de de tai tao dung du lieu Bai06.")
    target.parent.mkdir(exist_ok=True)
    X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
    pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "nhan": y}).to_csv(target, index=False)
    print("Du lieu theo Bai06: 200 mau, noise=0.2, random_state=42 ->", target)
