"""Tạo dữ liệu THAY THẾ để chạy thử, không phải CSV gốc của giảng viên."""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def create_data():
    rng = np.random.default_rng(7)
    n = 1200
    diem = np.round(np.clip(rng.normal(6.5, 1.6, n), 0, 10), 1)
    vang = rng.poisson(4, n)
    no = rng.poisson(2, n)
    lam = np.clip(rng.poisson(12, n), 0, 40)
    risk = -1.4*(diem-6.5) + 0.55*(vang-4) + 0.7*(no-2) + 0.09*(lam-12)
    # Cố định 119 ca để minh hoạ độ lệch lớp; không tối ưu để khớp kết quả PDF.
    noisy_risk = risk + rng.normal(0, 1.8, n)
    labels = np.zeros(n, dtype=int)
    labels[np.argsort(noisy_risk)[-119:]] = 1
    return pd.DataFrame({"diem_tb": diem, "so_buoi_vang": vang, "tin_chi_no": no,
                         "gio_lam_them": lam, "bo_hoc": labels})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghi-de", action="store_true", help="Cho phep thay CSV hien co")
    args = parser.parse_args()
    path = Path(__file__).resolve().parent / "data" / "sinh_vien_bo_hoc.csv"
    path.parent.mkdir(exist_ok=True)
    if path.exists() and not args.ghi_de:
        raise SystemExit("CSV da ton tai; dung --ghi-de neu muon tao lai du lieu mo phong.")
    create_data().to_csv(path, index=False)
    print("DA TAO DU LIEU MO PHONG THAY THE, seed=7:", path)
