# -*- coding: utf-8 -*-
"""Bài 4: lưu PNG 16 màu, so dung lượng; --anh nhận ảnh cá nhân RGB/RGBA."""

from pathlib import Path
import argparse
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def compress_image(image, prefix):
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError("Anh can dang RGB, RGBA hoac xam")
    image = image[..., :3]  # Bỏ alpha trước khi phân cụm như đề yêu cầu.
    if np.issubdtype(image.dtype, np.integer):
        image = image.astype(float)/np.iinfo(image.dtype).max
    image = np.clip(image, 0, 1)
    pixels = image.reshape(-1, 3)
    n = min(5000, len(pixels))
    sample = pixels[np.random.default_rng(0).choice(len(pixels), n, replace=False)]
    km = KMeans(n_clusters=16, n_init=4, random_state=0).fit(sample)
    new = km.cluster_centers_[km.predict(pixels)].reshape(image.shape)
    original_path = BASE_DIR / (prefix + "anh_goc.png")
    compressed_path = BASE_DIR / (prefix + "anh_16_mau.png")
    plt.imsave(original_path, image)
    plt.imsave(compressed_path, new)
    print(f"{prefix or 'china'}: shape={new.shape}; mau sau luu={len(np.unique(np.rint(new*255).astype(np.uint8).reshape(-1, 3), axis=0))}")
    print(f"PNG goc={original_path.stat().st_size} byte; PNG 16 mau={compressed_path.stat().st_size} byte; "
          f"ti le={original_path.stat().st_size/compressed_path.stat().st_size:.2f} lan")
    print("Da luu:", original_path.name, compressed_path.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anh", type=Path, help="Duong dan anh ca nhan")
    args = parser.parse_args()
    compress_image(load_sample_image("china.jpg"), "")
    if args.anh:
        print("Anh ca nhan:", args.anh)
        compress_image(plt.imread(args.anh), "ca_nhan_")
    else:
        print("CHUA CO ANH CA NHAN: dang minh hoa lan hai bang flower.jpg; chay --anh DUONG_DAN de hoan tat.")
        demo = BASE_DIR / "flower_mau.png"
        plt.imsave(demo, load_sample_image("flower.jpg"))
        compress_image(plt.imread(demo), "minh_hoa_")
    print("Dung luong PNG con phu thuoc nen va bang mau; khong bang ti le ly thuyet 6 lan.")
