"""Mục 6: học 5.000 pixel, lượng tử hoá ảnh thành 4, 16, 64 màu."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import plt, savefig
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image

image = load_sample_image("china.jpg")
height, width, _ = image.shape
pixels = image.reshape(-1, 3) / 255.
sample = pixels[np.random.default_rng(0).choice(len(pixels), 5000, replace=False)]
print(f"Anh {height} x {width}; {len(pixels)} pixel; {len(np.unique(image.reshape(-1, 3), axis=0))} mau")
print(" K   bit/pixel   ti le ly thuyet   MAE muc sang")
fig, axes = plt.subplots(1, 4, figsize=(14, 4))
axes[0].imshow(image); axes[0].set_title("Goc: 24 bit/pixel")
for ax, k in zip(axes[1:], [4, 16, 64]):
    km = KMeans(n_clusters=k, n_init=4, random_state=0).fit(sample)
    new = km.cluster_centers_[km.predict(pixels)]
    bits = int(np.ceil(np.log2(k)))
    print(f"{k:2d} {bits:10d} {24/bits:17.1f} {np.abs(new-pixels).mean()*255:14.2f}")
    ax.imshow(new.reshape(image.shape)); ax.set_title(f"{k} mau")
for ax in axes:
    ax.axis("off")
savefig("nen_mau.png")
print("Ti le 24/bit chi la ly thuyet cho chi so mau, chua tinh bang mau va dinh dang tep.")
