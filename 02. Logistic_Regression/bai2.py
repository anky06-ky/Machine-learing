# -*- coding: utf-8 -*-
# Khai báo file Python sử dụng mã hóa UTF-8 để hiển thị tiếng Việt có dấu.

"""Bài 2: vẽ sigmoid trên [-8, 8] và lưu sigmoid.png."""
# Mô tả ngắn mục đích của chương trình.


print("BAI 2: Dang ve do thi sigmoid...", flush=True)
# In thông báo ra màn hình.
# flush=True giúp nội dung được in ra ngay lập tức.


import numpy as np
from pathlib import Path
# Import thư viện NumPy.
# NumPy dùng để tạo mảng số và thực hiện các phép tính toán học.

import matplotlib.pyplot as plt
# Import pyplot của matplotlib.
# Dùng để vẽ biểu đồ.


def sigmoid(z):
    # Tạo hàm sigmoid nhận đầu vào là z.

    return 1 / (1 + np.exp(-z))
    # Công thức sigmoid:
    # sigmoid(z) = 1 / (1 + e^(-z))
    # np.exp(-z) chính là e mũ -z.


# Lấy 400 giá trị cách đều từ -8 đến 8.
z = np.linspace(-8, 8, 400)
# np.linspace(-8, 8, 400):
# tạo ra 400 giá trị cách đều nhau từ -8 đến 8.


xac_suat = sigmoid(z)
# Đưa toàn bộ 400 giá trị z vào hàm sigmoid.
# Kết quả là 400 giá trị xác suất tương ứng từ gần 0 đến gần 1.


plt.figure(figsize=(8, 5))
# Tạo khung hình để vẽ biểu đồ.
# figsize=(8, 5) nghĩa là hình rộng 8 inch, cao 5 inch.


plt.plot(
    z,
    xac_suat,
    color="#2563eb",
    linewidth=2.5,
    label="Sigmoid"
)
# Vẽ đường sigmoid.
# z          : trục X.
# xac_suat   : trục Y.
# color      : màu xanh.
# linewidth  : độ dày của đường.
# label      : tên của đường dùng trong chú thích.


plt.axhline(
    0.5,
    color="#dc2626",
    linestyle="--",
    label="y = 0.5"
)
# Vẽ một đường ngang tại y = 0.5.
# Đây là mức xác suất 50%.
# linestyle="--" nghĩa là đường nét đứt.


plt.axvline(
    0,
    color="#15803d",
    linestyle="--",
    label="z = 0"
)
# Vẽ một đường thẳng đứng tại z = 0.
# Tại z = 0 thì sigmoid(0) = 0.5.


plt.scatter(
    [0],
    [0.5],
    color="#111827",
    zorder=5
)
# Vẽ một điểm tại tọa độ (0, 0.5).
# Đây là điểm chính giữa của đường sigmoid.
# zorder=5 giúp điểm này được vẽ nổi lên trên các đường khác.


plt.title("Hàm sigmoid: σ(z) = 1 / (1 + exp(-z))")
# Đặt tiêu đề cho biểu đồ.


plt.xlabel("z")
# Đặt tên cho trục ngang X là z.


plt.ylabel("σ(z)")
# Đặt tên cho trục dọc Y là sigmoid(z).


plt.xlim(-8, 8)
# Giới hạn trục X từ -8 đến 8.


plt.ylim(-0.05, 1.05)
# Giới hạn trục Y từ -0.05 đến 1.05.
# Cho thêm một khoảng nhỏ để biểu đồ dễ nhìn.


plt.grid(alpha=0.25)
# Bật đường lưới trên biểu đồ.
# alpha=0.25 làm đường lưới mờ hơn.


plt.legend()
# Hiển thị phần chú thích của các đường:
# Sigmoid, y = 0.5 và z = 0.


plt.tight_layout()
# Tự động căn chỉnh khoảng cách giữa các thành phần.
# Tránh tiêu đề hoặc nhãn bị cắt.


# Lưu hình cạnh bộ mã bài tập để vị trí đầu ra không phụ thuộc thư mục chạy.
output_path = Path(__file__).resolve().parent / "sigmoid.png"
plt.savefig(output_path, dpi=200)
# Lưu biểu đồ thành file sigmoid.png.
# dpi=200 làm cho ảnh có độ phân giải khá rõ.


plt.close()
# Đóng biểu đồ sau khi lưu.
# Giúp giải phóng bộ nhớ.


print(f"Da luu do thi vao {output_path}.")
# Thông báo rằng ảnh đã được lưu thành công.


print(f"sigmoid(0) = {sigmoid(0):.4f}")
# Tính sigmoid tại z = 0.
# :.4f nghĩa là hiển thị 4 chữ số sau dấu phẩy.
# Kết quả phải là:
# sigmoid(0) = 0.5000
