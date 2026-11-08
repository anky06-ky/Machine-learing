# Nguồn dữ liệu

Theo yêu cầu, đã kiểm tra [Bài06 - SVM.ipynb](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb) ở commit `4461e7ff473f7d34a6a2f2ec2473f808012a4972`.

Bài06 dùng `make_blobs(n_samples=80, centers=2, cluster_std=1.0/1.8, random_state=42)`, `make_moons(n_samples=200, noise=0.2, random_state=42)` và `load_iris()`. Notebook không chứa `sinh_vien_bo_hoc.csv`, `khach_hang.csv`, `xor.csv` và không có đủ thông tin để khôi phục đúng ba file này từ các con số in trong PDF.

- `sinh_vien_bo_hoc.csv`: dữ liệu mô phỏng thay thế, seed 7; công thức và cách tạo nhãn công khai trong `../tao_du_lieu.py`. Không phải dữ liệu tải từ Bài06.
- `gia_nha.csv`: nguyên bản từ `01_Linear_Regression/data/gia_nha.csv`, đồng nhất với file người dùng thêm ở thư mục gốc.
- Bộ chữ số cho bài tập 2: `sklearn.datasets.load_digits`, đúng PDF; dữ liệu có sẵn trong thư viện.

Thay file sinh viên bằng CSV gốc khi có rồi chạy lại toàn bộ lab. Không chuyển Iris thành dữ liệu sinh viên và không đổi tên cột Iris để giả làm dữ liệu của đề.
