# Nguồn dữ liệu

Nguồn người dùng chỉ định: [Bài06 - SVM.ipynb](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb), commit `4461e7ff473f7d34a6a2f2ec2473f808012a4972`.

- `hai_vang_trang.csv`: **tái tạo đúng dữ liệu Bài06** bằng `make_moons(n_samples=200, noise=0.2, random_state=42)`. Notebook tạo mảng trong bộ nhớ, không chứa file CSV; lưu nguyên hai toạ độ và nhãn thành `x1,x2,nhan`. Script tái lập: `../lay_du_lieu_bai06.py`. Nhãn không được dùng khi phân cụm.
- `khach_hang.csv`: **mô phỏng thay thế**, seed 8, 210 khách, do `../tao_du_lieu.py` tạo. File này không có trong Bài06; chưa có dữ liệu gốc của PDF.
- `china.jpg`, `flower.jpg`, `load_digits`: các mẫu tích hợp scikit-learn dùng theo PDF; không phải dữ liệu lấy từ notebook Bài06 và không phải ảnh cá nhân của sinh viên.

Vì bộ hai vầng trăng Bài06 khác bộ Lab04 và vì file khách hàng là bản thay thế, kết quả có thể khác các số mẫu trong PDF. Mọi nhận xét và log phải dựa trên dữ liệu thực sự chạy.
