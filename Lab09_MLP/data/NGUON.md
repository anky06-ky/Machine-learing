# Nguồn dữ liệu

Nguồn tham chiếu: [Bài06 - SVM.ipynb](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb), commit `4461e7ff473f7d34a6a2f2ec2473f808012a4972`.

- `hai_vang_trang.csv`: tái tạo đúng `make_moons(n_samples=200, noise=0.2, random_state=42)` trong Bài06, cùng nội dung với Lab08. Có script tái lập `../lay_du_lieu_bai06.py`.
- `xor.csv`: mô phỏng thay thế 400 điểm, seed 9, script `../tao_du_lieu.py`; không có file này trong Bài06.
- `sinh_vien_mo_rong.csv`: nguyên dữ liệu đã có ở `Lab04_KNN/data/`, gồm giờ ôn, điểm giữa kỳ, chi tiêu, nhãn qua môn.
- `gia_nha.csv`: nguyên dữ liệu Lab01, đồng nhất file người dùng bổ sung ở thư mục gốc.
- `load_digits`: dữ liệu chữ số có sẵn trong scikit-learn, dùng theo PDF.

Không đổi tên Iris hay make_blobs thành XOR: chúng là các bài toán khác nhau. Khi có `xor.csv` gốc, thay file rồi chạy lại lab; không ép accuracy bằng các con số in trong PDF.
