# Học máy ứng dụng

Kho bài thực hành môn **Học máy ứng dụng**, tổ chức theo từng lab với README hướng dẫn riêng.

## Thông tin sinh viên

- **Họ và tên:** Trần An Kỳ.
- **Mã số sinh viên:** 2474802010460.
- **Môn học:** Học máy ứng dụng.
- **Mã lớp học phần:** 261_71ITAI41203_0101.
- **Sinh viên:** năm 3, khóa 30.

## Thông tin giảng viên

- **Họ và tên:** Nguyễn Thái Anh.
- **Học vị:** Thạc sĩ Khoa học Dữ liệu và Trí tuệ Nhân tạo (Data Science & Artificial Intelligence).
- **Đơn vị:** Khoa Công nghệ Thông tin, Trường Đại học Văn Lang.
- **Lĩnh vực nghiên cứu:** Khoa học dữ liệu, trí tuệ nhân tạo và học máy.
- **GitHub:** [AnhNguyenVLU](https://github.com/AnhNguyenVLU).

## Các bài thực hành

- [Bài 01: Hồi quy tuyến tính](./01_Linear_Regression/).
- [Bài 02: Hồi quy logistic](./02_Logistic_Regression/).
- [Bài 03: Naive Bayes](./03_Naive_Bayes/).
- [Bài 04: K láng giềng gần nhất](./Lab04_KNN/).
- [Bài 05: Cây quyết định và rừng ngẫu nhiên](./Lab05_CayQuyetDinh/).
- [Bài 07: Đánh giá mô hình](./Lab07_DanhGiaMoHinh/).
- [Bài 08: Phân cụm K-Means](./Lab08_KMeans/).
- [Bài 09: Mạng nơ-ron nhiều lớp](./Lab09_MLP/).

## Cài đặt và chạy

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r .\Lab07_DanhGiaMoHinh\requirements.txt
python .\Lab07_DanhGiaMoHinh\bai1.py
```

Lab07–09 dùng chung NumPy, pandas, Matplotlib và scikit-learn. Nếu `.venv` đã có thì dùng lại. Đổi tên lab và số bài để chạy bài khác; hướng dẫn đầy đủ nằm trong README từng lab. Không đưa venv, cache hay đồ thị sinh tự động lên Git.

**Nguồn dữ liệu:** bộ hai vầng trăng của Lab08/09 dùng đúng cấu hình [Bài06 của giảng viên](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb): 200 mẫu, noise=0.2, seed 42. Giá nhà dùng dữ liệu Lab01. Ba CSV sinh viên bỏ học, khách hàng và XOR hiện là **dữ liệu mô phỏng thay thế**, có ghi rõ trong README; kết quả có thể khác PDF. Bài 4 Lab08 nhận ảnh cá nhân qua `--anh`.

Kho lưu trữ: [anky06-ky/Machine-learing](https://github.com/anky06-ky/Machine-learing).
