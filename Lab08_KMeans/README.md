# Lab08: Phân cụm K-Means

**Sinh viên:** Trần An Kỳ - **MSSV:** 2474802010460. Môn Học máy ứng dụng, LHP 261_71ITAI41203_0101, năm 3 khóa 30.

Triển khai đủ 5 chương trình mẫu và 6 bài tập trong `Lab08_KMeans.pdf`.

## Dữ liệu

- `data/khach_hang.csv`: **bản mô phỏng thay thế**, 210 khách, seed 8, tạo bởi `tao_du_lieu.py`. Có ba cột `tuoi`, `thu_nhap` (triệu/tháng), `diem_chi_tieu` (1-100), không có nhãn. Chưa có file gốc; kết quả không được coi là tái lập số liệu mẫu của PDF.
- `data/hai_vang_trang.csv`: tái tạo đúng dữ liệu [Bài06 trên repo giảng viên](https://github.com/AnhNguyenVLU/Machine_Learning_VLU/blob/4461e7ff473f7d34a6a2f2ec2473f808012a4972/06.%20SVM/SVM.ipynb), `make_moons(n_samples=200, noise=0.2, random_state=42)`. Script tái lập: `lay_du_lieu_bai06.py`. Chi tiết nguồn: [data/NGUON.md](./data/NGUON.md).
- Ảnh `china.jpg`, `flower.jpg`: ảnh mẫu đi kèm scikit-learn, chạy ngoại tuyến; không phải ảnh cá nhân của sinh viên.
- `load_digits`: bộ chữ số có sẵn trong scikit-learn.

## Cài đặt và chạy

Từ thư mục gốc repository, PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\Lab08_KMeans\requirements.txt
.\.venv\Scripts\python.exe .\Lab08_KMeans\chay_tat_ca.py
```

Nếu venv đã có thì bỏ lệnh tạo. Đã chạy với Python 3.14.8; thư viện cố định trong `requirements.txt`. Không đưa venv lên Git. Các lab độc lập, mỗi thư mục giữ đủ dữ liệu cần thiết.

## Năm chương trình thực hành

1. `code/k1_tu_viet.py`: bảy điểm A-G, tâm ban đầu A/B, tự gán và dời tâm, inertia đối chiếu sklearn. Giữ tâm cũ nếu cụm rỗng; tính lại nhãn theo tâm cuối để tránh trả nhãn cũ khi hết số vòng.
2. `code/k2_khach_hang.py`: chuẩn hoá thu nhập/chi tiêu, K=5, `n_init=10`, seed 0; đổi tâm về đơn vị gốc, đếm khách, tuổi trung bình và dự đoán hai khách mới bằng đúng scaler đã học.
3. `code/k3_chon_k.py`: K=1-10, inertia và mức giảm, silhouette khi K>=2; lưu `figures/chon_k.png`.
4. `code/k4_ba_cai_bay.py`: thu nhập nhân 1.000 khi chưa chuẩn hoá, 20 seed khởi tạo random một lần, K-Means trên hai vầng trăng.
5. `code/k5_nen_anh.py`: ảnh china, học 5.000 pixel ngẫu nhiên, 4/16/64 màu, số bit chỉ số màu và sai lệch RGB; lưu hình đối chiếu.

## Sáu bài tập

1. `baitap08/bai1.py`: `silhouette_mot_diem(X, nhan, i)` trả a, b, s bằng NumPy; bỏ chính điểm i khỏi trung bình trong cụm. Cụm một điểm có s=0. So cả từng điểm và trung bình với sklearn; in khách s thấp nhất và đánh dấu trên hình. [Định nghĩa silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_samples.html).
2. `baitap08/bai2.py`: chuẩn hoá cả tuổi/thu nhập/chi tiêu, chọn K=2-10 theo silhouette, in tâm 3 chiều và tâm gốc 2 chiều theo cùng đơn vị để so sánh.
3. `baitap08/bai3.py`: 50 seed cho `random` và `k-means++`, `n_init=1`. Đếm số lần inertia làm tròn thành 23.0 theo đề; đồng thời đếm số lần đạt mức tốt nhất quan sát được trên dữ liệu hiện có. Không gọi mức tốt nhất quan sát là tối ưu toàn cục.
4. `baitap08/bai4.py`: lưu `figures/anh_goc.png`, `figures/anh_16_mau.png`, in dung lượng byte. Đọc ảnh khác bằng `plt.imread`, bỏ alpha nếu RGBA. Mặc định minh hoạ thêm bằng flower; **phần ảnh cá nhân cần truyền ảnh của bạn bằng `--anh`**.
5. `baitap08/bai5.py`: DBSCAN eps=0.2/0.3/0.4, min_samples=5, đếm cụm không gồm -1 và điểm nhiễu; so ARI với K-Means, lưu hình. Nhãn thật chỉ dùng để đánh giá, không dùng khi học.
6. `baitap08/bai6.py`: K-Means K=10 trên chữ số, sau khi fit mới lập crosstab nhãn/cụm; in tỷ lệ mỗi chữ số vào cụm chính và hai chữ số đông nhất trong từng cụm.

## Chạy riêng và ảnh cá nhân

```powershell
.\.venv\Scripts\python.exe .\Lab08_KMeans\code\k1_tu_viet.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai1.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai4.py --anh "D:\Anh\anh_cua_toi.png"
```

Đường dẫn ảnh ở trên là ví dụ, thay bằng ảnh thật trên máy. Chưa có ảnh cá nhân thì chương trình in rõ đang dùng ảnh mẫu; không gán ảnh mẫu thành ảnh do bạn chụp.

## Kết quả và nhận xét

- `ket_qua/`: log TXT thực tế, ảnh PNG **kết xuất từ log, không phải ảnh chụp terminal**, và `tong_hop.json` ghi mã thoát lần chạy gần nhất của từng bài. `--chi` giữ kết quả các bài khác. Cần chụp terminal riêng nếu giảng viên yêu cầu đúng dạng ảnh chụp màn hình.
- `figures/`: hình phân cụm, chọn K, ba bẫy, silhouette, khởi tạo, nén ảnh và DBSCAN.
- [Kết quả chạy](./KET_QUA.md): kết quả đo, không chép các số mẫu trong đề.

Số cụm không mang ý nghĩa thứ hạng. Chuẩn hoá quyết định tương quan khoảng cách giữa các cột; K-Means++ giảm rủi ro khởi tạo nhưng vẫn cần nhiều lần thử. Silhouette và khuỷu tay giúp chọn K, không thay thế mục tiêu phân khúc. Tỷ lệ 24/4=6 chỉ mô tả chi phí lý thuyết của chỉ số 16 màu, chưa gồm bảng màu; tỷ lệ dung lượng PNG thực tế phải đo từ file.

Nếu có `khach_hang.csv` gốc, thay đúng file trong `data/` rồi chạy lại runner; không dùng lệnh tạo mô phỏng ghi đè dữ liệu gốc.
