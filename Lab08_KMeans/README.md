# Bài 08: Phân cụm K-Means

Bài thực hành theo [Lab08_KMeans.pdf](./Lab08_KMeans.pdf), gồm năm chương trình hướng dẫn tương ứng mục 2-6 và sáu bài tập ở mục 7. Mỗi chương trình chạy độc lập; đường dẫn dữ liệu và hình được xác định theo vị trí file Python.

## Cài đặt

Từ thư mục gốc của repository, tạo môi trường nếu chưa có và cài thư viện:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r .\Lab08_KMeans\requirements.txt
```

Nếu đã có `.venv`, dùng lại môi trường đó. Không cần kích hoạt môi trường khi gọi trực tiếp Python như các lệnh dưới đây.

## Chạy phần hướng dẫn

Chạy lần lượt từ thư mục gốc của repository:

```powershell
.\.venv\Scripts\python.exe .\Lab08_KMeans\k1_tu_viet.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\k2_khach_hang.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\k3_chon_k.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\k4_ba_cai_bay.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\k5_nen_anh.py
```

1. **`k1_tu_viet.py`:** tự viết bước gán và bước dời tâm bằng numpy, chạy trên bảy điểm A-G với hai tâm ban đầu cố ý đặt tệ, so inertia với scikit-learn.
2. **`k2_khach_hang.py`:** phân 210 khách thành 5 cụm theo thu nhập và điểm chi tiêu, đổi tâm cụm về đơn vị gốc, xếp hai khách mới vào cụm.
3. **`k3_chon_k.py`:** in inertia, mức giảm so với K-1 và silhouette cho K từ 1 tới 10.
4. **`k4_ba_cai_bay.py`:** ba cái bẫy: quên chuẩn hoá, khởi tạo ngẫu nhiên một lần, cụm không tròn trên hai vầng trăng.
5. **`k5_nen_anh.py`:** nén ảnh `china.jpg` còn 4, 16 và 64 màu.

## Chạy bài tập

Sáu bài tập được đặt riêng trong `baitap08` theo yêu cầu của PDF:

```powershell
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai1.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai2.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai3.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai4.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai5.py
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai6.py
```

1. **Bài 1:** hàm `silhouette_mot_diem(X, nhan, i)` chỉ dùng numpy; tìm khách có s nhỏ nhất và đánh dấu trên hình `baitap08/silhouette_khach_yeu_nhat.png`.
2. **Bài 2:** phân cụm trên cả ba cột tuổi, thu nhập, điểm chi tiêu; chọn K từ 2 tới 10 bằng silhouette.
3. **Bài 3:** 50 lần chạy với `n_init=1`, so `init="random"` và `init="k-means++"`.
4. **Bài 4:** lưu `baitap08/anh_goc.png` và `baitap08/anh_16_mau.png`, so dung lượng hai tệp.
5. **Bài 5:** DBSCAN với eps 0.2, 0.3, 0.4 trên hai vầng trăng, lưu `baitap08/dbscan_hai_vang_trang.png`.
6. **Bài 6:** K-Means K = 10 trên chữ số viết tay, sau đó mới lập `pd.crosstab` với nhãn thật.

Phần hai của bài 4 dùng ảnh của chính em. Thay đường dẫn ví dụ bằng ảnh thật, JPG hoặc PNG đều được; PNG có kênh trong suốt sẽ được bỏ kênh thứ tư trước khi phân cụm:

```powershell
.\.venv\Scripts\python.exe .\Lab08_KMeans\baitap08\bai4.py --anh "D:\Anh\anh_cua_em.jpg"
```

## Dữ liệu và cách đánh giá

- `data/khach_hang.csv`: dữ liệu gốc của bài, 210 khách hàng với ba cột `tuoi`, `thu_nhap`, `diem_chi_tieu`, không có nhãn.
- `data/hai_vang_trang.csv`: dữ liệu hai vầng trăng 300 điểm của bài 4, cột `nhan` chỉ dùng để chấm sau khi phân cụm.
- Ảnh `china.jpg` và bộ chữ số `load_digits()` có sẵn trong scikit-learn, không cần tải thêm.
- Thu nhập và điểm chi tiêu được chuẩn hoá bằng `StandardScaler` trước khi phân cụm. K-Means dùng `n_init=10`, `random_state=0`, trừ những chỗ PDF cố ý dùng `n_init=1`.

## Giải thích kết quả

Năm chương trình hướng dẫn cho **đúng mọi con số** trong PDF:

- `k1`: dừng sau **3** vòng, tâm (1.25, 1.50) và (3.90, 5.10), inertia **8.5250**, trùng scikit-learn.
- `k2`: inertia **23.02**; khách thu nhập 29, chi tiêu 85 vào cụm **4** (thu nhập cao, chi nhiều); khách thu nhập 9, chi tiêu 75 vào cụm **0** (thu nhập thấp, chi nhiều).
- `k3`: từ K = 4 lên 5 inertia giảm 50.7, từ 5 lên 6 chỉ giảm 2.1; silhouette cao nhất **0.723** tại **K = 5**.
- `k4`: không chuẩn hoá thì năm tâm chỉ khác nhau theo thu nhập; **10/20** lần khởi tạo ngẫu nhiên đạt 23.0, tệ nhất 118.2; hai vầng trăng khớp nhãn thật **0.7500**.
- `k5`: ảnh 427 x 640 có 96615 màu; 16 màu nhỏ hơn **6.0** lần, lệch trung bình **7.97** mức sáng.

Bài tập:

- **Bài 1:** silhouette trung bình tự tính **0.722706**, đúng bằng `silhouette_score`. Khách có s nhỏ nhất là khách thứ **113** (đánh số từ 0): thu nhập 25.4 triệu, điểm chi tiêu 41, s = **0.0286**. Khách thuộc cụm thu nhập vừa chi vừa nhưng nằm ở mép phải-dưới của cụm, gần như ở giữa tâm cụm đó (17.4, 49.3) và tâm cụm thu nhập cao chi ít (30.1, 20.0). Vì vậy a = 0.9178 gần bằng b = 0.9448, nên s gần 0.
- **Bài 2:** K tốt nhất **vẫn là 5**, silhouette **0.481**, thấp hơn 0.723 của hai cột vì tuổi trải đều, không tạo thêm nhóm rõ. Năm tâm gần như giữ nguyên năm nhóm của mục 3, nay có thêm tọa độ tuổi. Bảng chéo cho thấy 205/210 khách giữ nguyên nhóm: 4 khách từ cụm giữa chuyển sang nhóm trẻ thu nhập thấp chi nhiều (tuổi trung bình 25.1), 1 khách từ nhóm thu nhập cao chi nhiều chuyển sang cụm giữa.
- **Bài 3:** `init="random"` chỉ đạt inertia 23.0 trong **30/50** lần, 20 lần còn lại dừng ở cách chia kém, tệ nhất 204.0. `init="k-means++"` đạt 23.0 cả **50/50** lần. K-Means++ đáng tin hơn vì chọn các tâm ban đầu nằm xa nhau, ít rơi vào trường hợp hai tâm cùng một nhóm.
- **Bài 4:** `anh_goc.png` **537.3 KB**, `anh_16_mau.png` **131.3 KB**, nhỏ hơn **4.09** lần, lệch trung bình 7.97 mức sáng. Tỷ lệ thật thấp hơn mức lý thuyết 6 lần vì `plt.imsave` không lưu chỉ số màu 4 bit mà vẫn ghi đủ giá trị màu của từng điểm, rồi PNG tự nén không mất dữ liệu; ảnh ít màu chỉ nén được tốt hơn chứ không đúng bằng 6 lần.
- **Bài 5:** K-Means khớp nhãn thật **0.7500**. DBSCAN với eps **0.2 / 0.3 / 0.4** đều chỉ tìm được **1** cụm, với **40 / 10 / 3** điểm nhiễu, nên **không** giá trị nào tách được hai vầng trăng tốt hơn K-Means. Thử thêm eps nhỏ hơn: 0.1 cho 6 cụm và 257 điểm nhiễu, 0.15 cho 16 cụm và 99 điểm nhiễu. eps là bán kính lân cận: nhỏ thì phần lớn điểm thành nhiễu và cụm bị vụn; lớn thì các vùng dày nối thành một. Bộ dữ liệu này có nhiễu lớn, hai vầng trăng chạm nhau, nên khi eps đủ lớn để nối liền một vầng thì cũng nối luôn sang vầng kia.
- **Bài 6:** chữ số **0** được gom gọn nhất, **176/178** ảnh (98.9%) vào cùng một cụm; tiếp theo là 6 (97.8%) và 7 (97.2%). Hay bị gom chung: **1 và 8** (một cụm chứa 100 ảnh số 1 và 100 ảnh số 8), **9, 8 và 5** (một cụm 139 / 52 / 41 ảnh). Số 1 còn bị tách ra ba cụm.

Nhận xét:

- K-Means chỉ chia cụm; việc đọc tâm cụm theo đơn vị gốc và đặt tên cho từng cụm là của con người. Số thứ tự cụm không có ý nghĩa.
- Luôn chuẩn hoá trước khi phân cụm, luôn đặt `n_init` và `random_state`.
- K-Means và DBSCAN đều có giới hạn: K-Means cắt bằng đường thẳng, DBSCAN cần vùng thưa ngăn cách các cụm.
