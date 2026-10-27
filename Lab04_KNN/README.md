# Bài 04: K láng giềng gần nhất (K-Nearest Neighbors - KNN)

Thư mục này chứa mã nguồn và tài liệu thực hành của Lab 04 về thuật toán **K láng giềng gần nhất** (K-Nearest Neighbors - KNN). KNN là một mô hình học lười (lazy learning), không học trước tham số mà giữ nguyên toàn bộ dữ liệu huấn luyện, chỉ tính toán khoảng cách và tìm láng giềng khi có mẫu dự đoán mới.

Các file mã nguồn được thiết kế chạy độc lập, tự động định vị tệp dữ liệu trong thư mục `data/` linh hoạt (hỗ trợ chạy từ thư mục gốc, thư mục con `code/` hoặc từ root của cả repository). Chú thích viết bằng tiếng Việt có dấu; đầu ra terminal được định dạng chuẩn xác, chống lỗi font trên Windows PowerShell.

---

## Cấu trúc thư mục

```text
Lab04_KNN/
├── README.md                 # Hướng dẫn chi tiết lý thuyết, kết quả và phân tích
├── Lab04_KNN.pdf             # Tài liệu thực hành và bài tập
├── requirements.txt          # Danh sách thư viện cần thiết (numpy, pandas, scikit-learn)
├── data/                     # Thư mục chứa các bộ dữ liệu mô phỏng
│   ├── gia_nha.csv           # Dữ liệu giá nhà (diện tích, giá) dùng cho hồi quy
│   ├── hai_vang_trang.csv    # 300 điểm dữ liệu hai vầng trăng khuyết lồng nhau
│   └── sinh_vien_mo_rong.csv # Dữ liệu sinh viên mở rộng (giờ ôn, điểm GK, chi tiêu, qua môn)
├── e1_khoang_cach.py         # Mục 2: Tự tính khoảng cách và tìm láng giềng (NumPy)
├── e2_chuan_hoa.py           # Mục 3: Đo hậu quả của việc quên chuẩn hoá
├── e3_chon_k.py              # Mục 4: Chọn k bằng kiểm định chéo 5 phần
├── e4_so_sanh.py             # Mục 5: So sánh KNN với Hồi quy logistic và Naive Bayes
├── e5_knn_hoi_quy.py         # Mục 6: KNN cho bài toán hồi quy (KNeighborsRegressor)
└── code/                     # Bản sao các tệp mã nguồn để chạy theo lệnh tài liệu
    ├── e1_khoang_cach.py
    ├── e2_chuan_hoa.py
    ├── e3_chon_k.py
    ├── e4_so_sanh.py
    └── e5_knn_hoi_quy.py
```

---

## Cài đặt và thực thi

### 1. Cài đặt thư viện
Từ thư mục gốc của repository, cài đặt các thư viện cần thiết:

```powershell
python -m pip install -r .\Lab04_KNN\requirements.txt
```

### 2. Thứ tự chạy 5 tệp mã nguồn (theo đúng Bảng 1)

Bạn có thể chạy trực tiếp tại thư mục `Lab04_KNN`:

```powershell
cd .\Lab04_KNN
python e1_khoang_cach.py
python e2_chuan_hoa.py
python e3_chon_k.py
python e4_so_sanh.py
python e5_knn_hoi_quy.py
```

Hoặc chạy theo cú pháp `code/...` như hướng dẫn trong tài liệu:

```powershell
python code/e1_khoang_cach.py
python code/e2_chuan_hoa.py
python code/e3_chon_k.py
python code/e4_so_sanh.py
python code/e5_knn_hoi_quy.py
```

---

## Tóm tắt 5 tệp mã nguồn (Bảng 1)

| Mục | Tệp | Nội dung chính |
| :---: | :--- | :--- |
| **2** | `e1_khoang_cach.py` | Tự tính khoảng cách Euclid và tìm láng giềng bằng NumPy, quan sát sự áp đảo thang đo |
| **3** | `e2_chuan_hoa.py` | Đo hậu quả của việc quên chuẩn hoá trên tập kiểm tra, giải pháp với `make_pipeline` |
| **4** | `e3_chon_k.py` | Chọn số láng giềng $k$ tối ưu bằng kiểm định chéo (Cross-Validation) 5 phần trên tập học |
| **5** | `e4_so_sanh.py` | So sánh KNN với Hồi quy logistic và Naive Bayes trên dữ liệu có ranh giới cong |
| **6** | `e5_knn_hoi_quy.py` | Ứng dụng KNN cho bài toán hồi quy dự đoán giá nhà và phân tích giới hạn ngoại suy |

---

## Chi tiết kết quả thực nghiệm

### Mục 2: `e1_khoang_cach.py` - Tự tính khoảng cách và láng giềng
- **Công thức khoảng cách Euclid:** $d(x, z) = \sqrt{\sum_{i=1}^d (x_i - z_i)^2}$
- **Hiện tượng áp đảo thang đo:**
  - Khi chưa chuẩn hoá: $d(A, B) = 102.2$, $d(A, C) = 3000.0 \Rightarrow$ máy nhầm tưởng $A$ giống $B$ hơn $C$ chỉ vì cột `chi_tieu` có đơn vị quá lớn (độ lệch chuẩn 1692 so với 7.82 của giờ ôn và 2.09 của điểm số).
  - Sau khi chuẩn hoá z-score: $d(A, B) = 3.85$, $d(A, C) = 1.79 \Rightarrow C$ gần $A$ nhất, khớp đúng với trực giác học tập.
- **Dự đoán cho bạn mới (ôn 12 giờ, điểm giữa kỳ 6.5):** Tìm được 5 láng giềng gần nhất gồm 4 bạn qua và 1 bạn rớt $\rightarrow$ Mô hình bỏ phiếu dự đoán **qua môn**.

### Mục 3: `e2_chuan_hoa.py` - Đo hậu quả của việc quên chuẩn hoá
- **Đoán bừa cả lớp qua (baseline):** $0.6000$ (60.00%)
- **Ba cột, không chuẩn hoá:** $0.5333$ (thấp hơn cả đoán bừa do cột chi tiêu gây nhiễu áp đảo).
- **Ba cột, có chuẩn hoá (StandardScaler):** $0.8667$ (độ chính xác tăng vọt lên 86.67%).
- **Bỏ cột chi tiêu, có chuẩn hoá:** $0.9000$ (đạt độ chính xác cao nhất 90.00%).
- **Bài học quan trọng:**
  1. Với KNN, chuẩn hoá là bắt buộc vì mô hình phụ thuộc hoàn toàn vào khoảng cách hình học.
  2. Dùng `make_pipeline(StandardScaler(), KNeighborsClassifier(...))` để ngăn chặn rò rỉ dữ liệu (data leakage) từ tập kiểm tra sang tập huấn luyện.

### Mục 4: `e3_chon_k.py` - Chọn $k$ bằng kiểm định chéo
- Bộ dữ liệu `hai_vang_trang.csv` (300 mẫu, 2 lớp hình trăng khuyết lồng nhau):
  - $k = 1$: Trên tập học đạt $1.0000$ (học vẹt, mỗi điểm tự bỏ phiếu cho chính mình), nhưng tập kiểm tra chỉ đạt $0.8000$.
  - $k$ quá lớn ($k = 101$): Biên phân chia phẳng dần, cẩu thả, bỏ qua chi tiết, độ chính xác giảm còn $0.8533$.
  - Kiểm định chéo 5 phần (5-fold CV) trên tập học tìm ra $k = 13$ tối ưu với điểm trung bình $0.8667$.
  - Đánh giá trên tập kiểm tra (chỉ mở một lần duy nhất sau khi chọn): **Accuracy = 0.9200**.
  - **Lưu ý:** Luôn ưu tiên chọn $k$ lẻ trong bài toán 2 lớp để tránh hiện tượng hoà phiếu.

### Mục 5: `e4_so_sanh.py` - So sánh KNN với các mô hình đã học
- **Hồi quy logistic:** $0.8533$ (chỉ tạo được ranh giới đường thẳng).
- **Gaussian Naive Bayes:** $0.8533$ (ranh giới dạng bậc hai/ellipsoid, không khớp hoàn toàn hình trăng khuyết).
- **KNN ($k = 13$):** **0.9200** (thắng áp đảo nhờ là mô hình phi tham số - non-parametric, biên quyết định uốn lượn linh hoạt theo dữ liệu).

### Mục 6: `e5_knn_hoi_quy.py` - Hồi quy và giới hạn ngoại suy
- **So sánh MAE dự đoán giá nhà:**
  - Hồi quy tuyến tính (Linear Regression): **0.2578 tỷ đồng** (chiến thắng vì quan hệ diện tích - giá nhà thực tế gần như thẳng).
  - KNN Regressor ($k = 5$): **0.4030 tỷ đồng**.
- **Hiện tượng KNN không ngoại suy được:**
  - Căn $150 \text{ m}^2$ và $250 \text{ m}^2$ đều nhận chung một mức dự đoán là **9.32 tỷ đồng** vì cả hai đều lấy 5 láng giềng rộng nhất trong tập học (từ 104.2 tới 117.5 $\text{m}^2$).
  - KNN chỉ nội suy dựa trên các điểm lân cận đã thấy trong quá khứ, không thể phóng chiếu ra ngoài miền dữ liệu như đường thẳng.

---

## Bảng đối chiếu kết quả chuẩn (Phụ lục A - Bảng 3)

| Đại lượng / Kịch bản thực nghiệm | Giá trị chuẩn |
| :--- | :---: |
| $d(A, B)$ và $d(A, C)$ khi chưa chuẩn hoá | `102.2` và `3000.0` |
| $d(A, B)$ và $d(A, C)$ khi đã chuẩn hoá | `3.85` và `1.79` |
| Accuracy 3 cột, không chuẩn hoá | `0.5333` |
| Accuracy 3 cột, có chuẩn hoá | `0.8667` |
| Accuracy bỏ cột chi tiêu, có chuẩn hoá | `0.9000` |
| Mức đoán bừa trên tập kiểm tra | `0.6000` |
| $k$ tốt nhất theo 5-fold CV | `13` (điểm CV: `0.8667`) |
| Hai vầng trăng, KNN ($k = 13$) | `0.9200` |
| Hai vầng trăng, Logistic Regression & Naive Bayes | `0.8533` |
| Giá nhà, MAE của KNN ($k = 5$) và Đường thẳng | `0.4030` và `0.2578` |

---

## Các lỗi thường gặp và cách xử lý (Phụ lục B)

1. **Accuracy thấp hơn đoán bừa:** Quên chuẩn hoá trước khi đưa vào KNN, dẫn đến một cột có giá trị lớn (như tiền tệ, chi tiêu) đè bẹp các đặc trưng khác. Dùng `make_pipeline(StandardScaler(), KNeighborsClassifier(...))`.
2. **Accuracy trên tập học bằng 1.0:** Đang chọn $k = 1$ và chấm điểm trên chính tập học (mỗi điểm là láng giềng của chính nó). Luôn chấm điểm trên tập kiểm tra hoặc dùng kiểm định chéo.
3. **`ValueError: Expected n_neighbors <= n_samples`:** Đặt $k$ vượt quá số mẫu có trong tập học. Cần giảm $k$ hoặc kiểm tra lại tỷ lệ chia tập học/kiểm tra.
4. **Dự đoán chậm:** KNN phải so khoảng cách với toàn bộ mẫu đã học. Với tập dữ liệu lớn, cần giảm số chiều hoặc cân nhắc mô hình tham số.
5. **Rò rỉ dữ liệu (Data Leakage):** Chuẩn hoá toàn bộ dữ liệu trước khi `train_test_split`. Hãy luôn chuẩn hoá độc lập bằng pipeline để tập kiểm tra hoàn toàn giữ vai trò dữ liệu mới.
