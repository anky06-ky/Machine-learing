# Bài 03: Naive Bayes

Thư mục này chứa mã nguồn và tài liệu thực hành của Lab03 về thuật toán phân loại **Naive Bayes** (mô hình sinh - generative classifier). Các bài chạy độc lập và tự động định vị dữ liệu trong thư mục `data/` theo vị trí file, cho phép chạy linh hoạt từ thư mục gốc hoặc trực tiếp trong thư mục bài tập. Chú thích dùng tiếng Việt có dấu; đầu ra terminal được định dạng chuẩn xác, tương thích tốt trên Windows PowerShell.

---

## Cấu trúc thư mục

```text
03_Naive_Bayes/
├── README.md                 # Hướng dẫn chi tiết lý thuyết, kết quả và nhận xét
├── Lab03_Naive_Bayes.pdf     # Đề bài và tài liệu hướng dẫn thực hành
├── requirements.txt          # Danh sách thư viện cần thiết
├── data/
│   ├── sinh_vien.csv         # Dữ liệu sinh viên (giờ ôn, điểm giữa kỳ, kết quả qua môn)
│   └── tin_nhan.csv          # 300 tin nhắn gán nhãn rác (1) hoặc thường (0)
├── d1_bayes_tay.py           # Bước 1: Tính định lý Bayes bằng tay (không dùng thư viện)
├── d2_tui_tu.py              # Bước 2: Túi từ (Bag of Words) và tiền xử lý văn bản
├── d3_loc_thu_rac.py         # Bước 3: Huấn luyện bộ lọc thư rác với MultinomialNB
├── d4_lam_tron.py            # Bước 4: Tự cài đặt và xử lý xác suất 0 bằng làm trơn Laplace
└── d5_gaussian.py            # Bước 5: GaussianNB cho biến liên tục, so sánh Logistic Regression
```

---

## Cài đặt và thực thi

Từ thư mục gốc của repository, cài đặt các thư viện cần thiết:

```powershell
python -m pip install -r .\03_Naive_Bayes\requirements.txt
```

Chạy từng chương trình từ thư mục gốc của repository:

```powershell
python .\03_Naive_Bayes\d1_bayes_tay.py
python .\03_Naive_Bayes\d2_tui_tu.py
python .\03_Naive_Bayes\d3_loc_thu_rac.py
python .\03_Naive_Bayes\d4_lam_tron.py
python .\03_Naive_Bayes\d5_gaussian.py
```

Hoặc di chuyển vào thư mục `03_Naive_Bayes` và thực thi:

```powershell
cd .\03_Naive_Bayes
python d1_bayes_tay.py
python d2_tui_tu.py
python d3_loc_thu_rac.py
python d4_lam_tron.py
python d5_gaussian.py
```

---

## Chi tiết từng bài thực hành

### Bước 1: Tính định lý Bayes bằng tay (`d1_bayes_tay.py`)

Cài đặt định lý Bayes tổng quát tính xác suất hậu nghiệm dựa trên công thức xác suất toàn phần:

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B \mid A) \cdot P(A) + P(B \mid \neg A) \cdot P(\neg A)}$$

#### Ví dụ 1: Bài toán xét nghiệm bệnh
- **Dữ kiện:** Tỷ lệ mắc bệnh trong cộng đồng là $1\%$ ($P(A) = 0.01$); độ nhạy xét nghiệm $99\%$ ($P(B \mid A) = 0.99$); tỷ lệ dương tính giả $1\%$ ($P(B \mid \neg A) = 0.01$).
- **Kết quả:** Người có kết quả dương tính chỉ có **$50.00\%$** khả năng thực sự có bệnh.
- **Khảo sát độ nhạy theo tỷ lệ mắc bệnh ban đầu:**

| Tỷ lệ mắc bệnh $P(A)$ | Xác suất có bệnh khi dương tính $P(A \mid B)$ |
|:----------------------:|:---------------------------------------------:|
| $0.001$ ($0.1\%$)      | $0.0902$ ($9.02\%$)                           |
| $0.010$ ($1.0\%$)      | $0.5000$ ($50.0\%$)                           |
| $0.050$ ($5.0\%$)      | $0.8390$ ($83.9\%$)                           |
| $0.100$ ($10\%$)       | $0.9167$ ($91.67\%$)                          |
| $0.500$ ($50\%$)       | $0.9900$ ($99.0\%$)                           |

#### Ví dụ 2: Dây chuyền sản xuất
- Dây chuyền A sản xuất $80\%$ sản phẩm, tỷ lệ lỗi $1\%$.
- Dây chuyền B sản xuất $20\%$ sản phẩm, tỷ lệ lỗi $5\%$.
- Khi nhặt ngẫu nhiên được một sản phẩm lỗi, xác suất sản phẩm đó đến từ dây chuyền B là:
  $$P(B \mid \text{lỗi}) = \frac{0.05 \times 0.20}{0.05 \times 0.20 + 0.01 \times 0.80} = \frac{0.010}{0.018} \approx \mathbf{0.5556}\ (55.56\%)$$
- Kiểm chứng thực tế trên 1000 sản phẩm: Dây A có 8 lỗi, Dây B có 10 lỗi. Tổng số lỗi = 18. Tỷ lệ từ B = $10 / 18 \approx 0.5556$.

---

### Bước 2: Túi từ (Bag of Words) và biểu diễn văn bản (`d2_tui_tu.py`)

Sử dụng `CountVectorizer` của `scikit-learn` để chuyển đổi văn bản sang vector đếm từ số học.
- Đặt `token_pattern=r"(?u)\S+"` nhằm tách từ đơn giản bằng khoảng trắng, giữ lại đầy đủ các từ tiếng Việt và từ đơn 1 ký tự.
- **Thống kê dữ liệu `tin_nhan.csv`:**
  - Tổng số mẫu: **300 tin nhắn**.
  - Tin thường (nhãn 0): **180 tin** ($60\%$).
  - Tin rác (nhãn 1): **120 tin** ($40\%$).
  - Kích thước từ điển: **40 từ**.

#### 10 từ đặc trưng xuất hiện nhiều nhất theo từng lớp:

| Hạng | 10 từ xuất hiện nhiều nhất ở Tin rác | 10 từ xuất hiện nhiều nhất ở Tin thường |
|:----:|:-------------------------------------|:---------------------------------------|
| 1    | `tay` (48 lần)                       | `thầy` (74 lần)                        |
| 2    | `ngay` (47 lần)                      | `bài` (73 lần)                         |
| 3    | `nhanh` (44 lần)                     | `cáo` (72 lần)                         |
| 4    | `chót` (43 lần)                      | `chiều` (68 lần)                       |
| 5    | `giảm` (42 lần)                      | `ạ` (65 lần)                           |
| 6    | `lúc` (42 lần)                       | `tài` (64 lần)                         |
| 7    | `quà` (41 lần)                       | `tập` (63 lần)                         |
| 8    | `click` (40 lần)                     | `gửi` (61 lần)                         |
| 9    | `phí` (40 lần)                       | `lớp` (60 lần)                         |
| 10   | `miễn` (39 lần)                      | `nhóm` (58 lần)                        |

> **Nhận xét:** Túi từ chỉ quan tâm đến tần suất từ xuất hiện, bỏ qua hoàn toàn thứ tự từ. Trong bài toán lọc thư rác, đặc trưng từ vựng phân hóa rất mạnh giữa hai nhóm nên mô hình hoạt động hiệu quả.

---

### Bước 3: Huấn luyện bộ lọc thư rác MultinomialNB (`d3_loc_thu_rac.py`)

Chia tập dữ liệu theo tỷ lệ 75:25 với `random_state=17` và phân tầng `stratify=y`:
- Tập huấn luyện: **225 tin**.
- Tập kiểm tra: **75 tin**.

#### Ma trận nhầm lẫn (Confusion Matrix) trên 75 tin kiểm tra:

| Thực tế \ Dự đoán | Dự đoán Thường (0) | Dự đoán Rác (1) |
|:------------------|:------------------:|:---------------:|
| **Thật: Thường (0)** | **TN = 45** (giữ đúng) | **FP = 0** (xóa nhầm) |
| **Thật: Rác (1)**    | **FN = 4** (lọt lưới)  | **TP = 26** (chặn đúng) |

#### Kết quả đánh giá mô hình:
- **Accuracy = 0.9467** ($94.67\%$ tổng số tin được phân loại chính xác).
- **Precision = 1.0000** ($100\%$ tin bị gắn cờ rác đều thực sự là rác; không có tin thường nào bị xóa nhầm).
- **Recall = 0.8667** ($86.67\%$ tin rác bị bắt giữ, còn lọt lưới 4 tin).

#### Thử nghiệm suy luận trên tin nhắn thực tế:
- `"trung thuong khuyen mai nhanh tay click ngay"`: $P(\text{rác}) = \mathbf{0.9992} \rightarrow$ **RAC**
- `"hop nhom bao cao chieu mai phong 305"`: $P(\text{rác}) = \mathbf{0.0983} \rightarrow$ **thuong**
- `"ban gui giup em tai lieu bai tap nhe"`: $P(\text{rác}) = \mathbf{0.0719} \rightarrow$ **thuong**

---

### Bước 4: Vấn đề số không và làm trơn Laplace (`d4_lam_tron.py`)

Tự lập trình thuật toán Naive Bayes trên tập dữ liệu mẫu nhỏ để minh họa hiện tượng "Zero Probability" (xác suất bằng 0 triệt tiêu toàn bộ tích xác suất).

- **Thử nghiệm với câu:** `"trung thuong bai tap"`
  - Từ `"bai"` chỉ xuất hiện ở lớp thường, chưa bao giờ xuất hiện ở lớp rác ($P(\text{"bai"} \mid \text{rác}) = 0$).
  - Từ `"trung"` chỉ xuất hiện ở lớp rác, chưa bao giờ xuất hiện ở lớp thường ($P(\text{"trung"} \mid \text{thường}) = 0$).

#### So sánh giữa không làm trơn và có làm trơn Laplace:

| Hệ số $\alpha$ | Điểm lớp Rác | Điểm lớp Thường | Kết luận | Đánh giá |
|:--------------:|:------------:|:---------------:|:--------:|:---------|
| $\alpha = 0.0$ (Không làm trơn) | $0.0000000000$ | $0.0000000000$ | `thuong` | **Mô hình bị tê liệt** vì cả 2 điểm số đều bằng 0 do có từ chưa từng gặp |
| $\alpha = 1.0$ (Làm trơn Laplace) | $\mathbf{0.0000073212}$ | $0.0000021656$ | **`rac`** | **Mô hình phân loại đúng**, điểm lớp rác vượt trội nhờ hai từ mạnh `trung`, `thuong` |

Công thức làm trơn Laplace với từ điển kích thước $V$:

$$P(w \mid c) = \frac{\text{count}(w, c) + \alpha}{\sum_{w'} \text{count}(w', c) + \alpha \cdot V}$$

---

### Bước 5: Gaussian Naive Bayes cho dữ liệu liên tục (`d5_gaussian.py`)

Sử dụng `GaussianNB` để phân loại dữ liệu dạng số thực liên tục (`gio_on` và `diem_giua_ky` từ `sinh_vien.csv`), so sánh trực tiếp với mô hình `LogisticRegression` trên cùng dữ liệu và cùng cách chia (`test_size=0.25`, `random_state=17`, `stratify=y`).

#### So sánh hiệu năng mô hình:

| Mô hình | Accuracy trên tập kiểm tra |
|:-------|:---------------------------:|
| **Naive Bayes (GaussianNB)** | **0.9000** ($90.00\%$) |
| **Hồi quy logistic (LogisticRegression)** | **0.9000** ($90.00\%$) |

#### Phân phối Gauss học được từ tập dữ liệu:

- **Lớp 0 (Rớt môn):**
  - Giờ ôn tập (`gio_on`): Trung bình $\mu = 8.18$ giờ, độ lệch chuẩn $\sigma = 5.29$.
  - Điểm giữa kỳ (`diem_giua_ky`): Trung bình $\mu = 3.99$, độ lệch chuẩn $\sigma = 1.92$.
- **Lớp 1 (Qua môn):**
  - Giờ ôn tập (`gio_on`): Trung bình $\mu = 20.03$ giờ, độ lệch chuẩn $\sigma = 5.81$.
  - Điểm giữa kỳ (`diem_giua_ky`): Trung bình $\mu = 6.70$, độ lệch chuẩn $\sigma = 1.50$.
- **Tỷ lệ tiên nghiệm (class priors):** Rớt môn: $41.11\%$, Qua môn: $58.89\%$.

#### Dự đoán xác suất cho học viên mới:

| Giờ ôn | Điểm GK | Xác suất qua môn $P(\text{qua})$ | Dự đoán nhãn (ngưỡng 0.5) |
|:------:|:-------:|:--------------------------------:|:-------------------------:|
| 5      | 4.0     | **0.0137** ($1.37\%$)            | 0 (Rớt)                   |
| 13     | 6.0     | **0.6550** ($65.50\%$)           | 1 (Qua)                   |
| 20     | 7.0     | **0.9855** ($98.55\%$)           | 1 (Qua)                   |
| 28     | 9.0     | **0.9999** ($99.99\%$)           | 1 (Qua)                   |

---

## Bảng tra cứu các biến thể Naive Bayes

| Biến thể | Kiểu dữ liệu phù hợp | Trường hợp sử dụng tiêu biểu |
|:--------|:---------------------|:-----------------------------|
| **`MultinomialNB`** | Dữ liệu đếm rời rạc (tần suất từ, số lần lặp) | Phân loại văn bản, lọc thư rác, phân tích chủ đề |
| **`GaussianNB`** | Dữ liệu số liên tục tuân theo phân phối chuẩn | Đặc trưng cảm biến, chiều cao, cân nặng, số đo |
| **`BernoulliNB`** | Dữ liệu nhị phân (có xuất hiện hay không, 0/1) | Phân loại văn bản với từ điển nhị phân, kiểm tra sự hiện diện đặc trưng |
