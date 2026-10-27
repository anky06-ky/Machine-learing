# Bài 05: Cây quyết định và rừng ngẫu nhiên

Bài thực hành tìm hiểu cách cây quyết định chia dữ liệu, kiểm soát quá khớp và kết hợp nhiều cây bằng rừng ngẫu nhiên. Mỗi chương trình chạy độc lập; đường dẫn dữ liệu được xác định theo vị trí file Python.

## Cấu trúc thư mục

```text
Lab05_CayQuyetDinh/
├── README.md
├── Lab05_CayQuyetDinh_RungNgauNhien.pdf
├── requirements.txt
├── fi_gini.py
├── f2_cay_nam_la.py
├── f3_do_sau.py
├── f4_khong_chuan_hoa.py
├── f5_rung.py
├── f6_do_quan_trong.py
├── f7_cay_hoi_quy.py
└── data/
    ├── khach_vay.csv
    └── gia_nha.csv
```

## Cài đặt và chạy

Từ thư mục gốc của repository, cài các thư viện cần thiết:

```powershell
python -m pip install -r .\Lab05_CayQuyetDinh\requirements.txt
```

Chạy lần lượt 7 bài:

```powershell
python .\Lab05_CayQuyetDinh\fi_gini.py
python .\Lab05_CayQuyetDinh\f2_cay_nam_la.py
python .\Lab05_CayQuyetDinh\f3_do_sau.py
python .\Lab05_CayQuyetDinh\f4_khong_chuan_hoa.py
python .\Lab05_CayQuyetDinh\f5_rung.py
python .\Lab05_CayQuyetDinh\f6_do_quan_trong.py
python .\Lab05_CayQuyetDinh\f7_cay_hoi_quy.py
```

Cũng có thể chuyển vào thư mục bài rồi chạy bằng tên file:

```powershell
cd .\Lab05_CayQuyetDinh
python fi_gini.py
python f2_cay_nam_la.py
python f3_do_sau.py
python f4_khong_chuan_hoa.py
python f5_rung.py
python f6_do_quan_trong.py
python f7_cay_hoi_quy.py
```

## Nội dung từng bài

1. **`fi_gini.py` — Tính Gini bằng tay:** dùng 10 khách vay minh hoạ, tính độ hỗn tạp và mức giảm Gini tại các ngưỡng thu nhập; đối chiếu câu hỏi đầu tiên với cây sâu 1 của scikit-learn.
2. **`f2_cay_nam_la.py` — Đọc cây nhỏ:** huấn luyện cây tối đa 5 lá, in điều kiện chia nhánh, đánh giá accuracy và dự đoán khả năng trả đúng hạn cho hai khách mới.
3. **`f3_do_sau.py` — Chọn độ sâu:** so sánh cây mọc tự do với các độ sâu từ 1 đến 15, chọn bằng kiểm định chéo 5 phần trên tập huấn luyện rồi đánh giá mô hình đã chọn trên tập kiểm tra.
4. **`f4_khong_chuan_hoa.py` — Thay đổi thang đo:** so sánh cây sâu 5 trên dữ liệu gốc, dữ liệu chuẩn hoá và dữ liệu đổi đơn vị thu nhập; kiểm tra cả accuracy và số dự đoán khác nhau.
5. **`f5_rung.py` — Rừng ngẫu nhiên:** minh hoạ lấy mẫu có hoàn lại, khảo sát số cây và so sánh Random Forest với cây mọc tự do và cây sâu 5.
6. **`f6_do_quan_trong.py` — Độ quan trọng đặc trưng:** đọc `feature_importances_`, thêm một cột ngẫu nhiên để thấy điểm quan trọng dựa trên giảm độ hỗn tạp vẫn có thể gán trọng số cho nhiễu. Điểm này không chứng minh quan hệ nhân quả.
7. **`f7_cay_hoi_quy.py` — Cây hồi quy:** học cây sâu 3 trên dữ liệu giá nhà, so sánh MAE với hồi quy tuyến tính và quan sát dự đoán ngoài miền diện tích đã học.

## Dữ liệu và cách đánh giá

- `khach_vay.csv`: 600 khách vay; các đặc trưng gồm `thu_nhap`, `ty_le_no`, `co_viec_on_dinh`, `so_nguoi_phu_thuoc`; nhãn `tra_dung_han` bằng 1 khi trả đúng hạn, bằng 0 khi trễ hạn.
- `gia_nha.csv`: dữ liệu giá nhà dùng chung nội dung với bài hồi quy; bài này dùng `dien_tich` để dự đoán `gia` (tỷ đồng).
- Phân loại: chia 75% huấn luyện và 25% kiểm tra, `random_state=17`, phân tầng theo nhãn.
- Hồi quy: chia 80% huấn luyện và 20% kiểm tra, `random_state=42`.
- Các mô hình cây và rừng dùng `random_state=0` để tái lập kết quả.

## Kết quả tham khảo

- Cây tối đa 5 lá: accuracy tập học **0.7978**, tập kiểm tra **0.7467**.
- Cây mọc tự do: sâu **13**, có **92 lá**, accuracy tập học **1.0000**, tập kiểm tra **0.7667**.
- Kiểm định chéo chọn độ sâu **5**, điểm trung bình **0.8333**; cây được chọn đạt accuracy kiểm tra **0.8533**.
- Trong thí nghiệm đổi thang đo, ba cây cùng đạt accuracy **0.8533** và có **0** dự đoán khác nhau so với cây gốc.
- Rừng 200 cây: accuracy tập học **1.0000**, tập kiểm tra **0.9000**.
- Cây hồi quy sâu 3: MAE **0.5039 tỷ đồng**, so với **0.2578 tỷ đồng** của hồi quy tuyến tính.

Các giá trị trên là kết quả chạy với dữ liệu đi kèm. Kết quả có thể thay đổi khi đổi dữ liệu, tham số hoặc phiên bản thư viện. Các phép so sánh nhiều mô hình trên tập kiểm tra phục vụ minh hoạ; khi triển khai thực tế, chọn siêu tham số bằng kiểm định chéo trên tập huấn luyện và giữ tập kiểm tra cho đánh giá cuối cùng.
