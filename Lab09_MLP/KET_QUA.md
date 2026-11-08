# Kết quả chạy Lab09_MLP

Cập nhật: 2026-11-08 20:58 +0700. Python 3.14.8.

Các giá trị dưới đây được lấy trực tiếp từ stdout/stderr, không chép số mẫu trong PDF.

**Nguồn:** `xor.csv` đi kèm hiện là dữ liệu mô phỏng thay thế có seed cố định. Xem README trước khi đối chiếu với PDF. Nếu đã thay CSV gốc, cập nhật ghi chú nguồn trong README.

Ảnh trong `ket_qua/` là kết xuất văn bản đầu ra, không phải ảnh chụp terminal.

## Môi trường

- numpy: 2.5.4
- pandas: 3.1.0
- scikit-learn: 1.9.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

## Dấu vân tay dữ liệu (SHA-256)

- `gia_nha.csv`: `97a74b22fb939c16f317e337e9b9d2de8a2ae678930475514bf9ea3789a8f8c3`
- `hai_vang_trang.csv`: `6b2fd8083f3e3b7f71a1059aed8dc86424bb0c62f7e30f0da796a946e1de66bb`
- `sinh_vien_mo_rong.csv`: `459210ab197cb2089ca6e7efc1a54d3f9687a157805c9bcdac84ca78bf6b171d`
- `xor.csv`: `c9a14a34ab7c2d8a232674eb80ccbb0037b2d958653f729bc598c84195aa1ca1`

## Đầu ra từng chương trình


### code/m1_lan_truyen_tien.py

[Log TXT](./ket_qua/m1_lan_truyen_tien.txt) · [Ảnh kết quả](./ket_qua/m1_lan_truyen_tien.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python code/m1_lan_truyen_tien.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Mot no-ron: z=1.80, sigmoid(z)=0.8581
 x1 x2  h1 h2  y
[[0 0 0 0 0]
 [0 1 1 0 1]
 [1 0 1 0 1]
 [1 1 2 1 0]]
Tinh ca bang mot lan: [0. 1. 1. 0.]
h2 chi bat khi hai dau vao deu 1; tru 2*h2 dua dau ra (1,1) ve 0.
```


### code/m2_xor.py

[Log TXT](./ket_qua/m2_xor.txt) · [Ảnh kết quả](./ket_qua/m2_xor.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python code/m2_xor.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Logistic: 0.5800
No-ron an  so tham so   tap hoc    tap kiem tra
        1           5     0.6200         0.6600
        2           9     0.7133         0.6800
        4          17     0.9067         0.9200
        8          33     0.8967         0.9000
XOR khong tach duoc bang mot duong thang; lop an phi tuyen tao duoc ranh gioi gap khuc.
```


### code/m3_do_lon_mang.py

[Log TXT](./ket_qua/m3_do_lon_mang.txt) · [Ảnh kết quả](./ket_qua/m3_do_lon_mang.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python code/m3_do_lon_mang.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Lop an       alpha     tap hoc   tap kiem tra (minh hoa)
(2,)          0.0001     0.8933         0.8800
(100, 100)     1e-05     0.9867         0.9600
(100, 100)         1     0.9667         0.9800
alpha=  1e-05, CV=0.9400
alpha=  0.001, CV=0.9533
alpha=    0.1, CV=0.9467
alpha=      1, CV=0.9533
alpha=      3, CV=0.8733
alpha=     10, CV=0.8400
Chon bang CV: alpha=0.001, test=0.9200
Tap hoc cao nhung test thap la dau hieu hoc vet; khong chon alpha tu bang test minh hoa.
```


### code/m4_chuan_hoa.py

[Log TXT](./ket_qua/m4_chuan_hoa.txt) · [Ảnh kết quả](./ket_qua/m4_chuan_hoa.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python code/m4_chuan_hoa.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Doan ca lop qua: 0.6000
Khong chuan hoa: 0.4000
Co chuan hoa: 0.9000
Khong chuan hoa, so ban doan qua: 0 tren 30
Cot chi tieu thang do lon chi phoi toi uu; scaler chi fit tren tap hoc trong pipeline.
```


### code/m5_chu_so.py

[Log TXT](./ket_qua/m5_chu_so.txt) · [Ảnh kết quả](./ket_qua/m5_chu_so.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python code/m5_chu_so.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Kich thuoc ma tran trong so: [(64, 100), (100, 10)]
Tong so tham so: 7510
So vong hoc: 144; loss vong 1=2.5206, vong 10=0.4448, cuoi=0.0052
MLP test accuracy=0.9844; sai 7/450
anh   6: that=9, doan=3
anh  61: that=5, doan=7
anh  77: that=2, doan=1
anh  85: that=9, doan=7
anh 276: that=8, doan=2
anh 428: that=7, doan=9
anh 443: that=4, doan=7
Dung som: 44 vong, test=0.9578
Logistic: 0.9711
KNN k=5: 0.9867
SVM RBF: 0.9911
Da luu: figures/chu_so_loss_dung_som.png
Da luu: figures/chu_so_doan_sai.png
Dung som co the dung qua som; MLP khong luon thang KNN hay SVM tren bo du lieu nho.
```


### baitap09/bai1.py

[Log TXT](./ket_qua/bai1.txt) · [Ảnh kết quả](./ket_qua/bai1.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python baitap09/bai1.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

H = ReLU(X @ W1 + b1), p = sigmoid(H @ W2 + b2)
Shape X, W1, W2: (5, 2) (2, 8) (8, 1)
Diem 0: numpy=0.025967627941, sklearn=0.025967627941, lech=0.000e+00
Diem 1: numpy=0.653237727516, sklearn=0.653237727516, lech=0.000e+00
Diem 2: numpy=0.026270486112, sklearn=0.026270486112, lech=0.000e+00
Diem 3: numpy=0.031952827233, sklearn=0.031952827233, lech=0.000e+00
Diem 4: numpy=0.969594806683, sklearn=0.969594806683, lech=0.000e+00
Khop toi sai so so thuc; cot thu hai la xac suat lop 1
Moi cot W1 la trong so cua mot no-ron an; bias duoc cong theo tung cot.
```


### baitap09/bai2.py

[Log TXT](./ket_qua/bai2.txt) · [Ảnh kết quả](./ket_qua/bai2.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python baitap09/bai2.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

x1 x2 h1 h2 y
[[0 0 0 0 0]
 [0 1 0 1 1]
 [1 0 1 0 1]
 [1 1 0 0 0]]
h1+h2 = |x1-x2|: hai bit giong nhau cho 0, khac nhau cho 1, dung bang XOR.
Khong co ReLU, (x1-x2)+(x2-x1) luon bang 0; phi tuyen la phan quyet dinh.
```


### baitap09/bai3.py

[Log TXT](./ket_qua/bai3.txt) · [Ảnh kết quả](./ket_qua/bai3.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python baitap09/bai3.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Activation    tap hoc   tap kiem tra
logistic      0.8333         0.8200
tanh          0.8333         0.8200
relu          0.9667         0.9800
Cao nhat tren test minh hoa: ['relu']
Ket luan chi cho bo du lieu va seed nay; de chon activation cho trien khai can CV tren tap hoc.
```


### baitap09/bai4.py

[Log TXT](./ket_qua/bai4.txt) · [Ảnh kết quả](./ket_qua/bai4.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python baitap09/bai4.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

learning_rate   so vong    loss cuoi    test accuracy
       0.0001       582     0.018928           0.9889
        0.001       144     0.005210           0.9844
         0.01        38     0.001077           0.9778
          0.1        30     0.099601           0.9489
Da luu: figures/toc_do_hoc.png
Toc do qua nho: giam loss cham, co the het max_iter. Qua lon: de dao dong/nhay qua vung tot.
Doi chieu duong loss va accuracy thuc te; toc do lon khong nhat thiet that bai o moi seed.
```


### baitap09/bai5.py

[Log TXT](./ket_qua/bai5.txt) · [Ảnh kết quả](./ket_qua/bai5.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python baitap09/bai5.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

  lop_an  alpha  accuracy_cv  do_lech_chuan  hang
   (50,) 1.0000     0.978469       0.005461     1
(50, 50) 1.0000     0.977725       0.004075     2
  (100,) 1.0000     0.977723       0.005273     3
  (100,) 0.0001     0.975498       0.008007     4
  (100,) 0.0100     0.975498       0.008007     4
   (50,) 0.0100     0.975498       0.003805     4
   (50,) 0.0001     0.974013       0.004707     7
(50, 50) 0.0100     0.968063       0.010684     8
(50, 50) 0.0001     0.968063       0.010684     8
   (10,) 1.0000     0.965108       0.008640    10
   (10,) 0.0100     0.948757       0.015861    11
   (10,) 0.0001     0.948016       0.016129    12
Tham so tot nhat: {'mlpclassifier__alpha': 1.0, 'mlpclassifier__hidden_layer_sizes': (50,)}
CV accuracy=0.9785
Test: MLP=0.9822; SVM RBF=0.9911; MLP vuot SVM=False
Scaler nam trong pipeline nen moi fold chi chuan hoa bang tap hoc cua fold do.
```


### baitap09/bai6.py

[Log TXT](./ket_qua/bai6.txt) · [Ảnh kết quả](./ket_qua/bai6.png)

```text
Tran An Ky - 2474802010460 | Lab09_MLP
LENH: python baitap09/bai6.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Mien dien tich tap hoc: [37.6, 117.5] m2
MLP: MAE=0.5008 ty; gia 250m2=28.1758 ty
Tuyen tinh: MAE=0.2578 ty; gia 250m2=19.9139 ty
Da luu: figures/mlp_hoi_quy_ngoai_suy.png
250m2 nam ngoai mien hoc: day la ngoai suy, khong co gia that de khang dinh mo hinh nao dung.
MLP ReLU cho duong gap khuc; khong bi rang buoc tiep tuc xu huong nhu mo hinh tuyen tinh.
```
