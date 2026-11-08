# Kết quả chạy Lab08_KMeans

Cập nhật: 2026-11-08 20:58 +0700. Python 3.14.8.

Các giá trị dưới đây được lấy trực tiếp từ stdout/stderr, không chép số mẫu trong PDF.

**Nguồn:** `khach_hang.csv` đi kèm hiện là dữ liệu mô phỏng thay thế có seed cố định. Xem README trước khi đối chiếu với PDF. Nếu đã thay CSV gốc, cập nhật ghi chú nguồn trong README.

Ảnh trong `ket_qua/` là kết xuất văn bản đầu ra, không phải ảnh chụp terminal.

## Môi trường

- numpy: 2.5.4
- pandas: 3.1.0
- scikit-learn: 1.9.1
- matplotlib: 3.11.2
- Pillow: 13.0.0

## Dấu vân tay dữ liệu (SHA-256)

- `hai_vang_trang.csv`: `6b2fd8083f3e3b7f71a1059aed8dc86424bb0c62f7e30f0da796a946e1de66bb`
- `khach_hang.csv`: `194f972d09b5c6da33638429296858e28d9dfebb8a78b6f1339d2887e1c0482f`

## Đầu ra từng chương trình


### code/k1_tu_viet.py

[Log TXT](./ket_qua/k1_tu_viet.txt) · [Ảnh kết quả](./ket_qua/k1_tu_viet.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python code/k1_tu_viet.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Vong 1: tam = [[1.0, 1.0], [1.5, 2.0]]
  cum 1: A
  cum 2: B, C, D, E, F, G
Vong 2: tam = [[1.0, 1.0], [3.5, 4.58]]
  cum 1: A, B
  cum 2: C, D, E, F, G
Vong 3: tam = [[1.25, 1.5], [3.9, 5.1]]
  cum 1: A, B
  cum 2: C, D, E, F, G
  Tam khong doi nua, dung.
Inertia tu tinh: 8.5250; scikit-learn: 8.5250
Tam scikit-learn: [[1.25 1.5 ]
 [3.9  5.1 ]]
```


### code/k2_khach_hang.py

[Log TXT](./ket_qua/k2_khach_hang.txt) · [Ảnh kết quả](./ket_qua/k2_khach_hang.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python code/k2_khach_hang.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Kich thuoc bang: (210, 3)
        tuoi  thu_nhap  diem_chi_tieu
count  210.0     210.0          210.0
mean    37.9      18.4           50.3
std      9.0       9.5           26.2
min     18.0       3.0            6.0
25%     33.0       9.0           23.2
50%     38.0      16.6           51.0
75%     44.0      29.0           76.0
max     58.0      35.2           97.0
Tam cum, don vi goc:
    thu_nhap  diem_chi_tieu  so_khach  tuoi_tb
0      29.9           20.8        35     43.2
1       7.6           80.6        35     24.9
2      17.0           51.1        60     39.5
3      29.6           79.2        40     34.8
4       8.4           19.6        40     45.3
Inertia: 25.40
Khach moi:
  thu_nhap  diem_chi_tieu  cum
     29.0             85    3
      9.0             75    1
Da luu: figures/phan_khuc_khach_hang.png
Doc y nghia cum theo tam; so thu tu cum khong phai thu hang hay nhan that.
```


### code/k3_chon_k.py

[Log TXT](./ket_qua/k3_chon_k.txt) · [Ảnh kết quả](./ket_qua/k3_chon_k.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python code/k3_chon_k.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

 K    inertia   giam so voi K-1   silhouette
 1     420.00              nan          nan
 2     249.07           170.93       0.4200
 3     157.14            91.93       0.5065
 4      74.94            82.20       0.6140
 5      25.40            49.54       0.7064
 6      22.50             2.90       0.6046
 7      19.89             2.62       0.5595
 8      17.70             2.19       0.4940
 9      15.87             1.83       0.4373
10      14.33             1.54       0.3722
K co silhouette cao nhat: 5
Da luu: figures/chon_k.png
Inertia luon giam khi K tang; khong chon K chi vi inertia nho nhat.
```


### code/k4_ba_cai_bay.py

[Log TXT](./ket_qua/k4_ba_cai_bay.txt) · [Ảnh kết quả](./ket_qua/k4_ba_cai_bay.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python code/k4_ba_cai_bay.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Bay 1: tam khi thu nhap ghi bang nghin dong, doi lai trieu de doc:
[[ 6.9 48.6]
 [11.5 47.2]
 [17.3 51.3]
 [26.8 49.7]
 [30.8 52.9]]
Bay 2: inertia 20 lan khoi tao: [ 25.4  25.4  25.4  25.4  25.4  25.4  25.4  25.4  25.4  25.4  25.4  25.4
  25.4  25.4  71.7  73.5  75.8  77.1 116.5 120.3]
n_init=10, inertia=25.40
Bay 3: ty le khop nhan (co xet hoan vi)=0.7500
Da luu: figures/bay_thang_do.png
K-Means gan theo tam gan nhat, khong bam duoc duong cong cua hai vang trang.
```


### code/k5_nen_anh.py

[Log TXT](./ket_qua/k5_nen_anh.txt) · [Ảnh kết quả](./ket_qua/k5_nen_anh.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python code/k5_nen_anh.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Anh 427 x 640; 273280 pixel; 96615 mau
 K   bit/pixel   ti le ly thuyet   MAE muc sang
 4          2              12.0          17.41
16          4               6.0           7.97
64          6               4.0           4.54
Da luu: figures/nen_mau.png
Ti le 24/bit chi la ly thuyet cho chi so mau, chua tinh bang mau va dinh dang tep.
```


### baitap08/bai1.py

[Log TXT](./ket_qua/bai1.txt) · [Ảnh kết quả](./ket_qua/bai1.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python baitap08/bai1.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Tu tinh trung binh=0.7063508042; sklearn=0.7063508042
Khach co s nho nhat: index=132 (dong CSV=134), cum=2
tuoi             41.0
thu_nhap         24.6
diem_chi_tieu    57.0
a, b, s: [0.87907684 1.02706936 0.14409204]
Da luu: figures/silhouette_khach_yeu_nhat.png
Khach s thap nam gan ranh gioi, hoac xa nhom cua minh so voi nhom ben canh; xem dau sao.
```


### baitap08/bai2.py

[Log TXT](./ket_qua/bai2.txt) · [Ảnh kết quả](./ket_qua/bai2.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python baitap08/bai2.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

K= 2, silhouette=0.3564
K= 3, silhouette=0.3928
K= 4, silhouette=0.4631
K= 5, silhouette=0.5086
K= 6, silhouette=0.4784
K= 7, silhouette=0.4784
K= 8, silhouette=0.4436
K= 9, silhouette=0.4258
K=10, silhouette=0.4106
K tot nhat: 5 ; con bang 5: True
Tam ba chieu, don vi goc:
  tuoi  thu_nhap  diem_chi_tieu
24.89      7.85          79.78
45.67      8.39          19.33
39.60     16.88          50.75
34.75     29.57          79.25
43.20     29.94          20.80
Tam hai chieu ban dau:
  thu_nhap  diem_chi_tieu
     7.59          80.63
     8.38          19.55
    17.03          51.12
    29.57          79.25
    29.94          20.80
Them tuoi doi khoang cach va cach gom nhom; so sanh toa do tam, khong doi chieu bang so cum.
```


### baitap08/bai3.py

[Log TXT](./ket_qua/bai3.txt) · [Ảnh kết quả](./ket_qua/bai3.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python baitap08/bai3.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

Inertia tot nhat quan sat duoc: 25.398732 (khong khang dinh toi uu toan cuc)
random    : dat 23.0 (lam tron 1 so) 0/50; dat muc tot nhat 34/50; min=25.3987, max=227.4046, mean=48.2979
k-means++ : dat 23.0 (lam tron 1 so) 0/50; dat muc tot nhat 50/50; min=25.3987, max=25.3987, mean=25.3987
On dinh hon theo tan suat dat muc tot nhat: k-means++
K-means++ uu tien tam xa nhau; van nen dung n_init>1 vi khong bao dam toi uu.
Da luu: figures/so_sanh_khoi_tao.png
```


### baitap08/bai4.py

[Log TXT](./ket_qua/bai4.txt) · [Ảnh kết quả](./ket_qua/bai4.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python baitap08/bai4.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

china: shape=(427, 640, 3); mau sau luu=16
PNG goc=550154 byte; PNG 16 mau=134431 byte; ti le=4.09 lan
Da luu: anh_goc.png anh_16_mau.png
CHUA CO ANH CA NHAN: dang minh hoa lan hai bang flower.jpg; chay --anh DUONG_DAN de hoan tat.
minh_hoa_: shape=(427, 640, 3); mau sau luu=16
PNG goc=419413 byte; PNG 16 mau=55631 byte; ti le=7.54 lan
Da luu: minh_hoa_anh_goc.png minh_hoa_anh_16_mau.png
Dung luong PNG con phu thuoc nen va bang mau; khong bang ti le ly thuyet 6 lan.
```


### baitap08/bai5.py

[Log TXT](./ket_qua/bai5.txt) · [Ảnh kết quả](./ket_qua/bai5.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python baitap08/bai5.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

KMeans: ARI=0.2462
eps=0.2: 7 cum, 31 diem nhieu, ARI=0.1282, tot hon KMeans=False
eps=0.3: 2 cum, 1 diem nhieu, ARI=0.0015, tot hon KMeans=False
eps=0.4: 1 cum, 0 diem nhieu, ARI=0.0000, tot hon KMeans=False
Da luu: figures/dbscan_hai_vang_trang.png
eps nho: de tach vun va co nhieu; eps lon: co the noi hai vang trang thanh mot cum.
```


### baitap08/bai6.py

[Log TXT](./ket_qua/bai6.txt) · [Ảnh kết quả](./ket_qua/bai6.png)

```text
Tran An Ky - 2474802010460 | Lab08_KMeans
LENH: python baitap08/bai6.py
MA THOAT: 0
NGUON: xem data/NGUON.md; lab co du lieu mo phong thay the.
BAN KET XUAT LOG CHAY THUC TE - KHONG PHAI ANH CHUP TERMINAL

cum          0    1    2   3    4    5    6    7    8    9
nhan_that
0          176    0    0   0    0    0    2    0    0    0
1            0  100    0  54    1    2    0    1    0   24
2            1    8    3   2   13    0    0    0    2  148
3            0    7    7   0  154    0    0    2   13    0
4            0    2   11   3    0    0  165    0    0    0
5            0    0    0   0    2    1    2  136   41    0
6            1    3    0   0    0  177    0    0    0    0
7            0    2  174   2    0    0    0    1    0    0
8            0  100    5   6    2    2    0    4   52    3
9            0    1    8  20    6    0    0    6  139    0
Ty le nam trong cum dong nhat cua moi chu so:
 nhan_that
0    0.9888
1    0.5495
2    0.8362
3    0.8415
4    0.9116
5    0.7473
6    0.9779
7    0.9721
8    0.5747
9    0.7722
Gom gon nhat: [0]
Cum 0: hai chu so dong nhat {0: 176, 2: 1}
Cum 1: hai chu so dong nhat {1: 100, 8: 100}
Cum 2: hai chu so dong nhat {7: 174, 4: 11}
Cum 3: hai chu so dong nhat {1: 54, 9: 20}
Cum 4: hai chu so dong nhat {3: 154, 2: 13}
Cum 5: hai chu so dong nhat {6: 177, 1: 2}
Cum 6: hai chu so dong nhat {4: 165, 0: 2}
Cum 7: hai chu so dong nhat {5: 136, 9: 6}
Cum 8: hai chu so dong nhat {9: 139, 8: 52}
Cum 9: hai chu so dong nhat {2: 148, 1: 24}
Nhom hinh dang giong nhau co the bi gom chung; cum khong duoc danh so theo chu so.
```
