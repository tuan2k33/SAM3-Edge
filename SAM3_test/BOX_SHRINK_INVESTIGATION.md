# Box bị co khi chạy SAM3 ở `imgsz` khác 1008: nhật ký nghiên cứu

Cập nhật: 2026-10-23. Trạng thái: xong, đã áp dụng vào `SAM3Fixed/export.py` và `SAM3Open/export.py` (bước 11).
Bối cảnh: `GETCOORDS_INVESTIGATION.md` là điểm xuất phát (giả thuyết "bug ở `_get_coords`"). File này kiểm tra lại giả thuyết đó
bằng nhãn thật và đi tìm nguyên nhân thật.

## 0. Tóm tắt (cập nhật dần)

- Hiện tượng **có thật và có hệ thống**: khi `imgsz` < 1008, box dự đoán co về góc trên trái (cạnh phải/dưới bị thiếu); khi `imgsz` > 1008 thì box hơi phình. Mức co thay đổi **trơn và đơn điệu theo H = imgsz/14**, đi qua 0 quanh H = 72–90.
- **Không phải do** lưới tọa độ của RPB (4 cách sửa, không cách nào làm phẳng được xu hướng), **không do** độ phân giải nội dung, **không do** token đệm của cửa sổ attention, **không do** RoPE toàn cục hay position embedding tuyệt đối của encoder.
- **Nằm trong decoder** (bước 8: encoder chạy ở H=72 nhưng decoder ở H nhỏ thì co quay lại nguyên vẹn).
- Trong decoder, **sine encoding của box (truy vấn)** là chỗ có tác dụng: nhân tọa độ box với `H/72` trước khi đưa vào nó (`sineq`) làm **độ dốc theo H giảm khoảng 4 lần** (chiều rộng +2.04 → +0.51 %/10 ô) và nén sai số ở **mọi cỡ từ 336 đến 1260** vào khoảng −4.6% … +0.4% (gốc: −13.1% … +1.5%), không đổi gì ở 1008, recall giữ nguyên. Quét số mũ cho quan hệ liều lượng trơn, đơn điệu ở cả 420, 700, 1260. Điểm không của độ co không nằm đúng một số mũ cố định (0.34–1.49 tùy cỡ và chiều), nên `H/72` là xấp xỉ tốt chứ chưa phải công thức chính xác.
- Lưới `71/72` **không phải nguyên nhân gốc**: nó chủ yếu dịch box (tâm) chứ không sửa kích thước; ở 1008 nó là no-op. Nhưng nó cần thiết để căn tâm khi dùng cùng `sineq`.
- **Chốt:** `sineq` với `a = 1` (hệ số `H/72`) + lưới `arange/(H-1)*71/72`. Bước 11 có số liệu tập chưa từng nhìn và engine TensorRT FP16 thật.

## 1. Câu hỏi và tiêu chí đánh giá

Câu hỏi: vì sao box co khi `imgsz` ≠ 1008 (model chỉ được huấn luyện ở 1008 = 72×72 ô patch 14 px), và có sửa được không.

Tiêu chí (theo chỉ đạo của chủ project):
1. Thước đo chính là **sai số chiều rộng/chiều cao theo tỷ lệ phần trăm** (`pred_size / gt_size − 1`) và **xu hướng của nó theo `H`**. Một bản sửa tốt là bản làm xu hướng **phẳng**, dao động quanh 0 trong dải khoảng ±15%; không đòi hỏi sai số bằng 0.
2. **IoU chỉ là tham khảo.**
3. Mọi phép chạy dùng **FP16** (xem mục 2).

## 2. Phương pháp

- Dữ liệu: **coco128** (ảnh COCO có nhãn box), 64 ảnh đầu, 226 box GT có diện tích ≥ 2% ảnh. Không nằm trong repo (`SAM3_test/data/`, git-ignore).
- Prompt: tên class COCO có GT trong ảnh, tối đa 4 class mỗi ảnh ("oracle prompt": đo hình học của box, không đo false positive).
- Dự đoán: score ≥ 0.5, NMS 0.6, box cắt trong ảnh. Ghép cặp với GT cùng class bằng IoU lớn nhất (≥ 0.3); thống kê chỉ dùng cặp IoU ≥ 0.5 (khoảng 190–210 cặp mỗi cấu hình).
- Sai số = dự đoán − GT. Báo cáo theo **patch** (`Δ chuẩn hóa / (14/imgsz)`) và theo **% kích thước box**.
- **Model FP16 thật** (`model.half()`, `scale` pixel giữ FP32), lưới tính ở FP32 rồi ép FP16. FP32 + `torch.autocast` bị loại vì tràn 4 GB VRAM (8.2 s/forward so với 0.2 s); đã kiểm chứng kết quả FP16 thật trùng bản autocast (IoU 0.803 so với 0.802, recall giống nhau, sai số patch trùng tới 0.01).
- Thu thập: `collect_box_pairs.py` lưu **từng cặp (GT, dự đoán)** thô vào `results_pairs/*.json` (mỗi (cỡ, biến thể) một file, chạy lại thì bỏ qua cái đã có). Phân tích offline trên CPU: `analyze_box_pairs.py`.
- Cẩn thận khi đọc: không có nhãn thật hoàn hảo (nhãn COCO có nhiễu); sai số chuẩn kèm sau dấu ±; 64 ảnh không nhiều nên chênh lệch dưới ~1% là không đáng tin.
- Một bẫy đã gặp: `_VisionEncoder.__init__` ghi đè `position_ids`/position embedding **trong chính model dùng chung**, nên các wrapper ở `imgsz` khác nhau dựng từ cùng một `Sam3Model` làm hỏng nhau. Script dựng lại wrapper mỗi khi đổi cỡ.

Ký hiệu: H = imgsz/14 (số ô mỗi cạnh); `d = g_i − e` là khoảng cách từ tọa độ ô `g_i` tới một cạnh `e` của box (`modeling_sam3.py`: `_get_coords` dòng 1718–1719, `_get_rpb_matrix` dòng 1746–1758 của transformers 5.20.2).

## 3. Các bước nghiên cứu

### Bước 1. Kiểm lại các khẳng định cũ

- "Wrapper 1 class và 2 class cho box khác nhau" **không tái hiện được**: box giống hệt (0.00 px) ở 1008 và 420, cho cả wrapper lặp-class cũ lẫn bản gộp-batch (`check_class_independence.py`). Đã xóa khỏi tài liệu cũ.
- "cap = 0.987654321 là tốt nhất" chỉ đúng với một box của ảnh bus (con số này do chủ project chọn tay, không suy ra từ đâu). Trên 91 box, `71/72` tốt tương đương và bằng đúng lưới gốc ở 1008.
- Công thức gốc `arange(0,H)/H` **cũng là code gốc của Meta** (facebookresearch/sam3 `decoder.py`): đây là quy ước lúc huấn luyện, không phải lỗi của bản port `transformers`.

### Bước 2. Sai lệch có thật khi so với nhãn thật không?

Có. Ở 420, bản gốc (đơn vị patch; dương = dự đoán lớn hơn GT):

| 420 | x1 | x2 | y1 | y2 | rộng | cao |
|---|---|---|---|---|---|---|
| `orig` | +0.15 | **−0.71** | +0.03 | **−0.49** | −0.87 (−10.4%) | −0.52 (−5.2%) |

Cạnh trái/trên gần như đúng, cạnh phải/dưới thiếu khoảng 0.5–0.7 patch. Khớp với quan sát của chủ project ("góc trên trái đúng, w và h giảm").

### Bước 3. Có phải do lưới `_get_coords`/RPB không? (4 biến thể)

| Biến thể | Công thức (`g_i`, `d`) |
|---|---|
| `orig` | `g_i = i/H` |
| `71/72` | `g_i = i/(H−1) · 71/72` |
| `1.0` | `g_i = i/(H−1)` |
| `cell` | `g_i = i/H`, `d' = (g_i − e)·H/72` trước mã hóa log |

Kết quả ở 420 (192–194 cặp mỗi biến thể):

| 420 | rộng | cao | x1 | x2 | y1 | y2 | IoU (tham khảo) |
|---|---|---|---|---|---|---|---|
| `orig` | −10.4% | −5.2% | +0.15 | −0.71 | +0.03 | −0.49 | 0.803 |
| `1.0` | −8.2% | −4.0% | +0.48 | −0.17 | +0.34 | 0.00 | 0.805 |
| `71/72` | −9.1% | −3.8% | +0.35 | −0.38 | +0.20 | −0.10 | 0.824 |
| `cell` | −11.1% | −5.1% | +0.35 | −0.63 | +0.36 | −0.20 | 0.807 |

Đọc: các biến thể lưới chủ yếu **dịch cả box** sang phải/xuống dưới (x1 và x2 cùng tăng 0.2–0.3 patch), **kích thước gần như không đổi**. Độ dốc theo `H` của chiều rộng: `orig` +2.04, `71/72` +1.98, `1.0` +1.86, `cell` +2.75 (% mỗi 10 ô): **không biến thể nào làm phẳng**. `cell` (ý tưởng của chủ project: đo khoảng cách theo đơn vị ô) còn làm xấu hơn.

Ở 1008: `71/72` và `cell` **giống hệt `orig`** (chênh tối đa 0.0 trên mọi số, tự kiểm tra của bản vá); `1.0` làm xấu (IoU 0.916 → 0.874, mọi cạnh dịch 0.3–0.6 patch). Lưới kết thúc ở 1.0 chỉ "đỡ" ở cỡ nhỏ, hại ở cỡ native.

Kết luận bước 3: **lưới của RPB không phải nguyên nhân chính.**

### Bước 4. Hình dạng của hiện tượng theo `imgsz` (bản gốc)

| imgsz | H | rộng % | cao % | IoU (ref) |
|---|---|---|---|---|
| 336 | 24 | −13.1 | −7.3 | 0.754 |
| 420 | 30 | −10.4 | −5.2 | 0.803 |
| 504 | 36 | −8.9 | −4.8 | 0.830 |
| 560 | 40 | −7.0 | −3.9 | 0.851 |
| 672 | 48 | −5.6 | −2.9 | 0.875 |
| 700 | 50 | −5.4 | −2.2 | 0.877 |
| 840 | 60 | −2.9 | −1.0 | 0.901 |
| **1008** | 72 | **−1.1** | **−0.2** | 0.916 |
| 1260 | 90 | **+0.6** | **+1.5** | 0.914 |

Độ dốc: chiều rộng +2.04%/10 ô, chiều cao +1.27%/10 ô. Mức co **trơn, đơn điệu theo H, đổi dấu ở trên 72**, nằm trong ±15% ở mọi cỡ nhưng có xu hướng rõ. Chiều cao đổi dấu sớm hơn chiều rộng.

### Bước 5. Do độ phân giải nội dung hay do H? (đối chứng pad)

Nội dung giữ độ phân giải thấp nhưng dán vào góc trên trái của canvas 1008 (xám), chạy ở H=72 với lưới `orig`; tọa độ chia lại về ảnh:

| nội dung | chạy ở lưới tự nhiên (H nhỏ) | dán vào canvas 1008 (H=72) |
|---|---|---|
| 336 | rộng −13.1%, cao −7.3% | **−2.0%, +0.2%** |
| 420 | −10.4%, −5.2% | **−1.3%, +0.0%** |
| 700 | −5.4%, −2.2% | **−1.4%, 0.0%** |

Kết luận: **không phải do vật có ít pixel**; lỗi gắn với **số ô H**. (Dán canvas tốn chi phí của 1008 nên không phải giải pháp "nhẹ".)

### Bước 6. Token đệm của cửa sổ attention? (bị bác)

Giả thuyết: `window_partition` đệm số 0 ở dưới/phải khi H không chia hết cho 24, không có mask, nên làm hỏng đặc trưng ở mép xa. Hai phép thử:
- Sai số theo vị trí (420): x2 lệch **−0.60** cho box nằm hẳn trong cửa sổ đầy đủ đầu tiên (x2 < 0.6), **−0.88, −0.81, −0.69** cho các nhóm còn lại: **đều khắp ảnh**, không dồn về mép đệm.
- H chia hết cho 24 (336: H=24, 672: H=48; không đệm) vẫn co: 336 là cỡ co **mạnh nhất** (−13.1%), 672 nằm đúng trên đường cong.

Kết luận: **bị bác.**

### Bước 7. Encoder: RoPE toàn cục và position embedding tuyệt đối (bị bác)

- `ropecell`: vị trí RoPE của 4 layer toàn cục nhân `H/72` (mỗi ô luôn 1/3 đơn vị như lúc huấn luyện).
- `peinterp`: position embedding = bản đồ 72×72 (lát 3×3 từ 24×24) nội suy bilinear xuống H×H.

| | 420 rộng/cao | 700 rộng/cao | 1008 |
|---|---|---|---|
| `orig` | −10.4% / −5.2% | −5.4% / −2.2% | −1.1% / −0.2% |
| `ropecell` | −10.9% / −5.8% | −5.4% / −2.2% | giống `orig` |
| `peinterp` | −10.5% / −5.7% | −5.5% / −2.6% | giống `orig` |

Kết luận: hai thành phần đó **không phải nguyên nhân** (và đúng như thiết kế, no-op ở 1008).

### Bước 8. Chia đôi: encoder hay decoder? (`encpad`)

Encoder chạy trên canvas 1008 (H=72, nơi nó đúng), cắt feature map về H×H của nội dung, rồi **decoder chạy với hình học H×H** (lưới, sine, RPB theo H×H):

| nội dung | thường (cả hai ở H nhỏ) | cả hai ở H=72 | **encoder H=72, decoder H nhỏ** |
|---|---|---|---|
| 336 | −13.1% / −7.3% | −2.0% / +0.2% | **−13.3% / −7.7%** |
| 420 | −10.4% / −5.2% | −1.3% / +0.0% | **−10.9% / −6.5%** |
| 700 | −5.4% / −2.2% | −1.4% / 0.0% | **−4.8% / −2.4%** |

Ba cỡ cho cùng kết luận: **lỗi nằm trong decoder khi H ≠ 72, không phải ở ViT hay neck.**

### Bước 9. Bên trong decoder: các nguồn thông tin vị trí

Decoder dùng ba nguồn: (a) sine của box → `ref_point_head` → `query_pos` (`encode_boxes`); (b) sine của khóa `fpn_pos_2` (cũng đi vào DETR encoder); (c) RPB (đã loại ở bước 3). Ba biến thể (chạy ở cỡ thường):
- `sinekey`: sine của khóa theo khung 72 (`(i+1)/72`, phần góc trên trái);
- `sineq`: tọa độ box vào `encode_boxes` nhân `H/72`;
- `frame72`: cả hai cộng RPB theo khung 72 (`cell`).

| 420 | cặp | recall@.5 | rộng | cao |
|---|---|---|---|---|
| `orig` | 191 | 0.845 | −10.4% | −5.2% |
| **`sineq`** | **192** | **0.850** | **−3.6%** | **−0.7%** |
| `frame72` | 130 | 0.575 | −1.6% | −1.0% |
| `sinekey` | 124 | 0.549 | −17.1% | −11.3% |

| 336 | cặp | recall@.5 | rộng | cao |
|---|---|---|---|---|
| `orig` | 169 | 0.748 | −13.1% | −7.3% |
| **`sineq`** | **175** | **0.774** | **−4.6%** | **−2.3%** |

Đọc:
- **`sineq` một mình** xóa phần lớn độ co ở cả hai độ phân giải, recall giữ nguyên hoặc nhỉnh hơn.
- **`sinekey` phá hỏng phát hiện** (recall 0.55): khóa và truy vấn lệch khung nhau; `frame72` hết co nhưng cũng sụp recall vì có `sinekey` bên trong. Kết luận: **không động vào sine của khóa.**
- Ở 1008, cả ba biến thể giống hệt `orig` (tự kiểm tra, chênh 0.0).

### Bước 10. `sineq` có làm phẳng đường cong ở mọi cỡ không?

**(a) `sineq` ở 9 cỡ** (so với `orig`; độ dốc = % mỗi 10 ô):

| | độ dốc rộng | độ dốc cao | dải rộng | dải cao |
|---|---|---|---|---|
| `orig` | +2.04 | +1.27 | [−13.1, +0.6]% | [−7.3, +1.5]% |
| **`sineq`** | **+0.51** | **+0.31** | **[−4.6, −1.1]%** | **[−2.3, +0.4]%** |

| imgsz | H | `orig` rộng / cao | `sineq` rộng / cao | IoU `orig` → `sineq` (ref.) |
|---|---|---|---|---|
| 336 | 24 | −13.1 / −7.3 | −4.6 / −2.3 | 0.754 → 0.773 |
| 420 | 30 | −10.4 / −5.2 | −3.6 / −0.7 | 0.803 → 0.824 |
| 504 | 36 | −8.9 / −4.8 | −3.3 / −1.4 | 0.830 → 0.851 |
| 560 | 40 | −7.0 / −3.9 | −2.6 / −0.8 | 0.851 → 0.872 |
| 672 | 48 | −5.6 / −2.9 | −2.6 / −0.4 | 0.875 → 0.890 |
| 700 | 50 | −5.4 / −2.2 | −2.5 / −0.2 | 0.877 → 0.895 |
| 840 | 60 | −2.9 / −1.0 | −1.4 / −0.3 | 0.901 → 0.910 |
| 1008 | 72 | −1.1 / −0.2 | −1.1 / −0.2 (giống hệt) | 0.916 → 0.916 |
| 1260 | 90 | +0.6 / +1.5 | −1.2 / +0.4 | 0.914 → 0.911 |

Đọc: sai số ở mọi cỡ được nén vào dải hẹp quanh 0, **kể cả phía trên 72** (ở 1260 `orig` phình +0.6%/+1.5%, `sineq` không còn đổi dấu), và đường cong gần phẳng từ 700 trở lên. Chiều rộng vẫn còn dư co khoảng 2–3% ở giữa (504–700). Tất cả nằm trong dải ±15% chấp nhận được.

**(b) Quét số mũ `sineq<a>`** (hệ số nhân tọa độ box là `(H/72)^a`, `a=1` là `sineq`):

| `a` | hệ số `(H/72)^a` ở 420 / 700 / 1260 | 420 rộng / cao | 700 | 1260 |
|---|---|---|---|---|
| 0 (`orig`) | 1 / 1 / 1 | −10.4 / −5.2 | −5.4 / −2.2 | +0.6 / +1.5 |
| 0.5 | 0.645 / 0.833 / 1.118 | −7.0 / −2.8 | −3.7 / −1.3 | −0.3 / +0.9 |
| **1** | 0.417 / 0.694 / 1.250 | −3.6 / −0.7 | −2.5 / −0.2 | −1.2 / +0.4 |
| 1.5 | 0.269 / 0.579 / 1.398 | +1.8 / +2.5 | −1.0 / +0.6 | −1.9 / 0.0 |
| 2 | 0.174 / 0.482 / 1.562 | +9.1 / +5.9 | (chưa chạy) | (chưa chạy) |

- Quan hệ **đơn điệu và trơn** ở cả ba cỡ (tăng `a` luôn làm box lớn dần), nên đây là một tham số điều chỉnh thật, không phải trùng hợp. Recall ở 420 gần như không đổi theo `a` (0.832–0.858).
- **Điểm không của độ co** (nội suy tuyến tính theo `a`): 420 rộng ≈ 1.33, cao ≈ 1.11; 700 cao ≈ 1.11, rộng chưa đổi dấu trong khoảng đã quét; 1260 rộng ≈ 0.34, cao ≈ 1.49. Không rơi đúng một `a` cố định, nhưng `a = 1` cho kết quả phẳng nhất khi xét cả ba cỡ cùng lúc (−3.6 / −2.5 / −1.2% rộng, −0.7 / −0.2 / +0.4% cao).

### Bước 11. Chốt bản vá, kiểm chứng tập chưa nhìn và engine thật

**(a) Tổ hợp `sineq^a` × lưới** (64 ảnh đầu, trung bình trên các cỡ; `|rộng|`/`|cao|` là sai số tuyệt đối trung bình theo %, `dốc` là %/10 ô):

| biến thể | cỡ | \|rộng\| | \|cao\| | dốc | lệch tâm x/y (patch) | IoU (ref.) |
|---|---|---|---|---|---|---|
| `orig` | 9 | 6.11 | 3.23 | +2.04 | 0.16 / 0.16 | 0.858 |
| `71/72` | 5 | 5.12 | 2.15 | +1.98 | 0.03 / 0.06 | 0.875 |
| `sineq` (a=1) | 9 | 2.55 | 0.74 | +0.51 | 0.20 / 0.19 | 0.871 |
| **`sineq+71/72` (a=1)** | 7 | 1.88 | 0.26 | +0.24 | 0.05 / 0.06 | 0.893 |
| `sineq1.2+71/72` | 7 | 1.03 | 0.46 | −0.15 | 0.06 / 0.06 | 0.894 |
| `sineq1.4+71/72` | 7 | 1.36 | 1.00 | −0.65 | 0.07 / 0.06 | 0.893 |

`sineq` sửa **kích thước**, `71/72` sửa **tâm**; hai tác dụng độc lập nên cộng được. Chọn `a = 1`: là hệ số "khung lưới 72" (tọa độ box được đo theo đơn vị ô như lúc huấn luyện, không phải một số tinh chỉnh), độ dốc dương nhỏ (+0.24), và `a = 1.2` chỉ hơn về sai số trung bình nhưng bắt đầu dốc âm (box phình ở H lớn), tức là đổi chỗ chứ không hết. Không có suy diễn lý thuyết cho riêng số mũ.

**(b) Tập chưa từng nhìn** (ảnh 65–128, không dùng để chọn gì):

| biến thể | cỡ | \|rộng\| | \|cao\| | dốc | lệch tâm x/y | IoU |
|---|---|---|---|---|---|---|
| `orig` | 7 | 5.86 | 3.44 | +1.78 | 0.14 / 0.20 | 0.856 |
| `71/72` | 7 | 5.27 | 2.85 | +1.51 | 0.09 / 0.07 | 0.868 |
| **`sineq+71/72`** | 7 | 1.93 | 0.84 | −0.07 | 0.05 / 0.09 | 0.888 |

Kết quả giữ nguyên, độ dốc về ≈ 0.

**(c) Tích hợp.** Cùng một khối "Fixes for running at imgsz != 1008" trong `SAM3Fixed/export.py` và `SAM3Open/export.py` (mỗi file tự chứa): `Sam3DetrDecoder._get_coords` → `arange/(H-1)*71/72`; `Sam3SinePositionEmbedding.encode_boxes` nhân box với `box_scale = (H/72)^1.0`, `H` lấy từ feature map lúc export (hằng số trong đồ thị). Ở 1008 hệ số = 1 và lưới = gốc. Eager prod == `sineq+71/72` của thí nghiệm (chênh 0.0000); Fixed và Open khớp nhau trong sai số fp16.

**(d) Engine TensorRT FP16 thật** (`eval_engine_person.py`, chỉ lớp `person`, 420):

| engine | tập | box | recall@.5 | rộng | cao | IoU |
|---|---|---|---|---|---|---|
| cũ (không `sineq`) | 64 đầu | 54/59 | 0.915 | −10.9±1.1% | −2.6% | 0.828 |
| **mới** | 64 đầu | 57/59 | 0.966 | −3.9±1.2% | −0.5% | 0.865 |
| cũ | chưa nhìn | 45/47 | 0.957 | −10.4±1.6% | −2.5% | 0.835 |
| **mới** | chưa nhìn | 45/47 | 0.957 | −1.8±1.4% | +0.3% | 0.859 |

Khớp bảng eager: bản vá được nướng đúng vào ONNX và TensorRT FP16. Cỡ mẫu nhỏ (47–59 box) nên chỉ đủ thấy hướng.

## 4. Câu hỏi còn mở

- **Cơ chế.** Vì sao nhân tọa độ box với `H/72` vào sine encoding của truy vấn làm kích thước box đúng hơn? Hiện chỉ có bằng chứng thực nghiệm (liều lượng trơn, đúng hướng ở 3 cỡ, no-op ở 1008), chưa giải thích được. Điểm không của độ co thay đổi theo cỡ và chiều (a ≈ 0.34–1.49), nên `H/72` là xấp xỉ tốt chứ chưa chắc là công thức đúng.
- Phần dư co chiều rộng ~2–3% ở 504–840 sau `sineq`; và x1, y1 hơi âm (−0.16, −0.30 patch ở 336). Có thể kết hợp thêm một phép chỉnh nhỏ, cần thử.
- Kiểm chứng trên dữ liệu nhiều hơn (mới 64 ảnh, nhãn COCO có nhiễu) và với box nhỏ hơn 2% diện tích ảnh (đã loại ở đây).
- Khoảng `a` cao hơn 1.5 mới chỉ chạy ở 420.

## 5. Tái hiện

Từ thư mục gốc repo, cần `HF_TOKEN` trong `.env` và coco128 ở `SAM3_test/data/coco128`:

```bash
# thu thập (mỗi lần chạy lưu từng (cỡ, biến thể) ngay khi xong)
python -u SAM3_test/collect_box_pairs.py --sizes 420 --variants orig,71/72,1.0,cell
python -u SAM3_test/collect_box_pairs.py --sizes 420in1008 --variants orig,encpad   # đối chứng pad, chia đôi
python -u SAM3_test/collect_box_pairs.py --sizes 420 --variants ropecell,peinterp,sinekey,sineq,frame72
# phân tích (CPU)
python SAM3_test/analyze_box_pairs.py
```

Biến thể và cú pháp cỡ: xem docstring của `collect_box_pairs.py`. Kết quả thô: `SAM3_test/results_pairs/*.json`; script chạy chuỗi: `results_pairs/run_*.sh`.
