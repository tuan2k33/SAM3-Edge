# Cắt sub-block SAM3: tóm tắt (không theo đuổi)

Thí nghiệm: tắt sub-block (đặt output = 0, residual đi thẳng) rồi đo recall/IoU. coco128, prompt oracle, FP16 thật, lưới 71/72, PyTorch eager, RTX 3050 Ti 4 GB, imgsz 420 và 1008, tách tập chọn / held-out. Script và log đã xoá vì không dùng; kết luận giữ lại ở đây.

106 sub-block = ViT 64 (32 lớp x attn, mlp) + encoder 18 (6 x self, cross-text, mlp) + decoder 24 (6 x self, text, img, mlp).

## Kết luận

Cắt được khoảng 24/106 mà recall/IoU gần như không đổi ở cả 420 và 1008, nhưng tốc độ chỉ tăng thêm 9-15% (chưa đo TensorRT). Hạ imgsz 1008 -> 420 nhanh hơn ~2.9 lần (650 -> 225 ms), lớn hơn nhiều. Quyết định: không đưa vào exporter.

| Tầng | Thời gian 420 / 1008 | Cắt được | Không cắt |
|---|---|---|---|
| ViT | 71% / 76% | 6: `vitattn:25,26,27`, `vitmlp:28,29,30` (an toàn tới ~10-12/64) | lớp 0-8, 4 attention global (7/15/23/31) |
| Encoder | 7% / 12% | 6: `encself:0,1` + `enccross:2-5` (hoặc `encself:0` + `enccross:1-5`) | `enccross:0`, `encself:2-5`, `encmlp:1-5`; chết hẳn ở 14/18 |
| Decoder | 21% / 11% | 12: `decself:4,1,3`, `dectext:4,2,1,0`, `decimg:5,4`, `decmlp:4,0,5` (0 mất recall) | `decimg:0`, `decself:0`; sập ở 20/24 |

Tốc độ (PyTorch eager, nhiễu đo ~6% ở 420, ~0.6% ở 1008): ViT 6 -13% / -5.9%; ViT 6 + encoder 6 + decoder 12 -14.8% / -9.1% (420 / 1008).

## Vì sao decoder cắt được nhiều nhất

Chỉ decoder có cụm sub-block trùng hướng: `dectext:0,1,2` + `decself:1,2` (VIF 18-38). Cắt cả cụm tốn ~49% so với cộng sai số từng cái. ViT và encoder gần như độc lập (VIF ~1-2): cắt k cái tốn ~k lần, không có giảm giá. Ma trận tương quan chỉ tìm được cụm, không dự đoán được sai số khi cắt nhiều tầng cùng lúc (sai tới ~7x), phải đo thật.

## Cảnh báo

- Recall tăng ở vài set 1008 held-out đi kèm số box dự đoán tăng, chưa phải cải thiện thật. IoU gần như không đổi.
- 1 box GT ~ 0.44% recall, chênh dưới 0.02 là nhiễu.
- Đường cong "cắt bao nhiêu thì sập" chỉ đo ở 420, chưa đo ở 1008.
- Chưa có precision/AP, chưa đo TensorRT, chưa thử thay 0 bằng trung bình `f(x)` (mean ablation).
