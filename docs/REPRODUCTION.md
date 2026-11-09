# Báo cáo reproduce PATRA (ICML 2026)

Reproduce Table 1 của [PATRA](https://arxiv.org/abs/2602.23161) cho PATRA-7B và Qwen2.5-7B, chỉ bằng inference với checkpoint công bố, không train lại.

**Kiểm toán độc lập ngày 07/11:** Đã đối chiếu lại nguồn công khai, chấm lại toàn bộ predictions và kiểm tra đường đi của chuỗi bằng code gốc trên CPU. Chưa chạy lại inference 7B trên GPU trong lần kiểm toán này.

## Kết luận

**Chưa reproduce được Table 1.** Với checkpoint, dữ liệu và script công bố:

- **Accuracy không tái lập được:** thấp hơn paper từ 10,19 đến 25,86 điểm ở cả 4 task. Tính chung, chạy lại đúng 521/1.097 câu (47,49%), paper tương đương 657/1.097 câu (59,89%).
- **ROUGE-L khá sát paper:** lệch từ +0,49 đến +1,72.
- **Chưa tìm thấy lỗi trong pipeline `official` giải thích khoảng lệch:** code, dữ liệu, cách gọi inference, ghép chuỗi và chấm điểm khớp bản công bố. Chưa đối chiếu response sinh trực tiếp bằng script gốc trên toàn tập, và chưa xác nhận checkpoint này là bản tạo Table 1.
- **Chưa đo được lợi ích accuracy từ đường chuỗi trên câu có nhãn:** thay `ts_encoder` và `align` bằng weight ngẫu nhiên không làm accuracy giảm có ý nghĩa thống kê. Kết quả này chưa áp dụng cho câu open-ended.

Báo cáo này chỉ nói về bản công bố, không khẳng định số trong paper là sai.

## Thiết lập

| Thành phần | Giá trị |
|---|---|
| Code | `decisionintelligence/PATRA@ed19f18`, không sửa |
| Model | `DecisionIntelligence/PATRA-7B@5cf8191e`, `Qwen/Qwen2.5-7B@d1497293` (bản base) |
| Dữ liệu | `DecisionIntelligence/PATRA-EVAL@c79132c5`: 1.097 câu có nhãn, 1.868 câu open-ended |
| Môi trường chính | Python 3.11, torch 2.7.1+cu128, transformers 4.52.4, RTX PRO 6000 Blackwell |
| Sinh câu trả lời | `do_sample=False`, `max_new_tokens=512`, batch size 1 |

Ngưỡng đạt do mình đặt, không phải của paper: accuracy lệch tối đa ±2 điểm, ROUGE-L tối đa ±1,5.

Riêng baseline Qwen: paper không nói cách đưa chuỗi vào, nên mình dùng danh sách giá trị dạng text. Vì vậy accuracy của Qwen không nhất thiết khớp paper.

## Kết quả Table 1

Số trong ngoặc là số của paper.

| Task | PATRA accuracy | Lệch | PATRA ROUGE-L | Lệch | Qwen accuracy | Qwen ROUGE-L |
|---|---|---|---|---|---|---|
| Comprehension | 30,17 (56,03) | −25,86 | 27,39 (25,67) | +1,72 | 41,38 (42,24) | 18,74 (18,77) |
| Recognition | 54,07 (64,69) | −10,62 | 26,02 (25,46) | +0,56 | 48,83 (45,51) | 10,60 (10,32) |
| Reasoning | 32,43 (44,59) | −12,16 | 29,04 (27,36) | +1,68 | 29,73 (36,48) | 17,56 (18,72) |
| Prescience | 42,59 (52,78) | −10,19 | 27,55 (27,06) | +0,49 | 38,89 (26,85) | 10,96 (10,67) |

- **PATRA:** accuracy không đạt ngưỡng ở cả 4 task. ROUGE-L đạt ở 2 task, 2 task còn lại vượt ngưỡng nhẹ theo hướng cao hơn paper.
- **Qwen:** ROUGE-L đạt ở cả 4 task, accuracy chỉ đạt ở Comprehension.
- **PATRA so với Qwen:** lợi thế ROUGE-L của paper vẫn thấy rõ, nhưng lợi thế accuracy thì không. Tính chung, PATRA đạt 47,49%, Qwen đạt 44,48% (McNemar p = 0,12). Nếu tính thêm các câu chỉ sai định dạng, hai bên gần như hòa: 523 và 522 câu đúng.

Nguồn: [tables/table1_repro.csv](../results/tables/table1_repro.csv), [analysis/paired.csv](../results/analysis/paired.csv).

## Đối chứng 1: có phải mình chạy sai không?

| Khả năng | Cách kiểm tra | Kết quả | Kết luận |
|---|---|---|---|
| Sai script hoặc cách chấm | Smoke bằng script gốc; kiểm toán CPU gọi lại hàm chấm gốc trên predictions đã lưu | Không lệch điểm từng câu và tổng hợp; lời gọi processor/generate của worker cũng khớp trên 2.965 mẫu `official` khi replay bằng model kiểm thử | Không thấy lỗi; replay chưa thay thế đối chiếu inference GPU nguyên bản trên toàn tập |
| Sai dữ liệu công bố | Tải lại bốn file HF đúng revision, so hash và từng record | Hash khớp registry; không lệch câu hỏi, nhãn hay chuỗi theo điều kiện | Đúng dữ liệu README trỏ tới; chưa có hash dữ liệu tạo Table 1 |
| Sai model công bố | Kiểm tra revision HF; so tensor với `Qwen2.5-7B-Instruct` | Đúng repo/revision README; backbone có thay đổi phù hợp LoRA | Đúng bản công bố; chưa xác thực đây là checkpoint tạo Table 1 |
| Lỗi load weight, ghép chuỗi | Probe GPU cũ so weight đã load; audit CPU mới chạy merge gốc | 61/61 tensor trong probe khớp; merge khớp 2.965/2.965 mẫu, gồm 34 mẫu nhiều chuỗi | Không thấy lỗi nạp module chuỗi hoặc đặt patch |
| Sai phiên bản thư viện | Chạy lại với bản `requirements.txt` resolve ra (`transformers 4.46.3`, `tokenizers 0.20.3`) | **1.097/1.097 câu trả lời giống hệt từng ký tự** | Không thấy ảnh hưởng của hai bản Transformers trên câu có nhãn; torch/GPU vẫn giữ nguyên |
| Khác prompt hoặc tham số sinh | Thử 4 biến thể | Xem bảng dưới | Chỉ giải thích được một phần |

| Biến thể (1.097 câu có nhãn) | Accuracy | p so với `official` |
|---|---|---|
| `official` | 47,49 | |
| Định dạng GRPO lúc train (`grpo_format`) | 52,05 | 0,003 |
| Xuống dòng theo `chat_template` (`prompt_chatml`) | 51,32 | 0,0002 |
| Prompt theo Appendix B của paper (`prompt_paper`) | 51,14 | 0,005 |
| Tắt `repetition_penalty` (`no_repetition_penalty`) | 48,31 | 0,55 |
| **Paper** | **59,89** | |

Prompt có ảnh hưởng thật, nhưng biến thể tốt nhất vẫn cách paper gần 8 điểm. Riêng Comprehension cao nhất chỉ đạt 37,93 so với 56,03. Đây là các lần thử để chẩn đoán, không thay thế dòng `official`, và p chưa hiệu chỉnh cho nhiều phép so sánh.

`grpo_format` thay cả prompt lẫn cách trích đáp án: lấy thẻ `<answer>` cuối cùng thay vì `\boxed{...}` đầu tiên. `prompt_paper` thay cả nội dung system prompt lẫn xuống dòng ChatML. Vì vậy đây chưa phải các phép thử cô lập riêng từng yếu tố.

Nguồn: [env/smoke.json](../results/env/smoke.json), [env/pip_freeze_readme.txt](../results/env/pip_freeze_readme.txt), [analysis/ts_probe_summary.json](../results/analysis/ts_probe_summary.json).

## Đối chứng 2: model có thực sự dùng chuỗi không?

Tất cả đều chạy trên 1.097 câu có nhãn, so với `official` (47,49%).

| Đối chứng | Thay đổi | Accuracy | p | Câu trả lời giống hệt |
|---|---|---|---|---|
| `ts_zero` | Thay chuỗi bằng toàn số 0 | 48,13 | 0,19 | 946/1.097 |
| `reinit_modules` | Thay `ts_encoder` và `align` bằng weight ngẫu nhiên mới | 48,77 | 0,21 | 441/1.097 |
| `no_align` | Bỏ qua `align` | 47,95 | 0,71 | 423/1.097 |
| `no_causal_mask` | Bỏ mask trong attention của `align` | 48,04 | 0,59 | |

Không đối chứng nào làm accuracy giảm có ý nghĩa thống kê. Đo trực tiếp bên trong model cũng cho cùng kết quả:

- **Đặc trưng chuỗi bị làm phẳng:** đổi chuỗi thành toàn 0 làm output của encoder đổi 99,27%, nhưng sau `align` chỉ còn đổi 12,35%.
- **Đáp án gần như không đổi:** với phần lý luận giữ nguyên, token đáp án giữ nguyên ở 99,82% câu khi dùng chuỗi 0 và 99,73% khi dùng chuỗi của câu khác.
- **Số trích dẫn không bám chuỗi:** phép kiểm tra cập nhật trong [CODE_REVIEW.md](CODE_REVIEW.md) cho tỷ lệ khớp giá trị thô là 3,44% với PATRA và 94,25% với Qwen. Tỷ lệ PATRA khớp số thô của chuỗi khác là 4,11%, nên chưa thấy khả năng trích số bám đúng chuỗi.
- **LAT không đọc câu hỏi:** do mask, LAT chỉ thấy 10 token đầu của prompt, nên gần như không đổi giữa các câu. Tuy vậy, bỏ mask cũng không làm thay đổi accuracy có ý nghĩa.
- **Weight của `ts_encoder` và `align` rất gần phân phối khởi tạo:** std từ 0,0199 đến 0,0200, 29/29 bias đúng bằng 0, không có thành phần nào vượt ngưỡng nhiễu của phép kiểm tra. LoRA không đổi bias; các thống kê này chưa phân biệt được chưa train, cập nhật ít hoặc cập nhật LoRA nhỏ so với khởi tạo. Vì `PATRA-init` chưa công bố nên không đo trực tiếp được mức thay đổi.

Riêng `ts_zero` mất 7 câu đúng và được thêm 14 câu đúng, tăng ròng 7/1.097 câu (p = 0,1892). Chưa đủ bằng chứng nói bỏ chuỗi giúp model tốt hơn; p lớn cũng chưa chứng minh hai điều kiện tương đương. Số 99,82% trong probe ở trên giữ nguyên phần lý luận rồi đo token đáp án tiếp theo. Khi sinh lại toàn response, nhãn trích ra giữ nguyên ở 1.061/1.097 câu; đây là hai phép đo khác nhau.

Để so sánh, Table 6 của paper báo Reasoning giảm từ 44,59 xuống 28,37 khi bỏ alignment. Hai phép này không hoàn toàn tương đương, vì paper train lại còn ở đây chỉ tắt module lúc chạy.

Nguồn: [analysis/paired.csv](../results/analysis/paired.csv), [analysis/ts_probe_summary.json](../results/analysis/ts_probe_summary.json).

## Có phải limitation của repo không?

**Có, nhưng là limitation của bản công bố.** Những gì đo được cho phép nói bản công bố không tái lập được số trong paper, chưa đủ để nói paper sai.

**Đã xác nhận:**

1. **Không tái lập được accuracy** từ checkpoint, dữ liệu và script công bố.
2. **Module chuỗi không tạo lợi ích accuracy đo được** trên câu có nhãn, với checkpoint công bố.
3. **Không tự kiểm chứng được bằng cách train lại:**
   - thiếu `train/src/llamafactory/train/grpo/trainer.py`, trong khi `workflow.py` có import file này
   - `PATRA-init` chưa công bố
   - `dataset_info.json` trỏ tới `sft-train.jsonl`, còn file trên Hugging Face là `sft.jsonl`
   - `GRPOConfig` đặt cứng learning rate 5e-6 và 2 epoch, khác giá trị 1e-5 và 5 epoch trong `train_patra_grpo.sh`
4. **Dữ liệu ít hơn mô tả:** paper nói khoảng 200.000 mẫu, repo có 27.906 train và 2.965 eval. Issue #1 cũng hỏi về việc này và chưa có trả lời.
5. **Quy trình đánh giá chưa được ghi rõ:** prompt trong script khác Appendix B, và `repetition_penalty` 1,05 được áp dụng ngầm từ `generation_config.json`, paper không nhắc tới. Model trên Hugging Face không có model card.
6. **Tập eval có nhãn chưa sạch:**
   - 38 câu MC không trả lời đúng được: 19 câu không có lựa chọn, 19 câu có gold nằm ngoài lựa chọn
   - với câu hỏi về mean, gold chỉ khớp đáp án tính từ chuỗi ở 17/43 câu; với median là 5/9

   Bỏ 38 câu trên, accuracy chỉ tăng khoảng 1–3 điểm, nên đây không phải nguyên nhân chính của khoảng lệch.

**Chưa kết luận được, cần tác giả xác nhận:**

- Checkpoint công bố có đúng là bản đã tạo ra số trong paper không.
- Paper có đánh giá bằng đúng script và prompt đã công bố không. Script eval được thêm ngày 01/09/2026, sau checkpoint ngày 20/06/2026.

Việc repo chỉ có 1 issue và không có báo cáo nào về accuracy không chứng minh được điều gì, vì mức sử dụng còn rất thấp (10 sao, 0 fork).

## Giới hạn của báo cáo này

- **Chỉ inference:** không train lại, nên không so trực tiếp được với ablation trong paper.
- **Câu có nhãn:** các đối chứng về chuỗi chưa chạy trên open-ended, nên chưa biết lợi thế ROUGE-L của PATRA có đến từ chuỗi không.
- **Baseline:** Qwen là bản base, trong khi PATRA xây trên bản Instruct. Có 190/2.965 câu của Qwen bị cắt ở 512 token.
- **Số của paper:** số câu đúng được quy đổi từ phần trăm trong Table 1.
- **Đối chiếu nhãn:** chỉ kiểm được các câu tính được từ chuỗi, không đại diện cho toàn tập.

## Chạy lại và cấu trúc thư mục

Mở [notebooks/reproduce.py](../notebooks/reproduce.py) trên molab, chọn run rồi bấm **Run**, sau đó tải zip kết quả. Smoke test và probe trong `results/env/` và `results/analysis/` là bản ghi của pipeline đầu tiên, không nằm trong notebook này.

```text
PATRA/        repo gốc, không sửa
notebooks/    notebook chạy lại model và các đối chứng trên molab
scripts/      script phân tích chạy trên CPU
results/
  runs/          câu trả lời và điểm từng câu của mỗi run
  tables/        bảng so với paper
  analysis/      các phân tích và đối chứng
  env/           kết quả smoke và pip freeze của run `env_readme`
```
