# Code review PATRA: kết quả kiểm tra 5 ý

File này là danh sách bằng chứng theo từng ý. Phần tổng hợp nằm ở [IMPLEMENTATION_INSIGHTS.md](IMPLEMENTATION_INSIGHTS.md).

Mọi số trong file này lấy từ [results/analysis/code_review_checks.json](../results/analysis/code_review_checks.json), do [scripts/code_review_checks.py](../scripts/code_review_checks.py) tạo ra. Ngoại lệ: mức nền 66,22% và 34 mẫu nhiều chuỗi lấy từ bảng phân nhóm của pipeline đầu tiên (không còn trong repo); các số của paper đọc trực tiếp từ PDF arXiv 2602.23161.

Phiên bản: `PATRA@ed19f18`, `PATRA-7B@5cf8191e`, `PATRA-TRAIN@511dcecc`. `modeling_qwen2.py` và `processing_qwen2_ts.py` nằm trong repo model trên Hugging Face, không nằm trong repo GitHub.

## Tóm tắt

| Ý | Kết luận | Mức quan trọng |
|---|---|---|
| 5. Bỏ `offset` và `scale` | Xác nhận. Đáp án train trích số thô mà model không thể biết | Cao nhất |
| 3. Reward [0, 2] | Xác nhận, mạnh hơn giả thuyết ban đầu. Với G = 2, độ lớn reward gần như không còn tác dụng | Cao |
| 2. Phân rã với 1–2 patch | Xác nhận. Trend/seasonal chỉ còn là tổng và hiệu của hai token | Cao |
| 4. PATRA và TimeOmni ở Reasoning | Xác nhận. Lệch 3 câu, không thể có ý nghĩa thống kê với mọi cách ghép cặp | Trung bình |
| 1. Trend và seasonal ghép chéo | Xác nhận lệch Eq. 7, nhưng chỉ là đổi tên, không phải lỗi chức năng | Thấp |

## Ý 5: `offset` và `scale` bị loại khỏi prompt

**Vị trí:** `processing_qwen2_ts.py:36-44`. Dòng ghi offset và scale vào prompt đã bị comment.

**Bằng chứng:**

- Model không phân biệt được hai chuỗi chỉ khác nhau về mức nền. Với chuỗi có độ lệch từ 3 trở lên, nó cũng không phân biệt được biên độ.
- Sau khi mã hóa, trung bình chuỗi luôn bằng 0, nên encoder không có tín hiệu nào cho câu hỏi về mean.
- Thông tin bị mất không nhỏ: 67,11% chuỗi eval có |offset| ≥ 1 và 64,83% có scale > 1 (train: 69,79% và 66,92%).
- Không câu nào trong tập eval hay `grpo.jsonl` có "Value Offset" trong prompt. Chuỗi trong `sft.jsonl` khớp bản đã chuẩn hóa ở 27.807/27.906 mẫu.

| Nguồn | Số được trích | Bằng giá trị thô của chuỗi đó | Bằng giá trị thô của chuỗi khác (đối chứng) | Gần giá trị đã chuẩn hóa |
|---|---|---|---|---|
| Đáp án train SFT | 53.021 | **77,02%** | 2,71% | 2,35% |
| Đáp án train GRPO, open-ended | 39.165 | **75,11%** | 2,68% | 2,38% |
| Đáp án eval, open-ended | 4.399 | 76,40% | 2,66% | 3,80% |
| PATRA, open-ended | 4.771 | **3,44%** | 4,11% | 3,29% |
| Qwen, open-ended | 26.639 | 94,25% | 2,68% | 3,27% |

**Diễn giải:** model được train để viết ra các số mà đầu vào của nó không chứa, nên việc tự nghĩ ra số là hệ quả dễ hiểu. Số trích dẫn của PATRA chỉ ở mức ngẫu nhiên, và 375/1.868 câu trả lời open-ended có số −0,49.

**Về accuracy, bằng chứng yếu hơn:**

- Câu có số trong đề: PATRA 44,32%, Qwen 50,42% (p = 0,067).
- Câu không có số trong đề: PATRA 49,05%, Qwen 41,58% (p = 0,0015).
- p chưa hiệu chỉnh cho nhiều phép so sánh. Qwen đọc trực tiếp số thô nên sự chênh lệch này không chỉ do offset và scale.
- Nhóm gần như không mất thông tin chỉ có 67 câu, quá ít để so sánh.

**Chưa đo:** tác động nhân quả. Điều kiện `value_context` có trong notebook nhưng chưa chạy, và checkpoint hiện tại không được train với thông tin đó. Phép thử đúng là train lại, mà việc này hiện chưa làm được.

## Ý 3: Reward [0, 2] và Eq. 14

**Vị trí:** `train/src/llamafactory/train/grpo/workflow.py:89` (`num_generations=2`), `:143-160` (task reward), `:164-180` (format reward); `grpo_config.py:463` (`scale_rewards` mặc định True); `trl/trainer/grpo_trainer.py:1313-1322`.

**Bằng chứng:**

1. **Eq. 14 đã khử tỷ lệ reward.** Nhân toàn bộ reward của group với 0,5 chỉ làm advantage đổi tối đa 0,0002 (group 8 mẫu, chỉ do ε). Vì vậy đưa reward về cùng khoảng không thể cân bằng các task thông qua độ lớn gradient như Section 4.2.2 và Appendix E.2.1 nêu.
2. **Với G = 2, advantage gần như chỉ còn là dấu.** Khi hai reward khác nhau, advantage xấp xỉ ±0,707:

   | Chênh lệch reward | 0,001 | 0,01 | 0,1 | 1,0 | 2,0 |
   |---|---|---|---|---|---|
   | Advantage | 0,6195 | 0,6972 | 0,7061 | 0,7070 | 0,7071 |

   Chênh ROUGE-L rất nhỏ gần như được tính bằng chênh lệch giữa đúng và sai.
3. **Tác dụng thật của [0, 2] là cân bằng task với format, không phải task với task:**

   | Tình huống (A so với B) | Task reward [0, 2] | Task reward [0, 1] |
   |---|---|---|
   | A đúng, format 0; B sai, format đủ 0,75 | A thắng | Hòa, advantage 0 |
   | A hơn 0,10 ROUGE-L; B hơn một thẻ (0,125) | A thắng | B thắng |
   | A hơn 0,05 ROUGE-L; B hơn một thẻ (0,125) | B thắng | B thắng |

4. **Format reward không nằm trong [0, 0,75] như Appendix E.2.1:**
   - Câu trả lời 800 ký tự kết thúc ngay sau `</answer>` nhận −0,461, vì penalty tính trên cả văn bản.
   - Thẻ `<task_type>` do chính prompt yêu cầu đặt sau `</answer>` lại bị trừ điểm: ví dụ `multiple_choice` chỉ đạt 0,675.

**Giả thuyết chưa đo:** với câu có nhãn, hai mẫu cùng đáp án và cùng format cho advantage 0, còn ROUGE-L liên tục nên câu open-ended gần như luôn có tín hiệu. Nếu đúng, mất cân bằng có thể ngược với hướng paper mô tả. Cần log train để kiểm tra.

**Giới hạn:**

- `trainer.py` của GRPO không được công bố. Code trên dựa vào bản TRL đi kèm trong repo, trong khi `workflow.py` truyền `m_processor` mà bản này không nhận, nên trainer thật có khác.
- Paper không nêu G. Số 2 lấy từ `workflow.py`.
- Repo không có tùy chọn cho "Original Reward" của Table 5. Vì vậy phần này không kết luận Table 5 sai, chỉ nói cơ chế được nêu trong paper không khớp với config công bố.

## Ý 2: Phân rã trend/seasonal với 1–2 patch

**Vị trí:** `modeling_qwen2.py:137` (số patch), `:172-192` (moving average), `:344` (kernel mặc định 25, `config.json` không ghi đè).

**Bằng chứng:**

- Kernel 25 token tương đương 400 điểm gốc, trong khi chuỗi dài nhất là 30 điểm ở eval và 60 điểm ở train.
- Eval: 11,6% chuỗi có 1 patch, 88,4% có 2 patch, không chuỗi nào hơn. Train GRPO: chỉ 7/28.834 chuỗi hơn 2 patch.
- **1 patch:** phần dư bằng 0 và trend trùng input, nên nhánh dư không mang thông tin chuỗi.
- **2 patch:** phần dư của hai token ngược dấu và bằng ±0,48 lần hiệu hai token. Hai token trend chỉ cách nhau 0,04 lần hiệu đó. Kết quả thực chất là trung bình và hiệu của hai token.

**Diễn giải:** Latent Decomposition không tách được trend hay seasonal trên dữ liệu công bố. Hình dạng bên trong mỗi patch 16 điểm đã bị MLP nén trước đó. Section 5.10 dùng chuỗi tổng hợp và không nêu độ dài. Với chuỗi tối đa 2 patch thì không thể tách được như vậy, nên kết quả tương quan 0,986 không đại diện cho dữ liệu này.

**Giới hạn:** Table 6 báo single-pattern làm Reasoning giảm từ 44,59 xuống 35,81. Đó là số sau khi train lại, nên chưa so trực tiếp được.

## Ý 4: PATRA và TimeOmni ở Reasoning

**Nguồn:** Table 1 và Section 5.4 của paper. Số câu được quy đổi từ phần trăm.

| Task | PATRA | Đối thủ mã nguồn mở tốt nhất | Lệch (câu) | p không ghép cặp | p McNemar thấp nhất có thể |
|---|---|---|---|---|---|
| Comprehension | 65/116 | ChatTS 52 | 13 | 0,088 | 0,0002 |
| Recognition | 469/725 | Qwen 330 | 139 | < 0,0001 | < 0,0001 |
| Reasoning | 66/148 | TimeOmni 63 | **3** | 0,725 | **0,25** |
| Prescience | 57/108 | ITFormer 48 | 9 | 0,221 | 0,0039 |

- **Reasoning:** dù đầu ra từng câu ghép cặp thế nào, p cũng không thấp hơn 0,25. Paper lại viết "substantially surpassing all textual baselines".
- **Lỗi có thể có trong bảng:** ô second-best của cột Reasoning tô cho Qwen 36,48, trong khi TimeOmni ghi 42,56. Hoặc phần tô màu sai, hoặc số của hàng đó có vấn đề.
- **Comprehension và Prescience:** chưa kết luận được nếu không có đầu ra từng câu của đối thủ.
- **Mức nền:** luôn chọn nhãn phổ biến nhất theo định dạng câu hỏi đạt 66,22% ở Reasoning, cao hơn mọi model trong Table 1, kể cả GPT-4o 50,00%. Mức này tính trên chính nhãn của tập eval, nên chỉ là tham chiếu.

## Ý 1: Trend và seasonal ghép chéo

**Vị trí:** `modeling_qwen2.py:200`, `:457`, `:470-471`.

- `series_decomp` trả về `(res, moving_mean)` và dòng 457 nhận đúng thứ tự, nên phần phân rã không sai.
- Dòng 470–471 ghép `text_trend` với phần dư và `text_seasonal` với moving average, ngược với cặp `(trend, t)`, `(sea, s)` của Eq. 7.
- **Không phải lỗi chức năng:** hai nhánh có cùng cấu trúc, khởi tạo độc lập và được lấy trung bình ở dòng 491, nên việc ghép chéo tương đương đổi tên hai nhánh. Vì vậy nó không giải thích được khoảng lệch accuracy.
- **Hệ quả thật:** cách diễn giải theo tên nhánh không chính xác, ví dụ Hình 5 nói Trend Query chú ý quá mức, trong khi `trend_query` thực ra đi với phần dư.
- Không nên dùng `fix_swap` lúc inference để đánh giá, vì weight đã được train theo cách ghép hiện tại.

## Phát hiện thêm

- **Rò rỉ giữa các mẫu trong batch:** `modeling_qwen2.py:456-457` ghép patch của cả batch thành một dãy rồi mới lấy moving average. Trend của một mẫu vì thế phụ thuộc vào mẫu bên cạnh, và điều này đã được xác nhận bằng số trên dữ liệu giả lập. Script SFT dùng batch 2 trên mỗi GPU, còn lần chạy lại dùng batch 1. Ở eval, việc này cũng xảy ra giữa các chuỗi trong 34 mẫu có nhiều chuỗi.
- **Mất kênh mask:** `modeling_qwen2.py:145` chỉ giữ kênh giá trị rồi đệm số 0 ở `:149`. Chuỗi 24 điểm có 8/32 vị trí là đệm, trùng với mức trung bình sau chuẩn hóa, nên model không phân biệt được đệm với dữ liệu thật.

## Liên hệ với khoảng lệch Table 1

Các ý trên giải thích vì sao module chuỗi khó tạo lợi ích, nhưng **không giải thích được khoảng lệch giữa bản công bố và Table 1**. Nguyên nhân khoảng lệch vẫn cần tác giả xác nhận, như đã ghi trong [REPRODUCTION.md](REPRODUCTION.md).
