# Insight về implementation của PATRA

**Luận điểm:** trong bản công bố, đường đi của chuỗi gần như không nối vào quyết định của model. Điều này không phải vì chuỗi vô dụng: một bộ phân loại nhỏ dùng 16 thống kê chuỗi vẫn lấy được tín hiệu mà PATRA không lấy được. Mọi kết luận ở đây chỉ nói về bản công bố, không khẳng định số trong paper là sai.

Nguồn số liệu:
- [alignment_mechanics.json](../results/analysis/alignment_mechanics.json), do [alignment_mechanics.py](../scripts/alignment_mechanics.py) tạo ra
- [benchmark_probes.json](../results/analysis/benchmark_probes.json), do [benchmark_probes.py](../scripts/benchmark_probes.py) tạo ra
- [code_review_checks.json](../results/analysis/code_review_checks.json), chi tiết ở [CODE_REVIEW.md](CODE_REVIEW.md)

## 1. Pattern-Aware Alignment thực chất là phép lấy trung bình

Tính lại `ts_encoder` và `align` trên CPU bằng weight công bố, trên 1.097 câu có nhãn. Bản tính lại khớp số đo GPU trong `ts_probe.csv` với r = 1,0 ở cả 3 đại lượng.

| Đại lượng | Kết quả |
|---|---|
| Điểm attention lớn nhất (trị tuyệt đối) | 0,0023 |
| Độ lệch lớn nhất khỏi phân phối đều | 0,0002 |
| Đầu ra đổi khi thay attention bằng trung bình thuần | 0,005% (tối đa 0,03%) |

| Token patch đầu tiên (1.063 câu một chuỗi) | Trước `align` | Sau `align` |
|---|---|---|
| Năng lượng là phần hằng số giữa các câu | 1,29% | **98,12%** |
| Cosine trung bình giữa hai câu bất kỳ | 0,0075 | **0,9814** |

**Cơ chế:** weight có giá trị nhỏ và điểm còn được chia cho căn bậc hai của `head_dim`, nên softmax gần như chia đều. Mỗi token chuỗi trộn với 10 LAT gần như cố định, nên đầu ra chủ yếu là hằng số. Phần đầu ra đổi theo chuỗi gần như tuyến tính theo token encoder, vì các phép chiếu đều tuyến tính khi attention đã đều.

**Giới hạn của insight này:** tắt `align` cũng không làm accuracy đổi có ý nghĩa (`no_align` 47,95%), nên việc sửa riêng `align` không đủ. Tín hiệu đi thẳng từ encoder vào backbone vẫn không được dùng.

## 2. Quy trình train gần như không có cách để sửa vấn đề trên

- **Encoder và `align` gần như giữ nguyên khởi tạo.** Script dùng `--finetuning_type lora`, không đặt `additional_target`, nên theo hành vi mặc định của PEFT weight gốc bị đóng băng. Chỉ có cập nhật LoRA rank 8 được học. Weight công bố phù hợp với điều này: std khoảng 0,02, bias đúng bằng 0, không có thành phần nào vượt mức nhiễu.
- **`--freeze_ts_encoder False` không có tác dụng trong chế độ này.** `forbidden_modules` chỉ được dùng ở nhánh `full` và `freeze` của `adapter.py`.
- **GRPO chỉ nhận tín hiệu gần như nhị phân.** Với `num_generations=2`, advantage gần như luôn là ±0,707.
- **Đáp án train đòi những số model không có.** 77,02% số trong đáp án SFT là giá trị thô, trong khi `offset` và `scale` đã bị bỏ khỏi prompt. Mức khớp với chuỗi khác chỉ là 2,71%. Số PATRA trích ở câu open-ended chỉ khớp 3,44%, ngang mức ngẫu nhiên.

**Giới hạn:** đây là suy luận từ code và weight, không có log train. Vì `PATRA-init` và `trainer.py` chưa công bố, chưa đo trực tiếp được weight đã đổi bao nhiêu.

## 3. Metric thưởng cho văn phong nhiều hơn cho hiểu chuỗi

ROUGE-L trên 1.868 câu open-ended:

| Cách trả lời | Comp. | Recog. | Reason. | Presc. | Chung |
|---|---|---|---|---|---|
| Baseline tốt nhất trong Table 1 | 18,77 | 14,61 | 18,72 | 15,53 | |
| Qwen, chạy lại | 18,74 | 10,60 | 17,56 | 10,96 | 15,60 |
| Qwen, cắt còn 35 từ đầu | 21,68 | 17,02 | 22,35 | 18,49 | 20,42 |
| **Một câu cố định cho mọi đề** | **20,49** | **18,88** | **24,12** | **20,56** | **21,21** |
| Đáp án của câu hỏi train gần nhất, không đọc chuỗi | 23,78 | 21,15 | 24,15 | 23,50 | 23,23 |
| PATRA, chạy lại | 27,39 | 26,02 | 29,04 | 27,55 | 27,58 |

- Câu cố định và độ dài 35 từ đều được chọn từ tập train. Câu đó là: "The series shows a gradual increasing trend. The values increase from 8.0 to 71.0, with some consistency and smoothness in the upward trend."
- Câu cố định này cao hơn mọi baseline trong Table 1 ở cả 4 task.
- Đổi chiều xu hướng trong đáp án mẫu (upward thành downward và ngược lại) vẫn đạt ROUGE-L **94,49** trên 1.169 câu. Metric gần như không phân biệt được đúng hay sai về chiều.
- Trung vị độ dài: đáp án mẫu 35 từ, PATRA 33 từ, Qwen 135 từ.

**Giới hạn:** PATRA vẫn cao hơn cách truy xuất không đọc chuỗi 4,35 điểm. Phần này chưa quy được nguyên nhân, vì đối chứng chuỗi chưa chạy trên open-ended. Riêng phép đổi chiều là một phép thử trên đáp án mẫu, không phải trên đầu ra của model.

## 4. Accuracy: không đọc chuỗi đã đi được rất xa, nhưng chuỗi vẫn có giá trị

Trên 1.097 câu có nhãn. Model và tham số được chọn trước trên tập train, kết quả được báo cáo đúng một cấu hình.

| Mô hình | Đọc câu hỏi | Đọc chuỗi | Đúng | Accuracy | p so với PATRA |
|---|---|---|---|---|---|
| Luôn trả lời False/B (nhãn phổ biến nhất của train) | Không | Không | 521 | 47,49 | 1,0 |
| **PATRA, chạy lại** | Có | Có | 521 | 47,49 | |
| Qwen, chạy lại | Có | Dạng text | 488 | 44,48 | 0,12 |
| Cây trên TF-IDF | Có | Không | 574 | 52,32 | 0,013 |
| Hồi quy logistic trên TF-IDF | Có | Không | 613 | 55,88 | 0,00001 |
| Cây trên TF-IDF cộng 16 thống kê chuỗi | Có | Có | **634** | **57,79** | < 0,00001 |
| Cùng model đó, xáo chuỗi lúc test | Có | Sai chuỗi | 588 | 53,60 | 0,002 |
| PATRA theo paper (quy đổi) | | | 657 | 59,89 | |

- **Giả thuyết ban đầu của tôi bị bác bỏ.** Tôi từng đoán benchmark không thưởng cho việc đọc chuỗi. Thực tế thêm 16 thống kê đơn giản tăng 60 câu so với cùng model chỉ đọc câu hỏi (p = 0,00001), và xáo chuỗi lúc test làm mất 46 câu.
- **Điều đó làm nổi bật khoảng trống:** chuỗi có tín hiệu lấy được, trong khi thay encoder và `align` của PATRA bằng weight ngẫu nhiên không làm giảm accuracy.
- **Câu hỏi train có thể dự đoán nhãn:** tín hiệu từ ngữ của câu True/False gồm "structural break", "high volatility" nghiêng về True và "seasonality", "low volatility" nghiêng về False. Có thể đây là thiên lệch khi sinh dữ liệu hoặc phân phối thật của dữ liệu; chưa phân biệt được. Hiện tượng này giống hypothesis-only baseline trong NLI ([Gururangan et al., 2018](https://arxiv.org/abs/1803.02324v1); [Poliak et al., 2018](https://preview.aclanthology.org/menus/S18-2023/)).
- **Không phụ thuộc vào việc trùng câu hỏi:** trên 975 câu chưa từng xuất hiện ở train, mô hình logistic đúng 531, PATRA đúng 450.

**Giới hạn:**
- Các p chưa hiệu chỉnh cho nhiều phép so sánh.
- Tham số C = 0,3 nằm ở biên lưới, nên có thể chưa tối ưu.
- 16 thống kê dùng giá trị thô, chỉ tính trên chuỗi đầu tiên của mỗi mẫu.
- Bảng này không nói PATRA thực sự dùng các tín hiệu từ ngữ trên. Nó chỉ cho thấy benchmark có thể đạt mức này mà không cần đọc chuỗi.

## Ý nghĩa cho hướng nghiên cứu

Các đề xuất dưới đây chưa được kiểm chứng trên LLM.

1. **Bắt đầu từ thông tin, không phải từ fusion.** Trả lại giá trị thô hoặc thống kê đã tính cho model. Bằng chứng: Qwen đọc text có 94,25% số trích khớp giá trị thô, và 16 thống kê đã tạo được tín hiệu đo được.
2. **Làm cho attention có thể chọn lọc được.** Nếu vẫn dùng cross-attention, cần chuẩn hóa thang đo của query và key, giảm tỷ lệ 10 LAT trên 1–2 token chuỗi, và train đầy đủ phần encoder thay vì chỉ LoRA.
3. **Đổi cách đánh giá.** Báo cáo cùng lúc baseline chỉ đọc câu hỏi và độ nhạy phản thực: cùng câu hỏi, đổi chuỗi thì đáp án có đổi đúng không. Thay ROUGE-L bằng phép kiểm tra đúng chiều và đúng số cho câu mô tả.

## Việc còn mở

- **Chạy `ts_zero` trên 1.868 câu open-ended:** cần GPU. Đây là phép thử duy nhất phân biệt được phần ROUGE-L còn lại đến từ chuỗi hay từ văn phong.
- **Khoảng lệch với Table 1:** các insight trên giải thích vì sao đường đi của chuỗi không có tác dụng ở bản công bố, nhưng không giải thích vì sao bản này thấp hơn paper.
