# Hỏi đáp trích xuất tiếng Việt UIT-ViQuAD 2.0

Dự án trích một đoạn trả lời từ context tiếng Việt hoặc trả chuỗi rỗng khi không có đáp án. Nhóm so sánh Jaccard, TF-IDF + Logistic Regression, Zero-shot và ba kiến trúc fine-tune trên cùng 3.814 câu dev.

## Cài đặt trên Windows

Mở PowerShell hoặc terminal VS Code ở thư mục gốc của dự án. Dùng Python 3.12 và chọn kernel `.venv` cho các notebook.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Nếu chỉ chạy demo, PyTorch chạy được trên CPU. Để fine-tune bằng GPU, dùng lệnh cài PyTorch cho Windows/CUDA từ [trang chính thức](https://pytorch.org/get-started/locally/) với Python trong `.venv`. Kiểm tra cell GPU trong notebook 04/05 hiện `Có CUDA: True` trước khi train. Lượt XLM-R và mDeBERTa chốt dùng Torch 2.14.1+cu130, Transformers 5.18.0, Datasets 5.0.1 và Accelerate 1.15.0. Xem phiên bản theo từng lượt trong `run_manifest.json` ở thư mục tương ứng. Nếu môi trường cũ báo xung đột, dùng thông báo `pip check` để xác định gói cần xử lý.

## Chạy demo

```powershell
.\.venv\Scripts\python.exe -m streamlit run demo/app.py
```

Mở địa chỉ Streamlit in ra trong terminal, thường là `http://localhost:8501`. Giao diện hiển thị tên seed, đường dẫn checkpoint, thiết bị và số cửa sổ thực tế sau mỗi lần dự đoán.

| Vai trò | Checkpoint cục bộ | Cách xử lý |
| --- | --- | --- |
| Chính | `results/finetune/phobert/seed-1024/best_model/` | PhoBERT-base-v2, tách từ underthesea 9.5.0, tối đa 256 token và chồng lặp 64 |
| Dự phòng | `results/finetune/runs/run-fe5f5f5b27578bd4/xlm-roberta/seed-42/best_model/` | XLM-RoBERTa-base, fast tokenizer, tối đa 384 token và chồng lặp 128 |

Ưu tiên PhoBERT vì F1 dev trung bình cao nhất trong ba kiến trúc, **56,01 +- 0,09%**. Seed 1024 đạt F1 cao nhất của PhoBERT (**56,14%**, checkpoint tốt nhất ở epoch 3). XLM-R seed 42 đạt **51,26%** F1 dev trong lượt đã chốt. Nếu PhoBERT gặp lỗi khi nạp hoặc dự đoán, ứng dụng ghi traceback trong terminal, hiển thị bước lỗi và thử XLM-R dự phòng. Chọn **Kiểm tra XLM-R dự phòng** để thử trực tiếp. Có thể chọn **Chạy trên CPU** để thử lại khi cần xác định lỗi liên quan đến GPU.

Checkpoint lớn được Git bỏ qua. Lấy **nguyên thư mục best_model** của PhoBERT seed 1024 từ Quy, gồm `config.json`, `model.safetensors`, `tokenizer_config.json`, `vocab.txt`, `bpe.codes` và các tệp tokenizer khác đi kèm. Lấy XLM-R từ lượt đã nêu hoặc dùng checkpoint của lượt tương ứng đã train bằng notebook 04. Đặt đúng thư mục trên, hoặc chỉ đến bản cục bộ đã nhận:

```powershell
$env:VIQUAD_PHOBERT_CHECKPOINT = 'D:\checkpoints\phobert-seed-1024\best_model'
$env:VIQUAD_XLMR_CHECKPOINT = 'D:\checkpoints\xlm-roberta-seed-42\best_model'
.\.venv\Scripts\python.exe -m streamlit run demo/app.py
```

`vinai/phobert-base-v2` trên Hugging Face là mô hình pretrained, còn demo cần trọng số QA seed 1024 do nhóm fine-tune. Xem [cách kiểm tra demo và bản nộp](reports/tong_duyet_ban_nop.md) để chạy ba ca mẫu từ dev và lưu bằng chứng.

## Thứ tự dữ liệu và thí nghiệm

1. **Notebook 01** đọc [mirror UIT-ViQuAD 2.0](https://huggingface.co/datasets/taidng/UIT-ViQuAD2.0), giữ snapshot ở `data/raw/`, kiểm tra offset và xuất JSON Lines ở `data/processed/`. Train có 8.000 mẫu (seed chọn mẫu 1024), dev có 3.814 mẫu, Test có 7.301 mẫu thiếu đáp án chuẩn. Nếu chỉ muốn kiểm tra kết quả đã nộp, sử dụng đúng file của bản đã chốt và đối chiếu SHA-256 trong `reports/data_version_*.json`.
2. **Notebook 02** chạy Jaccard và TF-IDF + Logistic Regression. **Notebook 03** cung cấp dự đoán Zero-shot của `deepset/xlm-roberta-base-squad2`. Các dự đoán Zero-shot đã bàn giao được dùng cho bảng hiện tại.
3. **Notebook 04 và 05** tương ứng XLM-R và mDeBERTa. **Notebook 06** chứa pipeline và kết quả PhoBERT. Mỗi kiến trúc có ba seed **42, 123, 1024**. Notebook 04/05 có `plan_only=True` để xem phiên bản, cấu hình và đường dẫn. Dùng `False` nếu cần huấn luyện. Chạy tới phần tổng hợp dev để lấy kết quả so sánh.
4. **Notebook 07** chấm các dự đoán đã lưu theo ID trên dev. Chọn các lượt `run-fe5f5f5b27578bd4` cho XLM-R và `run-a28388a1c7754133` cho mDeBERTa, sau đó chạy Bước 1–6 để tái tạo bảng và ví dụ lỗi từ dự đoán đã có. Xem [bảng so sánh dev](results/dev_model_comparison.json) và [phân tích lỗi](results/dev_error_analysis.json).

## Cách đọc kết quả

- **EM** so khớp toàn bộ câu trả lời sau chuẩn hóa. **F1** đo mức trùng token có xét số lần lặp. Hàm chấm dựa trên công thức SQuAD 2.0, với chuẩn hóa điều chỉnh cho tiếng Việt trong `src/evaluation.py`. Mỗi kiến trúc báo trung bình +- độ lệch chuẩn quần thể trên ba seed.
- **Accuracy có/không đáp án** và **Recall NoAns** dùng trực tiếp điều kiện `prediction_text == ""` so với `is_impossible`. Ma trận có hàng nhãn thật, cột dự đoán, cùng thứ tự HasAns, NoAns.
- **Dev** dùng để chọn checkpoint và báo điểm. File `data/processed/test.json` thiếu đáp án chuẩn nên dùng tạo dự đoán. Các bảng EM/F1 và accuracy trong dự án là điểm dev.

Bài báo nền tảng: [Nguyen và cộng sự, COLING 2020](https://aclanthology.org/2020.coling-main.233/). Bài báo giới thiệu UIT-ViQuAD, còn đồ án dùng bản mirror UIT-ViQuAD 2.0 nêu trên.
