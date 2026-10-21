# Ghi chú thay đổi Notebook 03: Zero-shot

Tài liệu này giải thích các đoạn mã đã thêm hoặc sửa trong notebooks/03_zeroshot_xlmr_squad2.ipynb và cách bạn có thể chạy lại phần Zero-shot.

## 1. Cố định seed — đoạn mã đã thêm

Seed được đặt trước khi nạp model. Mục đích là dùng cùng giá trị khởi đầu cho các thao tác ngẫu nhiên (nếu có).

~~~python
import random

# Cố định seed để các thao tác ngẫu nhiên (nếu có) có thể lặp lại.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
~~~

Seed giúp ghi rõ thiết lập chạy, nhưng không đảm bảo mọi thiết bị và phiên bản thư viện luôn tạo kết quả giống từng bit.

## 2. Ghi nhận lỗi inference — đoạn mã đã sửa

### Trước khi sửa

Nếu một mẫu gặp lỗi, cell chỉ đổi dự đoán thành chuỗi rỗng. Vì chuỗi rỗng cũng là câu trả lời no-answer hợp lệ, cách này có thể che giấu lỗi:

~~~python
except Exception as e:
    pred = ""
~~~

### Sau khi sửa

Trước vòng lặp, tạo danh sách để lưu lỗi:

~~~python
inference_errors = []
predictions = []
~~~

Khi một mẫu gặp lỗi, lưu ID và thông báo lỗi:

~~~python
except Exception as e:
    inference_errors.append({
        "id": ex["id"],
        "error": str(e),
    })
    pred = ""
~~~

Sau vòng lặp, in số lỗi. Nếu có lỗi, dừng trước khi tính metrics:

~~~python
print("done:", len(predictions))
print("Số mẫu gặp lỗi suy luận:", len(inference_errors))

if inference_errors:
    print("Một số lỗi đầu tiên:", inference_errors[:5])
    raise RuntimeError(
        "Có lỗi suy luận; chưa nên dùng kết quả này làm metrics cuối."
    )
~~~

Nếu có lỗi, cần xử lý lỗi trước khi coi metrics là kết quả cuối. Chuỗi rỗng do lỗi không được âm thầm tính như dự đoán no-answer hợp lệ.

## 3. Lưu cấu hình Zero-shot — đoạn mã đã thêm

Sau khi tạo metrics, notebook ghi cấu hình vào results/zeroshot_config.json. Cấu hình lưu tên model/tokenizer, split, số mẫu, độ dài chuỗi, stride, seed, thiết bị và phiên bản thư viện.

~~~python
zs_config = {
    "model": MODEL_NAME,
    "tokenizer": TOKENIZER_NAME,
    "mode": "zero-shot",
    "dataset": "UIT-ViQuAD2.0",
    "split": "dev",
    "sample_size": len(dev_eval),
    "max_len": MAX_LEN,
    "stride": STRIDE,
    "seed": SEED,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
}

config_path = RESULTS_DIR / "zeroshot_config.json"
with config_path.open("w", encoding="utf-8") as file:
    json.dump(zs_config, file, ensure_ascii=False, indent=2)

print("Đã lưu cấu hình:", config_path)
~~~

Notebook tiếp tục lưu hai file kết quả:

- results/zeroshot_metrics.json
- results/zeroshot_predictions.json

## 4. Những gì đã xóa

Không xóa thuật toán dự đoán, model hay cách tính metrics. Phần xử lý lỗi cũ gồm hai dòng dưới đây đã được thay bằng đoạn ghi nhận lỗi ở mục 2:

~~~python
except Exception as e:
    pred = ""
~~~

Output cũ của các cell Zero-shot bị ảnh hưởng cũng được xóa để tránh nhầm với lần chạy dùng mã mới. Các file kết quả cũ trong results/ chưa phải kết quả xác nhận sau những thay đổi này.

## 5. Cách chạy riêng phần Zero-shot

1. Mở notebook trong thư mục dự án và chọn kernel .venv của dự án.
2. Lưu notebook.
3. Chọn Restart Kernel and Clear All Outputs.
4. Chạy lần lượt từ đầu notebook đến hết phần Zero-shot; dừng trước tiêu đề Phần 2. Không chọn Run All, vì phần sau có các cell fine-tuning thử nghiệm.
5. Giữ SAMPLE_SIZE = 0 để đánh giá toàn bộ dev.
6. Kết quả mong đợi:
   - done: 3814
   - Số mẫu gặp lỗi suy luận: 0
   - Có dòng xác nhận đã lưu zeroshot_config.json
7. Kiểm tra trong results/ có đủ zeroshot_config.json, zeroshot_metrics.json và zeroshot_predictions.json.

Nếu số mẫu không phải 3.814 hoặc số lỗi khác 0, chưa dùng metrics làm kết quả cuối.

## 6. Phần thử nghiệm fine-tuning

Các cell sau tiêu đề Phần 2 nằm ngoài kết quả Zero-shot và có thể chạy huấn luyện. Không chạy phần này khi chỉ muốn tái lập Zero-shot. Nhãn và mô tả của phần thử nghiệm cần được rà soát riêng để khớp với những cell fine-tuning thực sự có trong notebook.
