"""Fine-tune XLM-RoBERTa or mDeBERTa for Vietnamese extractive QA.

Luồng chính trong file này:
1. Đọc train/dev JSONL và tải tokenizer.
2. Tokenize câu hỏi/context, rồi tạo nhãn start/end.
3. Dùng Trainer fine-tune mô hình cho từng seed.
4. Gọi src/evaluation.py để giải mã câu trả lời và tính EM/F1.
5. Lưu mô hình, dự đoán và bảng tổng hợp.

Run from the repository root:
    python src/train_qa.py --model xlm-roberta
    python src/train_qa.py --model mdeberta
    python src/train_qa.py --model both

The comments and docstrings are intentionally detailed for NLP learners.
"""

# Bước 1: Khai báo module và cache
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

# Đặt cache trong repository để tránh quyền ghi ra thư mục hồ sơ Windows.
# Biến môi trường phải được thiết lập trước khi import Hugging Face Datasets/Hub.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = PROJECT_ROOT / ".cache" / "huggingface"
DATASET_CACHE_DIR = Path(tempfile.gettempdir()) / "viquad_hf_datasets"
os.environ.setdefault("HF_HOME", str(HF_CACHE_DIR))
os.environ.setdefault("HF_DATASETS_CACHE", str(DATASET_CACHE_DIR))

import numpy as np
import torch
from datasets import load_dataset
from filelock import FileLock
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Bước 2: Chuẩn bị import nội bộ và constants
# Khi script được gọi trực tiếp bằng `python src/train_qa.py`, Python mặc định
# đưa thư mục src/ lên sys.path. Thêm thư mục gốc để import `src.*` ổn định.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import decode_predictions, make_compute_metrics
from src.data_preprocessing import (
    DOC_STRIDE,
    MAX_SEQ_LENGTH,
    prepare_train_features,
    prepare_validation_features,
)
from src.utils import save_json


# Dùng đường dẫn tương đối từ thư mục gốc để dễ chuyển dự án sang máy khác.
TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.json"
DEV_FILE = PROJECT_ROOT / "data" / "processed" / "dev.json"
RESULTS_DIR = PROJECT_ROOT / "results" / "finetune"

# Mỗi mô hình dùng checkpoint riêng do nhóm đã chọn trong README/AGENT.md.
MODEL_CHECKPOINTS = {
    "xlm-roberta": "xlm-roberta-base",
    "mdeberta": "microsoft/mdeberta-v3-base",
}

# Một seed cố định giúp tái lập một lần chạy. Ba seed cho phép ước lượng
# độ biến động khi khởi tạo trọng số và xáo trộn thứ tự minibatch.
DEFAULT_SEEDS = [42, 123, 1024]

# Bước 3: Đọc train/dev JSONL
def load_project_data(cache_dir: Path = DATASET_CACHE_DIR):
    """Đọc hai tệp JSON Lines đã chuẩn bị trong data/processed/."""
    for path in (TRAIN_FILE, DEV_FILE):
        if not path.exists():
            raise FileNotFoundError(
                f"Không tìm thấy {path}. Hãy chạy notebook EDA để tạo train/dev JSONL."
            )

    # datasets tự đọc từng dòng JSON thành một example và hỗ trợ .map() theo batch.
    loaded = load_dataset(
        "json",
        data_files={"train": str(TRAIN_FILE), "validation": str(DEV_FILE)},
        cache_dir=str(cache_dir),
    )
    return loaded["train"], loaded["validation"]


def print_preprocessing_summary(
    train_examples: Any,
    dev_examples: Any,
    train_features: Any,
    dev_features: Any,
    tokenizer: Any,
    training_args: TrainingArguments,
) -> None:
    """In vài thông tin để kiểm tra tiền xử lý trước khi bắt đầu fine-tune."""
    print("\n--- Tóm tắt dữ liệu trước khi huấn luyện ---")
    print(f"Số mẫu train gốc: {len(train_examples):,}")
    print(f"Số mẫu dev gốc: {len(dev_examples):,}")
    print(f"Số train features sau tiền xử lý: {len(train_features):,}")
    print(f"Số dev features sau tiền xử lý: {len(dev_features):,}")

    train_batch_size = training_args.per_device_train_batch_size
    accumulation_steps = training_args.gradient_accumulation_steps
    effective_batch_size_per_device = train_batch_size * accumulation_steps

    print("\nCấu hình huấn luyện:")
    print(f"Batch train trên mỗi GPU: {train_batch_size}")
    print(f"Số bước tích lũy gradient: {accumulation_steps}")
    print(
        "Batch hiệu dụng trên mỗi GPU: "
        f"{train_batch_size} x {accumulation_steps} = "
        f"{effective_batch_size_per_device}"
    )
    print(f"Batch đánh giá trên mỗi GPU: {training_args.per_device_eval_batch_size}")
    print(f"Số epoch: {training_args.num_train_epochs}")
    print(f"Learning rate: {training_args.learning_rate}")
    print(f"Dùng BF16: {training_args.bf16}")
    print(f"Dùng FP16: {training_args.fp16}")

    if len(train_features) == 0:
        print("\nKhông có train feature để xem nhãn.")
        return

    # Chọn một feature có span đáp án nếu tìm thấy; nếu không, xem feature đầu.
    diagnostic_feature_index = 0
    found_answer_span = False

    for feature_index in range(len(train_features)):
        feature = train_features[feature_index]
        input_ids = feature["input_ids"]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        start_position = int(feature["start_positions"])
        end_position = int(feature["end_positions"])

        if start_position != cls_index or end_position != cls_index:
            diagnostic_feature_index = feature_index
            found_answer_span = True
            break

    feature = train_features[diagnostic_feature_index]
    input_ids = feature["input_ids"]
    token_strings = tokenizer.convert_ids_to_tokens(input_ids)
    cls_index = input_ids.index(tokenizer.cls_token_id)
    start_position = int(feature["start_positions"])
    end_position = int(feature["end_positions"])

    print("\nVí dụ nhãn của một train feature:")
    print(f"Feature index: {diagnostic_feature_index}")
    print(f"Start position: {start_position}")
    print(f"End position: {end_position}")

    if start_position == cls_index and end_position == cls_index:
        print("Nhãn trỏ về CLS: mẫu không có đáp án hoặc cửa sổ này không chứa span.")
    elif found_answer_span:
        span_tokens = token_strings[start_position : end_position + 1]
        span_text = tokenizer.decode(
            input_ids[start_position : end_position + 1],
            skip_special_tokens=True,
        )
        print(f"Token được gán nhãn: {span_tokens}")
        print(f"Đoạn giải mã từ nhãn: {span_text}")


def make_training_options() -> dict[str, Any]:
    """Tạo cấu hình Trainer, dùng batch vật lý nhỏ để vừa VRAM 16 GB.

    Batch vật lý 4 x gradient accumulation 8 tương đương batch hiệu dụng 32
    trên một GPU. Gradient của 8 batch được cộng dồn trước một lần cập nhật trọng số.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Không phát hiện GPU CUDA. Hãy kích hoạt PyTorch CUDA và kiểm tra nvidia-smi."
        )

    # RTX 5070 Ti hỗ trợ BF16; nếu máy khác không hỗ trợ, dùng FP16 trên GPU.
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    # Tên tham số đổi giữa một số phiên bản Transformers; chọn tên tương thích.
    arguments = {
        "learning_rate": 2e-5,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 8,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 3,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "logging_strategy": "steps",
        "logging_steps": 100,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "dataloader_num_workers": 0,  # 0 thường ổn định hơn trên Windows.
        "report_to": "none",  # Không gửi log sang dịch vụ theo dõi bên ngoài.
    }

    supported = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    arguments[strategy_name] = "epoch"

    # Bỏ tham số tùy chọn không tồn tại ở phiên bản Transformers đang cài.
    return {key: value for key, value in arguments.items() if key in supported}


def file_sha256(path: Path) -> str:
    """Tính mã băm của đúng các byte trong một tệp."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe_data() -> dict[str, Any]:
    """Lấy danh tính train/dev và tìm hồ sơ T06 tương ứng."""
    splits = {}
    for name, path in (("train", TRAIN_FILE), ("dev", DEV_FILE)):
        splits[name] = {
            "file": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": file_sha256(path),
        }

    # Kết quả dev phụ thuộc train/dev. Ưu tiên hồ sơ T06 nếu hai hash khớp.
    dataset_version = f"train-dev-{splits['train']['sha256'][:12]}-{splits['dev']['sha256'][:12]}"
    manifest_file = None
    for path in sorted((PROJECT_ROOT / "reports").glob("data_version_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        recorded_splits = manifest.get("splits", {})
        if all(recorded_splits.get(name, {}).get("sha256") == info["sha256"]
               for name, info in splits.items()):
            dataset_version = manifest["dataset_version"]
            manifest_file = path.relative_to(PROJECT_ROOT).as_posix()
            break

    return {"dataset_version": dataset_version, "manifest": manifest_file, "splits": splits}


def describe_experiment(
    models: list[str], training_options: dict[str, Any], *, run_label: str | None = None
) -> dict[str, Any]:
    """Chốt dữ liệu, revision và cấu hình trước khi chọn thư mục kết quả."""
    data = describe_data()
    sources = {}
    for model_name in sorted(models):
        checkpoint = MODEL_CHECKPOINTS[model_name]

        # Tải tệp config nhỏ để biết commit thật sự được dùng. Đường dẫn cache
        # của tệp có dạng snapshots/<commit>/config.json.
        config_path = Path(hf_hub_download(repo_id=checkpoint, filename="config.json"))
        revision = config_path.parent.name
        if config_path.parent.parent.name != "snapshots" or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"Không xác định được revision của {checkpoint}: {config_path}")

        # Cả tokenizer và trọng số gốc được nạp từ cùng một commit.
        sources[model_name] = {
            "model": {"repo_id": checkpoint, "revision": revision},
            "tokenizer": {"repo_id": checkpoint, "revision": revision, "use_fast": True},
        }

    # Nếu cách ánh xạ nhãn, chấm điểm hoặc cấu hình train đổi, thư mục cũng đổi.
    pipeline_hashes = {}
    for name in ("train_qa.py", "data_preprocessing.py", "evaluation.py"):
        pipeline_hashes[name] = file_sha256(PROJECT_ROOT / "src" / name)
    package_versions = {}
    for name in ("torch", "transformers", "tokenizers", "datasets", "accelerate"):
        package_versions[name] = version(name)

    identity = {
        "data": data,
        "sources": sources,
        "training_options": training_options,
        "preprocessing": {"max_seq_length": MAX_SEQ_LENGTH, "doc_stride": DOC_STRIDE},
        "pipeline_sha256": pipeline_hashes,
        "package_versions": package_versions,
    }
    # Nhãn mới tạo một lượt độc lập dù dữ liệu và tham số giống hệt.
    # Giữ nhãn khi chạy lại cell để tiếp tục dùng kết quả đã xong của lượt đó.
    if run_label is not None:
        identity["run_label"] = run_label

    payload = json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    fingerprint = hashlib.sha256(payload).hexdigest()
    return {
        "experiment_id": f"run-{fingerprint[:16]}",
        "fingerprint": fingerprint,
        **identity,
    }


def save_run_json(path: Path, data: Any) -> None:
    """Ghi sang tệp tạm rồi thay thế để người đọc chỉ thấy JSON hoàn chỉnh."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Tệp tạm riêng cho mỗi lần ghi, kể cả nếu hai tiến trình ghi cùng lúc.
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, encoding="utf-8", delete=False, suffix=".tmp"
    ) as file:
        temp_path = Path(file.name)
    try:
        save_json(temp_path, data)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def make_training_arguments(
    output_dir: Path, seed: int, training_options: dict[str, Any]
) -> TrainingArguments:
    """Giữ nguyên cấu hình đã ghi; thêm đường dẫn và seed của lượt cụ thể."""
    supported = inspect.signature(TrainingArguments.__init__).parameters
    seed_options = {"seed": seed, "data_seed": seed}
    seed_options = {key: value for key, value in seed_options.items() if key in supported}
    return TrainingArguments(output_dir=str(output_dir), **seed_options, **training_options)


# Bước 4: Chọn thư mục theo danh tính thí nghiệm và seed.
def train_one_seed(
    model_name: str, seed: int, result_root: Path, experiment: dict[str, Any]
) -> dict[str, Any]:
    """Huấn luyện hoặc đọc kết quả hoàn chỉnh đúng phiên bản cho một seed."""
    output_dir = result_root / model_name / f"seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Tránh hai kernel cùng ghi checkpoint, metrics và predictions của một seed.
    with FileLock(str(output_dir / "seed.lock"), timeout=0):
        return _train_one_seed(model_name, seed, output_dir, experiment)


def _train_one_seed(
    model_name: str, seed: int, output_dir: Path, experiment: dict[str, Any]
) -> dict[str, Any]:
    sources = experiment["sources"][model_name]
    checkpoint = sources["model"]["repo_id"]
    revision = sources["model"]["revision"]
    identity = {
        "experiment_fingerprint": experiment["fingerprint"],
        "model": model_name,
        "seed": seed,
    }
    result_file = output_dir / "metrics.json"
    prediction_file = output_dir / "predictions.json"
    best_model_dir = output_dir / "best_model"
    complete_file = output_dir / "training_complete.json"

    # Hoàn tất train và hoàn tất ghi kết quả là hai mốc khác nhau. Chỉ tin
    # best_model khi marker được tạo sau khi lưu cả model và tokenizer.
    recovered_state = None
    weight_files = ("model.safetensors", "model.safetensors.index.json",
                    "pytorch_model.bin", "pytorch_model.bin.index.json")
    model_is_saved = (best_model_dir / "config.json").is_file() and any(
        (best_model_dir / name).is_file() for name in weight_files
    )
    if complete_file.is_file() and model_is_saved:
        saved_state = json.loads(complete_file.read_text(encoding="utf-8"))
        if saved_state.get("run_identity") == identity:
            recovered_state = saved_state

    if recovered_state is not None and result_file.is_file() and prediction_file.is_file():
        saved_result = json.loads(result_file.read_text(encoding="utf-8"))
        if (saved_result.get("run_identity") == identity
                and saved_result.get("training_attempt") == recovered_state["training_attempt"]):
            print(f"Đã có kết quả đúng phiên bản cho seed {seed}: {output_dir}")
            return saved_result

    # Seed phải được đặt trước khi khởi tạo mô hình để kiểm soát trọng số ban đầu.
    set_seed(seed)
    print(f"\n{'=' * 72}\nModel: {checkpoint} | seed: {seed}\n{'=' * 72}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Thư mục seed: {output_dir}")

    expected_data = experiment["data"]["splits"]
    for split_name, path in (("train", TRAIN_FILE), ("dev", DEV_FILE)):
        if file_sha256(path) != expected_data[split_name]["sha256"]:
            raise RuntimeError(f"{path.name} đã đổi sau khi lập kế hoạch. Hãy chạy lại thí nghiệm.")

    # Tách cache theo nội dung train/dev để tránh nạp dữ liệu cũ khi tên file giữ nguyên.
    data_cache_key = expected_data["train"]["sha256"] + expected_data["dev"]["sha256"]
    data_cache_id = hashlib.sha256(data_cache_key.encode("ascii")).hexdigest()[:16]
    train_examples, dev_examples = load_project_data(DATASET_CACHE_DIR / data_cache_id)
    for split_name, path in (("train", TRAIN_FILE), ("dev", DEV_FILE)):
        if file_sha256(path) != expected_data[split_name]["sha256"]:
            raise RuntimeError(f"{path.name} đã đổi trong lúc đọc dữ liệu. Hãy chạy lại thí nghiệm.")

    # Bước 5: Cố định tokenizer ở revision đã ghi trong hồ sơ.
    tokenizer_source = sources["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source["repo_id"],
        revision=tokenizer_source["revision"],
        use_fast=tokenizer_source["use_fast"],
    )
    if not tokenizer.is_fast:
        raise RuntimeError(f"Checkpoint {checkpoint} không cung cấp Fast Tokenizer.")


    # Bước 6: Tạo train features
    # Ánh xạ từng câu trả lời từ ký tự trong context thành chỉ số token.
    train_features = train_examples.map(
        lambda batch: prepare_train_features(batch, tokenizer),
        batched=True,
        remove_columns=train_examples.column_names,
        desc="Đổi train examples thành token features",
    )

    # Bước 7: Tạo dev metadata và map theo ID
    # Validation cần metadata (ID và offsets) để giải mã dự đoán về ký tự.
    dev_features_with_metadata = dev_examples.map(
        lambda batch: prepare_validation_features(batch, tokenizer),
        batched=True,
        remove_columns=dev_examples.column_names,
        desc="Tạo cửa sổ validation và giữ offsets",
    )
    feature_ids = list(dev_features_with_metadata["example_id"])
    feature_offsets = list(dev_features_with_metadata["offset_mapping"])
    examples_by_id = {
        example["id"]: {
            "context": example["context"],
            "answers": example["answers"],
            "is_impossible": example["is_impossible"],
        }
        for example in dev_examples
    }

    # Bước 8: Tạo nhãn dev cho Trainer
    # Trainer cũng cần nhãn start/end khi evaluate để gọi compute_metrics.
    # Tạo nhãn validation từ đáp án chuẩn giống cách đã làm cho train; metadata
    # ID/offset ở trên vẫn dùng để tính EM/F1 trên văn bản gốc, không phải logits.
    dev_features_with_labels = dev_examples.map(
        lambda batch: prepare_train_features(batch, tokenizer),
        batched=True,
        remove_columns=dev_examples.column_names,
        desc="Tạo nhãn start/end cho validation loss",
    )
    dev_model_features = dev_features_with_labels

    # Bước 9: Mỗi lần train mới có thư mục checkpoint riêng. Nếu một lần
    # trước dừng giữa chừng, checkpoint của nó không xen vào Trainer mới.
    if recovered_state is None:
        # Xóa marker cũ nếu nó không đủ điều kiện khôi phục. Nếu lần mới
        # lại bị ngắt, không được gán marker cũ cho best_model chưa ghi xong.
        complete_file.unlink(missing_ok=True)
        attempt_dir = Path(tempfile.mkdtemp(prefix="a-", dir=output_dir))
    else:
        attempt_dir = output_dir / "recovery"

    training_args = make_training_arguments(attempt_dir, seed, experiment["training_options"])
    print_preprocessing_summary(
        train_examples,
        dev_examples,
        train_features,
        dev_model_features,
        tokenizer,
        training_args,
    )

    # Bước 10: Chỉ khôi phục best_model đúng danh tính. Khi train mới,
    # từng seed bắt đầu từ revision pretrained đã chốt.
    if recovered_state is not None:
        print(f"Khôi phục phần đánh giá của seed {seed} từ best_model đúng phiên bản.")
        model = AutoModelForQuestionAnswering.from_pretrained(str(best_model_dir))
    else:
        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint, revision=revision)
    # Bước 11: Tạo callback metrics, collator và Trainer
    compute_metrics = make_compute_metrics(
        feature_ids, feature_offsets, examples_by_id
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Tên đối số tokenizer đổi thành processing_class ở các bản mới hơn.
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    tokenizer_argument = (
        {"processing_class": tokenizer}
        if "processing_class" in trainer_parameters
        else {"tokenizer": tokenizer}
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_features,
        eval_dataset=dev_model_features,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        **tokenizer_argument,
    )

    # Bước 12: Train; Trainer đánh giá dev sau mỗi epoch và giữ model F1 tốt nhất.
    if recovered_state is None:
        started_at = time.time()
        trainer.train()
        training_seconds = round(time.time() - started_at, 2)

        # Ghi best_model trước rồi mới báo mốc hoàn tất train. Nếu phần
        # dự đoán bị ngắt, lần sau có thể nạp lại đúng model để ghi kết quả.
        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(best_model_dir)
        scored_epochs = [row for row in trainer.state.log_history if "eval_f1" in row]
        best_epoch_row = max(scored_epochs, key=lambda row: row["eval_f1"])
        recovered_state = {
            "run_identity": identity,
            "training_attempt": attempt_dir.name,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_epoch": best_epoch_row.get("epoch"),
            "training_seconds": training_seconds,
        }
        save_run_json(complete_file, recovered_state)

    # Bước 13–14: Đánh giá trên dev và giải mã về đoạn văn gốc.
    final_metrics = trainer.evaluate()
    predictions = trainer.predict(dev_model_features).predictions
    decoded = decode_predictions(
        predictions[0], predictions[1], feature_ids, feature_offsets, examples_by_id
    )
    prediction_rows = []
    for example_id, example in examples_by_id.items():
        prediction_rows.append(
            {
                "id": example_id,
                "prediction_text": decoded[example_id],
                "ground_truths": example["answers"].get("text", []),
                "is_impossible": example["is_impossible"],
            }
        )

    # Bước 15: Lưu phiên bản dữ liệu và cấu hình thực tế ngay cùng metric.
    result = {
        "model": model_name,
        "checkpoint": checkpoint,
        "seed": seed,
        "dataset_version": experiment["data"]["dataset_version"],
        "data": experiment["data"],
        "source_versions": sources,
        "preprocessing": experiment["preprocessing"],
        "training_config": {
            **experiment["training_options"],
            "seed": training_args.seed,
            "data_seed": training_args.data_seed,
        },
        "experiment_id": experiment["experiment_id"],
        "run_identity": identity,
        "run_label": experiment.get("run_label"),
        "training_attempt": recovered_state["training_attempt"],
        "best_checkpoint": recovered_state["best_checkpoint"],
        "best_epoch": recovered_state["best_epoch"],
        "best_eval_f1": float(final_metrics["eval_f1"]),
        "best_eval_exact_match": float(final_metrics["eval_exact_match"]),
        "training_seconds": recovered_state["training_seconds"],
        "effective_batch_size": (
            training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
        ),
        "epochs_requested": training_args.num_train_epochs,
        "bf16": training_args.bf16,
    }

    # Chỉ ghi metrics sau predictions để metrics đánh dấu lượt đã hoàn tất.
    save_run_json(prediction_file, prediction_rows)
    save_run_json(result_file, result)
    print(f"Kết quả seed {seed}: EM={result['best_eval_exact_match']:.2f}, F1={result['best_eval_f1']:.2f}")
    return result


# Bước 16: Tổng hợp seeds và CLI
def run_experiment(
    models: list[str], seeds: list[int], *, plan_only: bool = False, run_label: str | None = None
) -> Path:
    """Xem kế hoạch hoặc chạy thí nghiệm; trả về thư mục để notebook đọc kết quả."""
    models = list(dict.fromkeys(models))
    seeds = sorted(set(seeds))
    if not models or not seeds:
        raise ValueError("Cần chọn ít nhất một mô hình và một seed.")

    if run_label is not None:
        run_label = run_label.strip()
        if not run_label:
            raise ValueError("Nhãn lượt train không được để trống.")

    training_options = make_training_options()
    experiment = describe_experiment(models, training_options, run_label=run_label)
    result_root = RESULTS_DIR / "runs" / experiment["experiment_id"]
    print(f"Phiên bản dữ liệu: {experiment['data']['dataset_version']}")
    print(f"Thư mục thí nghiệm: {result_root}")
    print(f"Seed yêu cầu: {seeds}")
    if run_label is not None:
        print(f"Nhãn lượt chạy: {run_label}")
    batch_size = training_options["per_device_train_batch_size"]
    accumulation_steps = training_options["gradient_accumulation_steps"]
    print(f"Batch hiệu dụng trên mỗi GPU: {batch_size} x {accumulation_steps} = {batch_size * accumulation_steps}")
    for model_name in models:
        print(f"{model_name}: revision {experiment['sources'][model_name]['model']['revision']}")

    if plan_only:
        print("Chế độ xem trước: chưa nạp trọng số, chưa huấn luyện.")
        return result_root

    result_root.mkdir(parents=True, exist_ok=True)
    manifest_path = result_root / "run_manifest.json"
    with FileLock(str(result_root / "manifest.lock")):
        if manifest_path.exists():
            saved_experiment = json.loads(manifest_path.read_text(encoding="utf-8"))
            if saved_experiment != experiment:
                raise RuntimeError(f"Hồ sơ thí nghiệm không khớp tại {manifest_path}.")
        else:
            save_run_json(manifest_path, experiment)

    all_results: list[dict[str, Any]] = []
    for model_name in models:
        for seed in seeds:
            all_results.append(train_one_seed(model_name, seed, result_root, experiment))

    # Tổng hợp riêng từng kiến trúc để không trộn metric giữa các model.
    summary: dict[str, Any] = {
        "experiment_id": experiment["experiment_id"],
        "dataset_version": experiment["data"]["dataset_version"],
        "requested_seeds": seeds,
        "runs": all_results,
        "models": {},
    }
    seed_suffix = "-".join(str(seed) for seed in seeds)
    summary_name = f"summary-seeds-{seed_suffix}.json"
    for model_name in models:
        model_runs = [row for row in all_results if row["model"] == model_name]
        best_run = max(
            model_runs,
            key=lambda row: (row["best_eval_f1"], row["best_eval_exact_match"]),
        )
        f1_values = [row["best_eval_f1"] for row in model_runs]
        em_values = [row["best_eval_exact_match"] for row in model_runs]
        summary["models"][model_name] = {
            "experiment_id": experiment["experiment_id"],
            "dataset_version": experiment["data"]["dataset_version"],
            "requested_seeds": seeds,
            "best_seed": best_run["seed"],
            "best_f1": best_run["best_eval_f1"],
            "best_exact_match": best_run["best_eval_exact_match"],
            "mean_f1": float(np.mean(f1_values)),
            "std_f1": float(np.std(f1_values)),
            "mean_exact_match": float(np.mean(em_values)),
            "std_exact_match": float(np.std(em_values)),
        }

        # Mỗi notebook kiến trúc có file summary riêng để đọc kết quả độc lập.
        model_summary_path = result_root / model_name / summary_name
        save_run_json(model_summary_path, summary["models"][model_name])

    summary_path = result_root / summary_name
    save_run_json(summary_path, summary)
    print("\nTổng hợp (F1/EM theo phần trăm):")
    for model_name, scores in summary["models"].items():
        print(
            f"{model_name}: seed tốt nhất={scores['best_seed']}, "
            f"F1={scores['best_f1']:.2f}, EM={scores['best_exact_match']:.2f}; "
            f"F1 trung bình={scores['mean_f1']:.2f} ± {scores['std_f1']:.2f}"
        )
    print(f"Đã lưu tổng hợp tại: {summary_path}")
    return result_root


# Bước 16: Đọc lựa chọn từ dòng lệnh
def parse_args() -> argparse.Namespace:
    """Đọc lựa chọn kiến trúc/seed từ dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Fine-tune XLM-RoBERTa và/hoặc mDeBERTa cho UIT-ViQuAD."
    )
    parser.add_argument(
        "--model",
        choices=["xlm-roberta", "mdeberta", "both"],
        default="both",
        help="Chọn một kiến trúc hoặc chạy cả hai.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="Các seed cần chạy; mặc định là 42 123 1024.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="In phiên bản dữ liệu và thư mục thí nghiệm, chưa huấn luyện.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Bật UTF-8 cho terminal Windows để argparse in được tiếng Việt.
    for output_stream in (sys.stdout, sys.stderr):
        if hasattr(output_stream, "reconfigure"):
            output_stream.reconfigure(encoding="utf-8")
    args = parse_args()
    selected_models = (
        list(MODEL_CHECKPOINTS) if args.model == "both" else [args.model]
    )
    run_experiment(selected_models, args.seeds, plan_only=args.plan_only)
