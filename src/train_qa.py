"""Fine-tune XLM-RoBERTa or mDeBERTa for Vietnamese extractive QA.

Run from the repository root:
    python src/train_qa.py --model xlm-roberta
    python src/train_qa.py --model mdeberta
    python src/train_qa.py --model both

The comments and docstrings are intentionally detailed for NLP learners.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import string
import sys
import tempfile
import time
from collections import defaultdict
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
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Khi script được gọi trực tiếp bằng `python src/train_qa.py`, Python mặc định
# đưa thư mục src/ lên sys.path. Thêm thư mục gốc để import `src.*` ổn định.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import (
    prepare_train_features,
    prepare_validation_features,
)


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

# Giới hạn độ dài câu trả lời để decoder không sinh một span quá dài.
MAX_ANSWER_LENGTH = 30
N_BEST_TOKENS = 20
NO_ANSWER_THRESHOLD = 0.0


def normalize_text(text: str) -> str:
    """Chuẩn hóa nhẹ tiếng Việt trước khi tính EM/F1.

    SQuAD tiếng Anh thường bỏ articles (a/an/the), nhưng tiếng Việt không có
    articles tương đương theo cách đó. Vì vậy ta chỉ viết thường, thay dấu câu
    bằng khoảng trắng và gom khoảng trắng thừa.
    """
    text = text.lower()
    text = "".join(" " if char in string.punctuation else char for char in text)
    return " ".join(text.split())


def exact_match(prediction: str, references: list[str]) -> float:
    """Trả 1 nếu dự đoán trùng hoàn toàn với ít nhất một đáp án chuẩn."""
    normalized_prediction = normalize_text(prediction)
    return float(any(normalized_prediction == normalize_text(ref) for ref in references))


def token_f1(prediction: str, reference: str) -> float:
    """F1 dựa trên các token cách nhau bởi khoảng trắng.

    Với tiếng Việt, phép tách này tính đơn vị theo âm tiết cách viết; đây là
    cách đơn giản, nhất quán để bắt đầu và dễ so sánh với baseline dự án.
    """
    predicted_tokens = normalize_text(prediction).split()
    reference_tokens = normalize_text(reference).split()

    # Hai chuỗi rỗng nghĩa là mô hình trả no-answer đúng.
    if not predicted_tokens and not reference_tokens:
        return 1.0
    if not predicted_tokens or not reference_tokens:
        return 0.0

    # Counter giữ số lần xuất hiện để các token lặp không bị tính quá mức.
    from collections import Counter

    common = Counter(predicted_tokens) & Counter(reference_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0

    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(prediction: str, references: list[str]) -> tuple[float, float]:
    """Lấy EM/F1 cao nhất trong danh sách đáp án chuẩn của một câu hỏi."""
    if not references:
        references = [""]
    em = exact_match(prediction, references)
    f1 = max(token_f1(prediction, reference) for reference in references)
    return em, f1


def decode_predictions(
    start_logits: np.ndarray,
    end_logits: np.ndarray,
    feature_ids: list[str],
    feature_offsets: list[list[Any]],
    examples_by_id: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Đổi dự đoán token thành đoạn văn bản lấy nguyên văn từ context.

    Ý tưởng: với mỗi feature (một cửa sổ context), thử ghép 20 vị trí bắt đầu
    tốt nhất với 20 vị trí kết thúc tốt nhất. Bỏ các cặp sai thứ tự, quá dài,
    hoặc nằm ngoài context. Cuối cùng giữ span có tổng logit cao nhất cho mỗi
    câu hỏi, kể cả khi câu hỏi tạo ra nhiều cửa sổ chồng lấn.
    """
    best_answer: dict[str, tuple[float, str]] = {}
    # CLS đại diện cho no-answer. Lấy score thấp nhất qua các cửa sổ vì một
    # cửa sổ khác có thể chứa bằng chứng tốt hơn cho câu hỏi có đáp án.
    best_null_score: dict[str, float] = defaultdict(lambda: math.inf)

    for feature_index, example_id in enumerate(feature_ids):
        example = examples_by_id[example_id]
        offsets = feature_offsets[feature_index]
        starts = start_logits[feature_index]
        ends = end_logits[feature_index]

        # Mã CLS của cả hai tokenizer đứng trong vị trí đầu của cặp đầu vào.
        null_score = float(starts[0] + ends[0])
        best_null_score[example_id] = min(best_null_score[example_id], null_score)

        # Chỉ khảo sát các vị trí có logit cao; giảm mạnh số cặp cần thử.
        top_starts = np.argsort(starts)[-N_BEST_TOKENS:]
        top_ends = np.argsort(ends)[-N_BEST_TOKENS:]
        context = example["context"]

        for start_index in top_starts:
            for end_index in top_ends:
                if end_index < start_index:
                    continue
                if end_index - start_index + 1 > MAX_ANSWER_LENGTH:
                    continue
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue

                char_start = int(offsets[start_index][0])
                char_end = int(offsets[end_index][1])
                if char_end <= char_start:
                    continue

                answer = context[char_start:char_end]
                score = float(starts[start_index] + ends[end_index])
                if score > best_answer.get(example_id, (-math.inf, ""))[0]:
                    best_answer[example_id] = (score, answer)

    predictions: dict[str, str] = {}
    for example_id in examples_by_id:
        answer_score, answer_text = best_answer.get(example_id, (-math.inf, ""))
        null_score = best_null_score.get(example_id, math.inf)

        # score difference dương nghĩa là CLS/no-answer có điểm cao hơn span.
        # Ngưỡng cố định 0 để so sánh công bằng giữa các seed và kiến trúc.
        score_difference = null_score - answer_score
        predictions[example_id] = (
            "" if score_difference > NO_ANSWER_THRESHOLD else answer_text
        )
    return predictions


def make_compute_metrics(
    feature_ids: list[str],
    feature_offsets: list[list[Any]],
    examples_by_id: dict[str, dict[str, Any]],
):
    """Tạo hàm metric cho Trainer; metadata offsets được đóng gói trong hàm."""

    def compute_metrics(evaluation_prediction: Any) -> dict[str, float]:
        # Với mô hình hỏi đáp, predictions là cặp (start_logits, end_logits).
        predictions = evaluation_prediction.predictions
        start_logits, end_logits = predictions[0], predictions[1]

        decoded = decode_predictions(
            start_logits,
            end_logits,
            feature_ids,
            feature_offsets,
            examples_by_id,
        )

        em_scores: list[float] = []
        f1_scores: list[float] = []
        for example_id, example in examples_by_id.items():
            references = example["answers"].get("text", [])
            em, f1 = score_prediction(decoded[example_id], references)
            em_scores.append(em)
            f1_scores.append(f1)

        # Nhân 100 để kết quả dễ đọc theo phần trăm như các báo cáo QA.
        return {
            "exact_match": 100 * float(np.mean(em_scores)),
            "f1": 100 * float(np.mean(f1_scores)),
        }

    return compute_metrics


def load_project_data():
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
        cache_dir=str(DATASET_CACHE_DIR),
    )
    return loaded["train"], loaded["validation"]


def make_training_arguments(output_dir: Path, seed: int) -> TrainingArguments:
    """Tạo cấu hình Trainer, dùng batch vật lý nhỏ để vừa VRAM 16 GB.

    Batch 8 x gradient accumulation 4 tương đương batch hiệu dụng 32. Tích lũy
    gradient cộng dồn đạo hàm qua 4 batch trước khi cập nhật trọng số một lần.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Không phát hiện GPU CUDA. Hãy kích hoạt PyTorch CUDA và kiểm tra nvidia-smi."
        )

    # RTX 5070 Ti hỗ trợ BF16; nếu máy khác không hỗ trợ, dùng FP16 trên GPU.
    use_bf16 = torch.cuda.is_bf16_supported()
    use_fp16 = not use_bf16

    # Tên tham số đổi giữa một số phiên bản Transformers; chọn tên tương thích.
    import inspect

    arguments = {
        "output_dir": str(output_dir),
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
        "seed": seed,
        "data_seed": seed,
    }

    supported = inspect.signature(TrainingArguments.__init__).parameters
    strategy_name = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
    arguments[strategy_name] = "epoch"

    # Bỏ tham số tùy chọn không tồn tại ở phiên bản Transformers đang cài.
    arguments = {key: value for key, value in arguments.items() if key in supported}

    # Một lần huấn luyện lặp từng epoch để có thể so sánh checkpoint tốt nhất.
    return TrainingArguments(**arguments)


def find_completed_run_state(output_dir: Path) -> dict[str, Any] | None:
    """Đọc checkpoint cuối để khôi phục bước ghi kết quả nếu máy từng dừng."""
    checkpoints = sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    if not checkpoints:
        return None

    state_file = checkpoints[-1] / "trainer_state.json"
    if not state_file.exists():
        return None

    state = json.loads(state_file.read_text(encoding="utf-8"))
    eval_rows = [row for row in state.get("log_history", []) if "eval_f1" in row]
    # Chỉ khôi phục nếu đã hoàn tất đủ ba epoch; checkpoint giữa chừng phải train tiếp.
    if eval_rows and max(row.get("epoch", 0) for row in eval_rows) >= 3:
        return state
    return None


def train_one_seed(model_name: str, checkpoint: str, seed: int) -> dict[str, Any]:
    """Huấn luyện một checkpoint với một seed và lưu metric/checkpoint."""
    output_dir = RESULTS_DIR / model_name / f"seed-{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_metrics = output_dir / "metrics.json"
    saved_model = output_dir / "best_model"
    if saved_metrics.exists() and saved_model.exists():
        # Cho phép chạy lại cả experiment mà không train lại các seed đã xong.
        print(f"Đã có kết quả seed {seed}; giữ nguyên checkpoint đã lưu.")
        return json.loads(saved_metrics.read_text(encoding="utf-8"))

    # Seed phải được đặt trước khi khởi tạo mô hình để kiểm soát trọng số ban đầu.
    set_seed(seed)
    print(f"\n{'=' * 72}\nModel: {checkpoint} | seed: {seed}\n{'=' * 72}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_examples, dev_examples = load_project_data()

    # Fast tokenizer cung cấp offset mapping cần thiết để khôi phục span văn bản.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError(f"Checkpoint {checkpoint} không cung cấp Fast Tokenizer.")

    # Ánh xạ từng câu trả lời từ ký tự trong context thành chỉ số token.
    train_features = train_examples.map(
        lambda batch: prepare_train_features(batch, tokenizer),
        batched=True,
        remove_columns=train_examples.column_names,
        desc="Đổi train examples thành token features",
    )

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

    # Nếu quá trình train hoàn tất nhưng dừng lúc ghi summary, tái dùng best_model.
    best_model_dir = output_dir / "best_model"
    recovered_state = find_completed_run_state(output_dir)
    if recovered_state and best_model_dir.exists():
        print("Đã thấy checkpoint hoàn tất; nạp best_model để hoàn thiện tệp kết quả.")
        model = AutoModelForQuestionAnswering.from_pretrained(str(best_model_dir))
    else:
        # Mỗi seed khởi tạo mới từ checkpoint gốc, không kế thừa seed trước.
        model = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
    compute_metrics = make_compute_metrics(
        feature_ids, feature_offsets, examples_by_id
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Tên đối số tokenizer đổi thành processing_class ở các bản mới hơn.
    import inspect

    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    tokenizer_argument = (
        {"processing_class": tokenizer}
        if "processing_class" in trainer_parameters
        else {"tokenizer": tokenizer}
    )

    trainer = Trainer(
        model=model,
        args=make_training_arguments(output_dir, seed),
        train_dataset=train_features,
        eval_dataset=dev_model_features,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        **tokenizer_argument,
    )

    # Huấn luyện: Trainer tính loss start/end, lan truyền gradient và cập nhật trọng số.
    if recovered_state and best_model_dir.exists():
        # Trường hợp này chỉ khôi phục phần metrics/predictions sau khi train đủ epoch.
        training_seconds = None
    else:
        started_at = time.time()
        trainer.train()
        training_seconds = time.time() - started_at

    # load_best_model_at_end=True đã nạp lại checkpoint có F1 dev cao nhất.
    final_metrics = trainer.evaluate()
    if not recovered_state:
        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(best_model_dir)

    history = (
        recovered_state.get("log_history", [])
        if recovered_state
        else trainer.state.log_history
    )
    scored_epochs = [row for row in history if "eval_f1" in row]
    best_epoch_row = max(scored_epochs, key=lambda row: row["eval_f1"])

    result = {
        "model": model_name,
        "checkpoint": checkpoint,
        "seed": seed,
        "best_checkpoint": (
            recovered_state.get("best_model_checkpoint")
            if recovered_state
            else trainer.state.best_model_checkpoint
        ),
        "best_epoch": best_epoch_row.get("epoch"),
        "best_eval_f1": float(final_metrics.get("eval_f1", 0.0)),
        "best_eval_exact_match": float(final_metrics.get("eval_exact_match", 0.0)),
        "training_seconds": round(training_seconds, 2) if training_seconds is not None else None,
        "effective_batch_size": 4 * 8,
        "epochs_requested": 3,
        "bf16": torch.cuda.is_bf16_supported(),
    }

    # JSON giúp người mới xem kết quả, seed, checkpoint và thời lượng sau mỗi lần chạy.
    result_file = output_dir / "metrics.json"
    result_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Lưu dự đoán cuối cùng để tiện kiểm tra thủ công và phân tích lỗi.
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
    (output_dir / "predictions.json").write_text(
        json.dumps(prediction_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Kết quả seed {seed}: EM={result['best_eval_exact_match']:.2f}, F1={result['best_eval_f1']:.2f}")
    return result


def run_experiment(models: list[str], seeds: list[int]) -> None:
    """Chạy lần lượt các model/seed, rồi tổng hợp trung bình và seed tốt nhất."""
    all_results: list[dict[str, Any]] = []
    for model_name in models:
        checkpoint = MODEL_CHECKPOINTS[model_name]
        for seed in seeds:
            all_results.append(train_one_seed(model_name, checkpoint, seed))

    # Tổng hợp riêng từng kiến trúc để không trộn metric giữa các model.
    summary: dict[str, Any] = {"runs": all_results, "models": {}}
    for model_name in models:
        model_runs = [row for row in all_results if row["model"] == model_name]
        best_run = max(
            model_runs,
            key=lambda row: (row["best_eval_f1"], row["best_eval_exact_match"]),
        )
        f1_values = [row["best_eval_f1"] for row in model_runs]
        em_values = [row["best_eval_exact_match"] for row in model_runs]
        summary["models"][model_name] = {
            "best_seed": best_run["seed"],
            "best_f1": best_run["best_eval_f1"],
            "best_exact_match": best_run["best_eval_exact_match"],
            "mean_f1": float(np.mean(f1_values)),
            "std_f1": float(np.std(f1_values)),
            "mean_exact_match": float(np.mean(em_values)),
            "std_exact_match": float(np.std(em_values)),
        }

        # Mỗi notebook kiến trúc có file summary riêng để đọc kết quả độc lập.
        model_summary_path = RESULTS_DIR / model_name / "summary.json"
        model_summary_path.parent.mkdir(parents=True, exist_ok=True)
        model_summary_path.write_text(
            json.dumps(summary["models"][model_name], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\nTổng hợp (F1/EM theo phần trăm):")
    for model_name, scores in summary["models"].items():
        print(
            f"{model_name}: seed tốt nhất={scores['best_seed']}, "
            f"F1={scores['best_f1']:.2f}, EM={scores['best_exact_match']:.2f}; "
            f"F1 trung bình={scores['mean_f1']:.2f} ± {scores['std_f1']:.2f}"
        )
    print(f"Đã lưu tổng hợp tại: {summary_path}")


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
    run_experiment(selected_models, args.seeds)
