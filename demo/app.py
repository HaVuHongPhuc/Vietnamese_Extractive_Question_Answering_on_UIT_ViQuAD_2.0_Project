"""Demo Streamlit cho câu hỏi trích xuất trên một đoạn văn tiếng Việt."""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preprocessing import prepare_validation_features
from src.evaluation import decode_predictions

PHOBERT_DIR = PROJECT_ROOT / "results/finetune/phobert/seed-1024/best_model"
XLMR_DIR = PROJECT_ROOT / "results/finetune/runs/run-fe5f5f5b27578bd4/xlm-roberta/seed-42/best_model"
LOGGER = logging.getLogger(__name__)
DEMO_ERRORS = (ImportError, OSError, ValueError, RuntimeError, MemoryError)
SAMPLE_IDS = {
    "Có đáp án": "0017-0029-0002",
    "Không có đáp án": "0001-0013-0002",
    "Đoạn văn dài": "0017-0041-0003",
}


def checkpoint_path(default: Path, env_key: str) -> Path:
    """Cho phép thành viên chỉ tới bản checkpoint cục bộ đã nhận."""
    value = os.environ.get(env_key)
    return (PROJECT_ROOT / Path(value).expanduser()).resolve() if value else default


@st.cache_resource
def load_checkpoint(directory: str, use_fast: bool, weight_stamp: int, device_name: str):
    """Nạp trọng số cục bộ một lần cho mỗi checkpoint."""
    path = Path(directory)
    for filename in ("config.json", "model.safetensors"):
        if not (path / filename).is_file():
            raise FileNotFoundError(f"Thiếu {filename} trong {path}")
    tokenizer = AutoTokenizer.from_pretrained(directory, use_fast=use_fast, local_files_only=True)
    if use_fast and not tokenizer.is_fast:
        raise ValueError("XLM-R cần fast tokenizer để lấy offset.")
    if not use_fast and tokenizer.__class__.__name__ != "PhobertTokenizer":
        raise ValueError("Checkpoint chính phải dùng PhobertTokenizer.")
    model = AutoModelForQuestionAnswering.from_pretrained(directory, local_files_only=True)
    device = torch.device(device_name)
    model.to(device)
    model.eval()
    return tokenizer, model, device


def load_model(path: Path, use_fast: bool, device_name: str):
    # Đổi khóa cache nếu người dùng thay tệp trọng số ở cùng đường dẫn.
    weights = path / "model.safetensors"
    stamp = weights.stat().st_mtime_ns if weights.is_file() else 0
    return load_checkpoint(str(path), use_fast, stamp, device_name)


def predict(question: str, context: str, tokenizer, model, device, use_phobert: bool):
    """Tính logit cho từng cửa sổ rồi giải mã theo cùng quy tắc đã đánh giá."""
    stage = "Tách từ và tạo cửa sổ" if use_phobert else "Tokenize và tạo cửa sổ"
    try:
        if use_phobert:
            from demo.phobert_inference import prepare_phobert_windows

            model_inputs, offsets, decode_context = prepare_phobert_windows(question, context, tokenizer)
        else:
            batch = {"id": ["demo"], "question": [question], "context": [context]}
            features = prepare_validation_features(batch, tokenizer)
            model_inputs = {key: features[key] for key in tokenizer.model_input_names if key in features}
            offsets = features["offset_mapping"]
            decode_context = context

        stage = f"Chạy mô hình trên {device}"
        start_logits, end_logits = [], []
        with torch.inference_mode():
            for window_index in range(len(offsets)):
                # Mỗi lượt suy luận dùng một cửa sổ để hạn chế bộ nhớ.
                one_window = {
                    name: torch.tensor([rows[window_index]], device=device)
                    for name, rows in model_inputs.items()
                }
                out = model(**one_window)
                start_logits.append(out.start_logits[0].cpu().float().numpy())
                end_logits.append(out.end_logits[0].cpu().float().numpy())

        stage = "Giải mã đáp án"
        result = decode_predictions(
            np.stack(start_logits),
            np.stack(end_logits),
            ["demo"] * len(offsets),
            offsets,
            {"demo": {"context": decode_context}},
        )["demo"]
    except DEMO_ERRORS as error:
        raise RuntimeError(f"{stage}: {error}") from error
    # PhoBERT dùng offset trên context đã tách. Sau khi chọn span mới đổi "_".
    return (result.replace("_", " ") if use_phobert else result), len(offsets)


def predict_with_checkpoint(question: str, context: str, path: Path, device_name: str, use_phobert: bool):
    try:
        tokenizer, model, device = load_model(path, use_fast=not use_phobert, device_name=device_name)
    except DEMO_ERRORS as error:
        raise RuntimeError(f"Nạp checkpoint lên {device_name}: {error}") from error
    return predict(question, context, tokenizer, model, device, use_phobert)


@st.cache_data
def load_dev_examples(data_file: str, file_stamp: int) -> dict:
    """Lấy ba ca thử từ dev đã đánh giá để người dùng kiểm tra demo."""
    data = Path(data_file)
    if not data.is_file():
        return {}
    selected = {}
    with data.open(encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            if row["id"] in SAMPLE_IDS.values():
                selected[row["id"]] = row
    return selected


def main() -> None:
    st.set_page_config(page_title="Hỏi đáp trích xuất tiếng Việt", layout="wide")
    st.title("Hỏi đáp trích xuất tiếng Việt")
    st.write("Nhập một đoạn văn và câu hỏi. Mô hình trích câu trả lời từ đoạn văn hoặc trả về không có đáp án.")

    primary_dir = checkpoint_path(PHOBERT_DIR, "VIQUAD_PHOBERT_CHECKPOINT")
    fallback_dir = checkpoint_path(XLMR_DIR, "VIQUAD_XLMR_CHECKPOINT")
    mode = st.radio("Mô hình", ["Tự động", "Kiểm tra XLM-R dự phòng"], horizontal=True)
    use_cpu = st.checkbox("Chạy trên CPU")
    st.caption("Mô hình chính: PhoBERT-base-v2, seed 1024. Dự phòng: XLM-RoBERTa-base, seed 42.")

    chosen = st.selectbox("Ca thử có sẵn từ dev", ("Tự nhập", *SAMPLE_IDS))
    if chosen != "Tự nhập":
        dev_file = PROJECT_ROOT / "data" / "processed" / "dev.json"
        stamp = dev_file.stat().st_mtime_ns if dev_file.is_file() else 0
        row = load_dev_examples(str(dev_file), stamp).get(SAMPLE_IDS[chosen])
        if row is None:
            st.caption("Thiếu file dev cho ca thử này. Bạn vẫn có thể tự nhập đoạn văn và câu hỏi.")
        else:
            if st.button("Nạp ca thử"):
                st.session_state["qa_context"] = row["context"]
                st.session_state["qa_question"] = row["question"]
            gold = " | ".join(row["answers"]["text"]) or "Không có đáp án"
            st.caption(f"Mẫu {row['id']} · Đáp án chuẩn: {gold}")

    with st.form("qa"):
        context = st.text_area("Đoạn văn", height=240, max_chars=18000, key="qa_context")
        question = st.text_input("Câu hỏi", max_chars=500, key="qa_question")
        submitted = st.form_submit_button("Trả lời")

    if not submitted:
        return
    if not context.strip() or not question.strip():
        st.info("Nhập đủ đoạn văn và câu hỏi.")
        return

    device_name = "cuda" if not use_cpu and torch.cuda.is_available() else "cpu"
    use_phobert = mode == "Tự động"
    fallback_reason = "Đang kiểm tra mô hình dự phòng theo lựa chọn của bạn."
    with st.spinner("Đang nạp mô hình và tìm đáp án..."):
        if use_phobert:
            try:
                answer, window_count = predict_with_checkpoint(
                    question, context, primary_dir, device_name, use_phobert=True,
                )
            except RuntimeError as error:
                LOGGER.exception("PhoBERT gặp lỗi khi xử lý yêu cầu")
                fallback_reason = f"PhoBERT gặp lỗi ở bước {error}"
                use_phobert = False

        if not use_phobert:
            st.warning(f"Dùng XLM-R dự phòng. {fallback_reason}")
            try:
                answer, window_count = predict_with_checkpoint(
                    question, context, fallback_dir, device_name, use_phobert=False,
                )
            except RuntimeError as error:
                LOGGER.exception("XLM-R gặp lỗi khi xử lý yêu cầu")
                st.error(f"XLM-R gặp lỗi ở bước {error}")
                return

    active_path = primary_dir if use_phobert else fallback_dir
    active_name = "PhoBERT-base-v2 · seed 1024" if use_phobert else "XLM-RoBERTa-base · seed 42"
    st.write(f"Checkpoint đang dùng: **{active_name}**")
    st.caption(f"Thiết bị: {device_name} · {active_path}")
    st.caption(f"Đã xử lý {window_count} cửa sổ.")
    if answer:
        st.success(answer)
        if use_phobert:
            st.caption("PhoBERT lấy câu trả lời từ đoạn văn đã tách từ, sau đó chuyển dấu gạch dưới thành khoảng trắng.")
    else:
        st.info("Mô hình dự đoán không có đáp án trong đoạn văn.")


if __name__ == "__main__":
    main()
