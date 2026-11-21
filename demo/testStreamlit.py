from pathlib import Path

import streamlit as st
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="UIT-ViQuAD Demo - Machine Reading Comprehension",
    page_icon="📖",
    layout="wide"
)

# Tiêu đề bài báo cáo
st.title("📖 Trình diễn hệ thống Đọc hiểu máy tiếng Việt")
st.markdown("### Bài báo: *A Vietnamese Dataset for Evaluating Machine Reading Comprehension (COLING 2020)*")
st.write("Giao diện mô phỏng cách hoạt động của mô hình đọc hiểu trên bộ dữ liệu UIT-ViQuAD.")

# Lấy mô hình fine-tune trên UIT-ViQuAD đã lưu trong project.
project_root = Path(__file__).resolve().parents[1]
local_model_dir = (
    project_root / "results" / "finetune" / "xlm-roberta"
    / "seed-42" / "best_model"
)

# Nếu checkpoint local không dùng được, dùng model đa ngôn ngữ SQuAD 2 làm dự phòng.
FALLBACK_MODEL = "deepset/xlm-roberta-base-squad2"


@st.cache_resource
def load_qa_model(model_name: str):
    """Tải tokenizer và mô hình QA trực tiếp, không dùng pipeline đã bị bỏ ở Transformers 5."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def answer_question(tokenizer, model, device, question: str, context: str) -> dict:
    """Chọn span start/end tốt nhất, hỗ trợ context dài bằng sliding window."""
    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=384,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
        return_tensors="pt",
    )

    # Chỉ đưa các tensor mà mô hình nhận làm đầu vào; offsets/metadata không phải input model.
    model_inputs = {
        name: value.to(device)
        for name, value in encoded.items()
        if name in tokenizer.model_input_names
    }

    best_answer_score = float("-inf")
    best_answer = ""
    best_start = 0
    best_end = 0
    best_answer_confidence = 0.0
    best_null_score = float("inf")
    best_null_confidence = 0.0

    with torch.inference_mode():
        # Xử lý từng cửa sổ một để tránh đưa context rất dài vào GPU cùng lúc.
        for feature_index in range(model_inputs["input_ids"].shape[0]):
            one_feature = {
                name: value[feature_index : feature_index + 1]
                for name, value in model_inputs.items()
            }
            outputs = model(**one_feature)
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]
            start_probs = torch.softmax(start_logits, dim=-1)
            end_probs = torch.softmax(end_logits, dim=-1)

            input_ids = one_feature["input_ids"][0].tolist()
            cls_id = tokenizer.cls_token_id
            if cls_id is None or cls_id not in input_ids:
                raise ValueError("Không tìm thấy token CLS để kiểm tra no-answer.")
            cls_index = input_ids.index(cls_id)

            null_score = float((start_logits[cls_index] + end_logits[cls_index]).item())
            if null_score < best_null_score:
                best_null_score = null_score
                best_null_confidence = float(
                    (start_probs[cls_index] * end_probs[cls_index]).item()
                )

            sequence_ids = encoded.sequence_ids(feature_index)
            offsets = encoded["offset_mapping"][feature_index].tolist()
            candidate_count = min(20, start_logits.numel())
            top_starts = torch.topk(start_logits, candidate_count).indices.tolist()
            top_ends = torch.topk(end_logits, candidate_count).indices.tolist()

            for start_index in top_starts:
                for end_index in top_ends:
                    if end_index < start_index:
                        continue
                    if end_index - start_index + 1 > 30:
                        continue
                    if sequence_ids[start_index] != 1 or sequence_ids[end_index] != 1:
                        continue

                    char_start = int(offsets[start_index][0])
                    char_end = int(offsets[end_index][1])
                    if char_end <= char_start:
                        continue

                    answer_text = context[char_start:char_end]
                    if not answer_text.strip():
                        continue

                    span_score = float(
                        (start_logits[start_index] + end_logits[end_index]).item()
                    )
                    if span_score > best_answer_score:
                        best_answer_score = span_score
                        best_answer = answer_text
                        best_start = char_start
                        best_end = char_end
                        best_answer_confidence = float(
                            (start_probs[start_index] * end_probs[end_index]).item()
                        )

    # Nếu điểm CLS/no-answer thắng span tốt nhất, trả lời rỗng.
    if best_null_score - best_answer_score > 0.0:
        return {
            "answer": "",
            "start": 0,
            "end": 0,
            "score": best_null_confidence,
        }

    return {
        "answer": best_answer,
        "start": best_start,
        "end": best_end,
        "score": best_answer_confidence,
    }


# Ưu tiên mô hình đã fine-tune trên UIT-ViQuAD của project.
model_candidates = []
if local_model_dir.is_dir():
    model_candidates.append(str(local_model_dir))
model_candidates.append(FALLBACK_MODEL)

qa_model = None
qa_tokenizer = None
qa_device = None
active_model = None
load_errors = []
for model_name in model_candidates:
    try:
        qa_tokenizer, qa_model, qa_device = load_qa_model(model_name)
        active_model = model_name
        break
    except Exception as error:
        load_errors.append(f"{model_name}: {error}")

if qa_model is None:
    st.error("Không tải được mô hình hỏi đáp.")
    st.info(
        "Kiểm tra checkpoint local hoặc kết nối Internet để tải mô hình dự phòng. "
        "Khởi chạy bằng: python -m streamlit run demo/testStreamlit.py"
    )
    with st.expander("Chi tiết lỗi tải mô hình"):
        st.code("\n\n".join(load_errors))
    st.stop()

st.caption(f"Mô hình đang dùng: {active_model} ({qa_device})")

# Giao diện chính chia làm 2 cột
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📝 Dữ liệu đầu vào (Context & Question)")
    
    # Đoạn văn mẫu dựa trên Wikipedia (giống cách thu thập của UIT-ViQuAD)
    sample_context = (
        "Trường Đại học Công nghệ Thông tin (ĐHQG-HCM) là một trung tâm đào tạo "
        "và nghiên cứu khoa học công nghệ uy tín. Nhóm tác giả Nguyễn và các cộng sự "
        "đã công bố bộ dữ liệu UIT-ViQuAD tại hội nghị COLING 2020 nhằm thúc đẩy "
        "nghiên cứu xử lý ngôn ngữ tự nhiên cho tiếng Việt. Bộ dữ liệu chứa hơn 23.000 "
        "cặp câu hỏi-đáp được gán nhãn thủ công từ các bài viết trên Wikipedia."
    )
    
    context = st.text_area("Nhập đoạn văn bản (Context):", value=sample_context, height=200)
    question = st.text_input("Nhập câu hỏi (Question):", value="Bộ dữ liệu UIT-ViQuAD được công bố năm nào?")

with col2:
    st.subheader("🎯 Kết quả trích xuất câu trả lời")
    
    if st.button("Tìm câu trả lời", type="primary"):
        if context and question:
            with st.spinner("Mô hình đang đọc hiểu văn bản..."):
                # Chạy mô hình QA và hiển thị span được trích xuất.
                result = answer_question(
                    qa_tokenizer, qa_model, qa_device, question, context
                )

                if result["answer"]:
                    st.success(f"**Câu trả lời:** {result['answer']}")
                    start_idx = result["start"]
                    end_idx = result["end"]
                    highlighted_context = (
                        context[:start_idx]
                        + f"**[{context[start_idx:end_idx]}]**"
                        + context[end_idx:]
                    )
                    st.markdown(f"**Vị trí tìm thấy:** {highlighted_context}")
                else:
                    st.warning("Mô hình dự đoán context không chứa câu trả lời.")

                st.metric(
                    label="Điểm xác suất start × end",
                    value=f"{result['score']:.4f}",
                )
        else:
            st.warning("Vui lòng điền đầy đủ thông tin đoạn văn và câu hỏi.")

# Phần thông tin về bộ dữ liệu UIT-ViQuAD
st.divider()
st.subheader("📊 Thống kê về bộ dữ liệu UIT-ViQuAD (COLING 2020)")
metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
metrics_col1.metric("Số lượng câu hỏi", "23,074")
metrics_col2.metric("Số đoạn văn (Wikipedia)", "5,109")
metrics_col3.metric("Số bài viết", "174")
