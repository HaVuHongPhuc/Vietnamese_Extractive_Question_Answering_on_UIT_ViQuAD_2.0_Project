"""Kiểm tra môi trường Challenge 1; không tải backbone và không huấn luyện."""

import argparse
import importlib
from importlib import metadata
import sys


def main() -> None:
    """Ví dụ: tensor token [2, 8] đi qua model nhỏ thành logits [2, 3]."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()

    print("Python:", sys.version.split()[0])
    print("Interpreter:", sys.executable)

    # Import thực tế để phát hiện cả thiếu package lẫn lỗi binary dependency.
    packages = {
        "torch": "torch",
        "transformers": "transformers",
        "accelerate": "accelerate",
        "datasets": "datasets",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "sentencepiece": "sentencepiece",
        "protobuf": "google.protobuf",
        "tqdm": "tqdm",
        "ipykernel": "ipykernel",
        "jupyterlab": "jupyterlab",
    }
    for package_name, module_name in packages.items():
        importlib.import_module(module_name)
        print(f"{package_name}=={metadata.version(package_name)}")

    import torch
    from sklearn.metrics import f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        DistilBertConfig,
        DistilBertForSequenceClassification,
        Trainer,
        TrainingArguments,
        get_linear_schedule_with_warmup,
    )

    # Xác nhận các API mà notebook baseline/K-fold sẽ sử dụng import được.
    apis = (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        get_linear_schedule_with_warmup,
        StratifiedGroupKFold,
        f1_score,
    )
    print("Notebook APIs imported:", len(apis))

    cuda_available = torch.cuda.is_available()
    print("CUDA available:", cuda_available)
    print("PyTorch CUDA runtime:", torch.version.cuda)
    if args.require_cuda and not cuda_available:
        raise RuntimeError(
            "CUDA is unavailable. Check the NVIDIA driver and PyTorch build."
        )

    device = torch.device("cuda" if cuda_available else "cpu")
    if cuda_available:
        print("GPU:", torch.cuda.get_device_name(0))
        print("BF16 supported:", torch.cuda.is_bf16_supported())

    # Model ngẫu nhiên, cực nhỏ chỉ để kiểm tra forward trên máy hiện tại.
    # Đây không phải pretrained backbone hoặc model dùng để nộp cuộc thi.
    config = DistilBertConfig(
        vocab_size=32,
        dim=16,
        hidden_dim=32,
        n_layers=1,
        n_heads=2,
        max_position_embeddings=32,
        num_labels=3,
    )
    model = DistilBertForSequenceClassification(config).to(device).eval()

    # input_ids và attention_mask shape: [batch_size=2, sequence_length=8].
    input_ids = torch.randint(0, 32, (2, 8), device=device)
    attention_mask = torch.ones_like(input_ids)

    use_bf16 = cuda_available and torch.cuda.is_bf16_supported()
    with torch.inference_mode():
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            ).logits

    # logits shape: [batch_size=2, num_labels=3].
    assert tuple(logits.shape) == (2, 3), logits.shape
    assert torch.isfinite(logits).all().item(), "Non-finite logits"
    print("PASS: tiny inference; logits shape:", tuple(logits.shape))
    print("PASS: environment checks completed; no training or model download.")


if __name__ == "__main__":
    main()