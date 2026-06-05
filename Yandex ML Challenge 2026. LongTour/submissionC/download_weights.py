"""Download base weights into ./weights BEFORE building the image (run locally,
needs internet). The grading container has NO internet, so weights must be baked
into the image via `COPY . .`.

    pip install -U "huggingface_hub>=0.23"
    python3 download_weights.py            # default Qwen3-1.7B
    MODEL_REPO=Qwen/Qwen3-4B python3 download_weights.py   # stretch variant

If you fine-tune with train_sft.py, that script writes merged weights to ./weights
directly and you can SKIP this download step.
"""
import os
from huggingface_hub import snapshot_download

REPO_ID = os.environ.get("MODEL_REPO", "Qwen/Qwen3-1.7B")
LOCAL_DIR = "weights"


def main():
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "tokenizer*", "*.jinja"],
    )
    print(f"weights downloaded to: {path}")


if __name__ == "__main__":
    main()
