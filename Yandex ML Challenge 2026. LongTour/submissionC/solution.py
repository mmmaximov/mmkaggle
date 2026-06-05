"""Inference for Task C — Effective Inference On School Questions.

Reads  /workspace/input.pickle : list of {"rid": int, "question": str}
Writes /workspace/output.json  : list of {"rid": int, "answer": str}

Strategy vs baseline:
  * stronger base model (Qwen3-1.7B by default, swap via MODEL_DIR)
  * Russian system prompt that elicits the structured, LaTeX, "Ответ:"-style
    answers the judge rewards (matches the training references)
  * larger MAX_NEW_TOKENS so answers aren't truncated (baseline 512 cut ~25-30%)
  * robust: every rid gets exactly one answer; empty string on any failure.
"""
import json
import os
import pickle
import sys
import time
import hashlib

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def weights_fingerprint(d):
    """Short content-sensitive fingerprint of the weights dir, so the run log
    unambiguously shows WHICH model is loaded (base vs fine-tuned)."""
    try:
        files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
        h = hashlib.sha256()
        meta = []
        for fn in files:
            sz = os.path.getsize(os.path.join(d, fn))
            meta.append(f"{fn}:{sz}")
            h.update(fn.encode()); h.update(str(sz).encode())
        st = [f for f in files if f.endswith(".safetensors")]
        if st:  # hash first 8MB of the largest shard -> sensitive to actual weights
            big = max(st, key=lambda f: os.path.getsize(os.path.join(d, f)))
            with open(os.path.join(d, big), "rb") as fh:
                h.update(fh.read(8 * 1024 * 1024))
        return h.hexdigest()[:16], meta
    except Exception as e:
        return f"err:{e}", []


MODEL_DIR = os.environ.get("MODEL_DIR", "./weights")
INPUT_PATH = os.environ.get("INPUT_PATH", "input.pickle")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "output.json")

MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 1024))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", 4096))
GPU_MEM_UTIL = float(os.environ.get("GPU_MEM_UTIL", 0.92))
# optional online quantization for bigger models (e.g. "fp8" on L4/Ada, or "awq").
# fp8 lets Qwen3-4B fit time + <10GB submission; leave empty for plain bf16.
QUANT = (os.environ.get("QUANT", "").strip().lower() or None)

SYSTEM_PROMPT = (
    "Ты — опытный школьный преподаватель и репетитор. Твоя задача — дать "
    "точный, полный и понятный ответ на школьный вопрос на русском языке.\n"
    "Следуй правилам:\n"
    "1. Для задач по математике, физике, химии и информатике решай пошагово, "
    "поясняй каждый шаг и записывай формулы и вычисления в LaTeX: внутри текста "
    "через $...$, отдельные формулы через $$...$$.\n"
    "2. Если задача имеет числовой или конкретный ответ, заверши решение "
    "отдельной строкой вида «**Ответ:** ...».\n"
    "3. Для вопросов по русскому языку, литературе, истории, биологии, "
    "географии и обществознанию давай связный, хорошо структурированный ответ; "
    "если просят сочинение, эссе или рассуждение — пиши развёрнутый цельный текст.\n"
    "4. Опирайся на школьную программу, не выдумывай факты.\n"
    "5. Пиши только сам ответ по существу, без вступлений вроде «Конечно» и без "
    "обращений к пользователю."
)


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def write_output(rows, answers):
    result = [{"rid": row["rid"], "answer": answers.get(row["rid"], "")}
              for row in rows]
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)


def main() -> None:
    t0 = time.time()
    with open(INPUT_PATH, "rb") as f:
        rows = pickle.load(f)
    log(f"loaded {len(rows)} questions")

    # Write a valid placeholder immediately, so that even if model init or
    # generation fails later, /workspace/output.json still exists (valid file
    # with empty answers) instead of a hard RuntimeError / missing-file.
    answers = {row["rid"]: "" for row in rows}
    write_output(rows, answers)

    fp, meta = weights_fingerprint(MODEL_DIR)
    log(f"WEIGHTS_FINGERPRINT={fp}")
    log(f"WEIGHTS_FILES={meta}")

    # Fail loudly with diagnostics if weights are missing (most common RE cause).
    if not os.path.isdir(MODEL_DIR) or not any(
            fn.endswith(".safetensors") for fn in os.listdir(MODEL_DIR)):
        log(f"FATAL: no model weights in '{MODEL_DIR}'. Contents: "
            f"{os.listdir(MODEL_DIR) if os.path.isdir(MODEL_DIR) else 'DIR MISSING'}")
        log("Run download_weights.py (or train_sft.py) and bake ./weights into the image.")
        return  # placeholder output.json already written

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

    # Build chat prompts (system + user). enable_thinking=False keeps it fast
    # enough for the 15-min / 4000-question budget.
    prompts = []
    for row in rows:
        q = (row.get("question") or "").strip()
        msgs = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": q},
        ]
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # some templates don't accept enable_thinking
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        prompts.append(text)

    llm_kwargs = dict(
        model=MODEL_DIR,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        tokenizer_mode="auto",
        enforce_eager=False,
        seed=0,
    )
    if QUANT:
        llm_kwargs["quantization"] = QUANT
        log(f"using quantization={QUANT}")
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(
        temperature=0.0,
        top_k=-1,
        max_tokens=MAX_NEW_TOKENS,
        repetition_penalty=1.05,  # gently avoid degenerate loops on a small model
    )

    outputs = llm.generate(prompts, sampling_params=sampling)

    # Map results back by index; guarantee one answer per rid.
    for row, out in zip(rows, outputs):
        try:
            answers[row["rid"]] = out.outputs[0].text.strip()
        except Exception:
            pass

    write_output(rows, answers)
    log(f"done: {len(rows)} answers in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
