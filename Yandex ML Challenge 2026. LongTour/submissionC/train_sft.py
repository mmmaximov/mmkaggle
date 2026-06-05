"""LoRA SFT (distillation) for Task C — run on Colab/rented GPU (T4/L4/A100).

Fine-tunes the base model on the 10k query->answer pairs so the small model
reproduces the structured, Russian, LaTeX, "Ответ:"-style answers the judge
rewards. Trains LoRA, MERGES it into the base, and writes merged weights to
./weights — exactly the folder solution.py / Dockerfile expect.

Colab usage:
    pip install -U transformers==4.56.1 peft datasets accelerate pyarrow
    python train_sft.py --data dataset_ml_challenge.parquet --base Qwen/Qwen3-1.7B \
        --out weights --epochs 2

The loss is computed ONLY on the assistant answer (prompt tokens are masked),
and the system prompt is identical to solution.py so train/inference match.
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from peft import LoraConfig, get_peft_model

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


class SFTDataset(Dataset):
    """Tokenizes (system+user) prompt and answer; masks loss on the prompt."""

    def __init__(self, df, tokenizer, max_len=2048):
        self.tok = tokenizer
        self.max_len = max_len
        self.rows = df.reset_index(drop=True)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        q = str(self.rows.loc[i, "query"]).strip()
        a = str(self.rows.loc[i, "answer"]).strip()
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]
        try:
            prompt = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
        except TypeError:
            prompt = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True)

        prompt_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        answer_ids = self.tok(a, add_special_tokens=False)["input_ids"]
        eos = self.tok.eos_token_id
        answer_ids = answer_ids + [eos]

        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids[:]  # train only on answer

        # enforce max_len by keeping the tail (answer sits at the end, so it is
        # preserved; over-long prompt is trimmed from the left). labels stay aligned.
        if len(input_ids) > self.max_len:
            input_ids = input_ids[-self.max_len:]
            labels = labels[-self.max_len:]
        return {"input_ids": input_ids, "labels": labels,
                "attention_mask": [1] * len(input_ids)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="dataset_ml_challenge.parquet")
    p.add_argument("--base", default="Qwen/Qwen3-1.7B")
    p.add_argument("--out", default="weights")
    p.add_argument("--max_len", type=int, default=2048)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--bs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--eval_frac", type=float, default=0.02)
    args = p.parse_args()

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["query", "answer"]).drop_duplicates("query")
    df["query"] = df["query"].str.strip()
    df["answer"] = df["answer"].str.strip()
    df = df[(df["answer"].str.len() > 0) & (df["query"].str.len() > 0)]
    df = df.reset_index(drop=True)
    print(f"training rows after clean: {len(df)}")

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # quick token-length report to sanity-check max_len / truncation share
    sample = df["answer"].sample(min(500, len(df)), random_state=0)
    alens = np.array([len(tok(a, add_special_tokens=False)["input_ids"]) for a in sample])
    trunc_share = float((alens > args.max_len - 200).mean())
    print(f"answer tokens med/95p/99p = {np.median(alens):.0f}/"
          f"{np.percentile(alens,95):.0f}/{np.percentile(alens,99):.0f}; "
          f"~{trunc_share*100:.1f}% may be truncated at max_len={args.max_len}")

    # eval split for overfit monitoring
    n_eval = max(1, int(len(df) * args.eval_frac))
    df_eval = df.iloc[:n_eval].reset_index(drop=True)
    df_train = df.iloc[n_eval:].reset_index(drop=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_ds = SFTDataset(df_train, tok, max_len=args.max_len)
    eval_ds = SFTDataset(df_eval, tok, max_len=args.max_len)
    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100,
                                      padding="longest")

    targs = TrainingArguments(
        output_dir="sft_ckpt",
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="no",
        report_to=[],
        gradient_checkpointing=True,
        optim="adamw_torch",
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds,
                      eval_dataset=eval_ds, data_collator=collator)
    trainer.train()

    # save the LoRA adapter separately (cheap insurance), then merge + save full
    model.save_pretrained("lora_adapter")
    print("merging LoRA and saving merged weights...")
    merged = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    merged.save_pretrained(args.out, safe_serialization=True)
    tok.save_pretrained(args.out)
    print(f"merged model saved to ./{args.out} — bake this into the Docker image")


if __name__ == "__main__":
    main()
