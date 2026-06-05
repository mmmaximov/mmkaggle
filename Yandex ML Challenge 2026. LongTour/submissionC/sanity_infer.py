"""Quick sanity check of the (merged) model on a few questions — run on Colab
AFTER train_sft.py, BEFORE building the Docker image, to eyeball answer quality
without spinning up vLLM or burning a submission attempt.

    python sanity_infer.py --model weights --data dataset_ml_challenge.parquet
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

DEFAULT_QS = [
    "сколько будет 3/4 + 7/8",
    "разбери слово 'подоконник' по составу",
    "как пользоваться Present Perfect, объясни с примерами",
    "уравняй реакцию Fe + HNO3",
    "напиши короткое сочинение про осень",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="weights")
    p.add_argument("--data", default="")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--max_new", type=int, default=1024)
    args = p.parse_args()

    qs = DEFAULT_QS[: args.n]
    if args.data:
        import pandas as pd
        df = pd.read_parquet(args.data)
        qs = df["query"].dropna().sample(args.n, random_state=1).tolist()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    for q in qs:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)
        ids = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=args.max_new,
                                 do_sample=False, temperature=None, top_p=None,
                                 repetition_penalty=1.05,
                                 pad_token_id=tok.eos_token_id)
        ans = tok.decode(out[0][ids["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        print("=" * 70)
        print("Q:", q)
        print("-" * 5)
        print("A:", ans[:1200])
        print()


if __name__ == "__main__":
    main()
