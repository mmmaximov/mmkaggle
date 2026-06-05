# Task C — Effective Inference On School Questions

Stage 1 (Qwen3-1.7B + системный промпт + max_tokens=1024) дал 59.2.
Дальше — дообучение под судью и опционально модель крупнее.

## Файлы
- solution.py — инференс (vLLM): системный промпт, max_tokens=1024, заглушка-output,
  само-проверка весов, опц. квантизация QUANT=fp8. Гарантирует 1 ответ на rid.
- train_sft.py — LoRA-SFT на 10k парах -> merge в ./weights (Colab/GPU). Лосс только на ответе.
- sanity_infer.py — быстрый прогон на нескольких вопросах (transformers), проверить качество ДО сборки.
- download_weights.py — скачать базовые веса (если НЕ дообучаешь).
- Dockerfile — образ инференса (vllm+transformers), веса запекаются через COPY.

## Почему SFT — главный рычаг
Судья — reward-модель, любит ответы в стиле эталонов (рус., LaTeX, пошагово, финальное «Ответ:»).
SFT на 10k парах подгоняет модель ровно под это распределение.
Промпт в train_sft.py и solution.py идентичны — train и inference согласованы (важно!).

## Рекомендуемая последовательность (лимит: 10/сутки, 40 всего)

### Путь A — Qwen3-1.7B + SFT (следующая сдача, низкий риск)
  pip install -U transformers==4.56.1 peft datasets accelerate pyarrow
  python train_sft.py --data dataset_ml_challenge.parquet --base Qwen/Qwen3-1.7B --out weights --epochs 3
  python sanity_infer.py --model weights      # глазами проверить ответы
train_sft.py пишет merged-веса прямо в ./weights -> download_weights.py НЕ нужен.

### Путь B — Qwen3-4B + SFT + FP8 (стретч, больше балл)
  python train_sft.py --base Qwen/Qwen3-4B --out weights --epochs 2 --bs 1 --grad_accum 16
В инференсе включить fp8: в Dockerfile добавить `ENV QUANT=fp8` или запуск с `-e QUANT=fp8`.
ОБЯЗАТЕЛЬНО замерить локально время на ~4000 вопросов до сдачи.

## Локальная проверка как у грейдера (перед сдачей!)
  docker build -t sol .
  docker run --rm --gpus all --network none \
    -v "$PWD/example-dataset.pickle":/workspace/input.pickle \
    -v "$PWD":/workspace/out -e OUTPUT_PATH=/workspace/out/output.json sol
  cat output.json        # ответы осмысленные, не пустые
В корне zip рядом с solution.py ОБЯЗАТЕЛЬНО папка weights/ с *.safetensors, иначе RE на старте.
Упаковка плоская (без обёрточной папки):
  zip -r ../submission_C.zip Dockerfile solution.py weights -x '*__pycache__*'

## Бюджеты (1xL4, 15g RAM, 8 CPU; образ<20ГБ, посылка<10ГБ, 15мин/4000)
- 1.7B bf16 (~3.4 ГБ): время и размер с запасом.
- 4B fp8 (~4 ГБ): впритык по времени — мерить; по размеру ок.
- 4B bf16 (~8 ГБ): посылка впритык, время рискованно -> предпочесть fp8.

## Ручки тюнинга
- MAX_NEW_TOKENS (1024 деф.; 768 быстрее, 1280 полнее), QUANT (пусто|fp8|awq), GPU_MEM_UTIL (0.92).
- SFT: --epochs, --lr, --lora_r (16->32 для большей ёмкости).
- Системный промпт менять синхронно в solution.py, train_sft.py, sanity_infer.py.
