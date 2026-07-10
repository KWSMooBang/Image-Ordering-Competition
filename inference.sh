CUDA_VISIBLE_DEVICES=0 python -m src.caption_augmented.infer \
  --data-dir data \
  --caption-backend blip \
  --caption-device cuda \
  --order-model Qwen/Qwen3.5-4B \
  --order-adapter checkpoints/caption_augmented/filtered_shuffle_a100x4_qwen_lora \
  --output outputs/caption_augmented/filtered_shuffle_submission.csv \
  --raw-output outputs/caption_augmented/filtered_shuffle_raw_outputs.jsonl \
  --qwen-torch-dtype bfloat16 \
  --attn-implementation sdpa