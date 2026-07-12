# Constrained Likelihood TTA

This is an independent experiment package. It does not import from or modify the existing
`caption_augmented` and `data_augmentation` idea packages.

## Pipeline

1. Generate and cache one caption per image.
2. Train Qwen3.5 with BF16 LoRA SFT (or optional QLoRA) on original rows plus deterministic
   shuffled views.
3. At inference, score only the 24 valid chronological permutations.
4. Split candidates into 2-4 beam groups with distinct first-image labels. This prevents
   duplicate beams from pruning valid candidates while bounding KV-cache memory.
5. Restore every TTA view to original image labels and average candidate log-likelihoods.
6. Select the highest-scoring order and convert it to submission format.

All 24 candidates include identity. `No_ordering=True` rows are retained as valid identity
supervision and are shuffled during augmentation.

## Commands

```bash
bash scripts/constrained_likelihood_tta_train.sh
bash scripts/constrained_likelihood_tta_infer.sh
```

The training script defaults to A100 x4 BF16 LoRA with effective batch size 16. Set
`LOAD_IN_4BIT=1` to switch back to QLoRA. Set `GENERATE_CAPTIONS=1` to refresh captions:

```bash
GENERATE_CAPTIONS=1 bash scripts/constrained_likelihood_tta_train.sh
```

Reduce `CANDIDATE_BATCH_SIZE` if inference runs out of memory. A cheap training preflight is:

```bash
python -m src.constrained_likelihood_tta.train \
  --data-dir data \
  --caption-missing-policy empty \
  --output-dir outputs/constrained_likelihood_tta/train_dry_run \
  --max-samples 4 \
  --max-steps 1 \
  --dry-run
```
