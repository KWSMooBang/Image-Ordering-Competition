python -m src.data.make_split `
  --input data/raw/train.csv `
  --output-dir data/splits `
  --val-size 0.2 `
  --seed 42

python -m src.data.make_pairs `
  --input data/raw/train.csv `
  --split-dir data/splits `
  --output-dir data/interim `
  --pair-mode canonical

python -m src.training.train_pairwise `
  --config configs/pairwise_baseline.yaml