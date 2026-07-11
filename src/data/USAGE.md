# Pairwise data module

프로젝트 루트에 `src/data/`를 그대로 복사하세요.

## 1. 원본 샘플 기준 분할

```powershell
python -m src.data.make_split `
  --input data/train.csv `
  --output-dir data/splits `
  --val-size 0.2 `
  --seed 42
```

생성 파일:

```text
data/splits/train_ids.csv
data/splits/val_ids.csv
data/splits/split_manifest.json
```

## 2. Pairwise CSV 생성

기존 이미지 경로가 `data/train/<Id>/<filename>`인 구조를 기준으로 합니다.

```powershell
python -m src.data.make_pairs `
  --input data/train.csv `
  --split-dir data/splits `
  --output-dir data/interim `
  --pair-mode canonical `
  --image-root data/train `
  --check-images
```

생성 파일:

```text
data/interim/train_pairs.csv
data/interim/val_pairs.csv
```

`canonical`은 원본 샘플당 6개 pair를 만듭니다. 학습 Dataset에서
`swap_probability=0.5`를 주면 A/B 방향을 무작위로 뒤집으므로 12개를
디스크에 중복 저장할 필요가 없습니다.

## 3. Dataset 사용

### SigLIP 등 Hugging Face 비전 인코더

```python
from torch.utils.data import DataLoader
from src.data import PairwiseDataset, build_hf_image_transform

image_transform = build_hf_image_transform(
    "google/siglip-base-patch16-224"
)

train_dataset = PairwiseDataset(
    pairs="data/interim/train_pairs.csv",
    image_root="data/train",
    transform=image_transform,
    swap_probability=0.5,
)

val_dataset = PairwiseDataset(
    pairs="data/interim/val_pairs.csv",
    image_root="data/train",
    transform=image_transform,
    swap_probability=0.0,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
```

각 배치는 다음 필드를 가집니다.

```text
sample_id
pair_id
sentence
image_a
image_b
image_a_index
image_b_index
label
no_ordering
swapped
```

`label == 1`은 `image_a`가 `image_b`보다 시간상 먼저라는 뜻입니다.

## 4. No_ordering 실험

기본값은 모든 데이터를 보존합니다.

제외 실험:

```powershell
python -m src.data.make_pairs `
  --input data/train.csv `
  --split-dir data/splits `
  --output-dir data/interim_no_ordering_removed `
  --exclude-no-ordering
```