<#
.SYNOPSIS
  SNU AI Challenge - caption_augmented smoke test.

.DESCRIPTION
  Runs, in order: GPU check -> pytest -> caption cache smoke -> whole-mode
  inference smoke -> pairwise-mode inference smoke -> adaptive-mode inference
  smoke. Stops immediately on the first failure so you can see exactly which
  step broke.

.PARAMETER DataDir
  Path to the competition data folder. Default: data

.PARAMETER OrderModel
  Qwen ordering model to use. Default: Qwen/Qwen3.5-4B

.PARAMETER Adapter
  Path to a trained LoRA adapter directory (e.g. the output of
  scripts/caption_augmented_train.sh). Omit to smoke-test the base model
  zero-shot (useful before training, to check the environment/code works at
  all). Pass it after training to actually verify the fine-tuned checkpoint -
  without it, this script silently ignores your trained weights.

.PARAMETER MaxSamples
  Number of test rows to use for the smoke test. Default: 4

.PARAMETER SkipPytest
  Skip the `python -m pytest` step (useful if torch/GPU deps aren't fully
  installed yet and you only want the inference smoke tests).

.EXAMPLE
  .\smoke_test.ps1
.EXAMPLE
  .\smoke_test.ps1 -OrderModel "Qwen/Qwen3.5-9B" -MaxSamples 8
.EXAMPLE
  # After training, verify the actual checkpoint instead of the base model:
  .\smoke_test.ps1 -Adapter "checkpoints/caption_augmented/qwen3_5_4b_qlora" -SkipPytest
#>

param(
    [string]$DataDir = "data",
    [string]$OrderModel = "Qwen/Qwen3.5-4B",
    [string]$Adapter = "",
    [int]$MaxSamples = 4,
    [string]$OutputDir = "outputs/caption_augmented",
    [switch]$SkipPytest
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Action
    )
    Write-Host ""
    Write-Host "==> $Name" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Name (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

if ($Adapter -and -not (Test-Path $Adapter)) {
    Write-Host "Adapter path not found: $Adapter" -ForegroundColor Red
    exit 1
}

$adapterArgs = @()
if ($Adapter) {
    $adapterArgs = @("--order-adapter", $Adapter)
    Write-Host "Using trained adapter: $Adapter" -ForegroundColor Yellow
} else {
    Write-Host "No -Adapter given - testing base model zero-shot (pass -Adapter to test your trained checkpoint)." -ForegroundColor Yellow
}

Write-Host "==> GPU check (nvidia-smi)" -ForegroundColor Cyan
try {
    nvidia-smi
} catch {
    Write-Host "nvidia-smi not found or failed - continuing anyway, but check your GPU/driver setup." -ForegroundColor Yellow
}

if (-not $SkipPytest) {
    Invoke-Step "Unit tests (pytest)" {
        python -m pytest -q
    }
}

Invoke-Step "Caption cache smoke ($MaxSamples samples, BLIP)" {
    python -m src.caption_augmented.captions `
        --data-dir $DataDir --split test `
        --output "$OutputDir/test_captions_smoke.jsonl" `
        --caption-backend blip --caption-device cuda `
        --max-samples $MaxSamples
}

Invoke-Step "Whole-mode inference smoke" {
    python -m src.caption_augmented.infer `
        --data-dir $DataDir --max-samples $MaxSamples `
        --output "$OutputDir/smoke_whole.csv" `
        --raw-output "$OutputDir/smoke_whole_raw.jsonl" `
        --order-model $OrderModel `
        --tta-permutations 1 `
        @adapterArgs
}

Invoke-Step "Pairwise-mode inference smoke" {
    python -m src.caption_augmented.infer `
        --data-dir $DataDir --max-samples $MaxSamples `
        --output "$OutputDir/smoke_pairwise.csv" `
        --raw-output "$OutputDir/smoke_pairwise_raw.jsonl" `
        --order-model $OrderModel `
        --comparison-mode pairwise `
        @adapterArgs
}

Invoke-Step "Adaptive-mode inference smoke" {
    python -m src.caption_augmented.infer `
        --data-dir $DataDir --max-samples $MaxSamples `
        --output "$OutputDir/smoke_adaptive.csv" `
        --raw-output "$OutputDir/smoke_adaptive_raw.jsonl" `
        --order-model $OrderModel `
        --comparison-mode adaptive `
        --tta-permutations 3 `
        @adapterArgs
}

Write-Host ""
Write-Host "All smoke tests passed." -ForegroundColor Green
Write-Host "  $OutputDir/smoke_whole.csv"
Write-Host "  $OutputDir/smoke_pairwise.csv"
Write-Host "  $OutputDir/smoke_adaptive.csv"