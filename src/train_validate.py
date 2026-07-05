from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from src.data_utils import validate_data_dir

DEFAULT_TRAIN_COMMAND = (
    "python -m src.train_smoke "
    "--data-dir {data_dir} "
    "--output-dir {output_dir}/train_smoke "
    "--max-samples 16 "
    "--max-steps 2"
)

PROFILES = {
    "local": {
        "require_gpu": False,
        "min_gpu_memory_gb": 0.0,
        "min_disk_gb": 1.0,
    },
    "cloud-cpu": {
        "require_gpu": False,
        "min_gpu_memory_gb": 0.0,
        "min_disk_gb": 20.0,
    },
    "cloud-gpu": {
        "require_gpu": True,
        "min_gpu_memory_gb": 12.0,
        "min_disk_gb": 20.0,
    },
}

REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "pillow": "PIL",
    "torch": "torch",
    "pyyaml": "yaml",
}

OPTIONAL_PACKAGES = {
    "tqdm": "tqdm",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "qwen-vl-utils": "qwen_vl_utils",
    "peft": "peft",
    "bitsandbytes": "bitsandbytes",
}


@dataclass
class CheckResult:
    name: str
    status: str
    required: bool
    details: str


def pass_result(name: str, details: str, required: bool = True) -> CheckResult:
    return CheckResult(name=name, status="PASS", required=required, details=details)


def warn_result(name: str, details: str, required: bool = False) -> CheckResult:
    return CheckResult(name=name, status="WARN", required=required, details=details)


def fail_result(name: str, details: str, required: bool = True) -> CheckResult:
    return CheckResult(name=name, status="FAIL", required=required, details=details)


def render_train_command(command_template: str, data_dir: str | Path, output_dir: str | Path) -> list[str]:
    rendered = command_template.format(data_dir=str(data_dir), output_dir=str(output_dir))
    return shlex.split(rendered)


def has_blocking_failures(results: Sequence[CheckResult]) -> bool:
    return any(result.required and result.status == "FAIL" for result in results)


def check_python_version(min_version: tuple[int, int]) -> CheckResult:
    current = sys.version_info
    if (current.major, current.minor) >= min_version:
        return pass_result(
            "python",
            f"Python {current.major}.{current.minor}.{current.micro} on {platform.platform()}",
        )
    return fail_result(
        "python",
        f"Python {min_version[0]}.{min_version[1]}+ is required, got {current.major}.{current.minor}.{current.micro}",
    )


def check_packages(extra_required: Sequence[str], strict_optional: bool) -> list[CheckResult]:
    results: list[CheckResult] = []
    required_packages = dict(REQUIRED_PACKAGES)
    for module_name in extra_required:
        required_packages[module_name] = module_name

    for package_name, module_name in required_packages.items():
        if importlib.util.find_spec(module_name) is None:
            results.append(fail_result(f"package:{package_name}", f"Missing import module `{module_name}`"))
        else:
            results.append(pass_result(f"package:{package_name}", f"Import module `{module_name}` is available"))

    for package_name, module_name in OPTIONAL_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            result = fail_result if strict_optional else warn_result
            results.append(result(f"package:{package_name}", f"Optional import module `{module_name}` is missing", strict_optional))
        else:
            results.append(pass_result(f"package:{package_name}", f"Import module `{module_name}` is available", required=False))

    return results


def check_data(data_dir: str | Path, image_check_limit: int | None) -> CheckResult:
    summary, errors, warnings = validate_data_dir(data_dir, image_check_limit=image_check_limit)
    detail = (
        f"train={summary.train_rows}, test={summary.test_rows}, "
        f"sample={summary.sample_rows}, checked_images={summary.checked_image_paths}"
    )
    if warnings:
        detail += "; warnings=" + " | ".join(warnings)
    if errors:
        return fail_result("data", " | ".join(errors))
    return pass_result("data", detail)


def check_disk(path: str | Path, min_disk_gb: float) -> CheckResult:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(target)
    free_gb = usage.free / (1024**3)
    if free_gb >= min_disk_gb:
        return pass_result("disk", f"{free_gb:.2f} GiB free at {target}")
    return fail_result("disk", f"{free_gb:.2f} GiB free at {target}; requires at least {min_disk_gb:.2f} GiB")


def check_gpu(require_gpu: bool, min_gpu_memory_gb: float) -> CheckResult:
    try:
        import torch
    except ImportError as exc:
        if require_gpu:
            return fail_result("gpu", f"torch import failed: {exc}")
        return warn_result("gpu", f"torch import failed: {exc}")

    if not torch.cuda.is_available():
        if require_gpu:
            return fail_result("gpu", "CUDA is required for this profile, but torch.cuda.is_available() is false")
        return warn_result("gpu", "CUDA is not available; CPU-only smoke validation will be used")

    gpu_count = torch.cuda.device_count()
    memories = []
    for index in range(gpu_count):
        props = torch.cuda.get_device_properties(index)
        memories.append((props.name, props.total_memory / (1024**3)))

    max_memory = max(memory for _, memory in memories)
    details = ", ".join(f"{name}: {memory:.2f} GiB" for name, memory in memories)
    if max_memory >= min_gpu_memory_gb:
        return pass_result("gpu", f"{gpu_count} CUDA device(s): {details}")
    return fail_result("gpu", f"Max GPU memory {max_memory:.2f} GiB < required {min_gpu_memory_gb:.2f} GiB; {details}")


def check_train_command(command: list[str], timeout_seconds: int) -> CheckResult:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except FileNotFoundError as exc:
        return fail_result("train_command", f"Command not found: {exc}")
    except subprocess.TimeoutExpired:
        return fail_result("train_command", f"Command timed out after {timeout_seconds} seconds: {' '.join(command)}")

    stdout_tail = completed.stdout[-1000:].strip()
    stderr_tail = completed.stderr[-1000:].strip()
    details = f"exit_code={completed.returncode}; command={' '.join(command)}"
    if stdout_tail:
        details += f"; stdout_tail={stdout_tail}"
    if stderr_tail:
        details += f"; stderr_tail={stderr_tail}"

    if completed.returncode == 0:
        return pass_result("train_command", details)
    return fail_result("train_command", details)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate whether this workspace can run training locally or on a target server.")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="local")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs/train_validation")
    parser.add_argument("--train-command", default=DEFAULT_TRAIN_COMMAND)
    parser.add_argument("--skip-train-command", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--image-check-limit", type=int, default=None)
    parser.add_argument("--min-python", default="3.10")
    parser.add_argument("--min-disk-gb", type=float, default=None)
    parser.add_argument("--min-gpu-memory-gb", type=float, default=None)
    parser.add_argument("--require-gpu", action="store_true", help="Force GPU to be required regardless of profile")
    parser.add_argument("--allow-cpu", action="store_true", help="Force GPU to be optional regardless of profile")
    parser.add_argument("--required-package", action="append", default=[], help="Extra required import module name")
    parser.add_argument("--strict-optional-packages", action="store_true")
    parser.add_argument("--report-json", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    profile = PROFILES[args.profile]
    min_python = tuple(int(part) for part in args.min_python.split(".", maxsplit=1))
    output_dir = Path(args.output_dir)
    report_json = Path(args.report_json) if args.report_json else output_dir / "train_validation_report.json"

    require_gpu = bool(profile["require_gpu"])
    if args.require_gpu:
        require_gpu = True
    if args.allow_cpu:
        require_gpu = False

    min_disk_gb = args.min_disk_gb if args.min_disk_gb is not None else float(profile["min_disk_gb"])
    min_gpu_memory_gb = (
        args.min_gpu_memory_gb if args.min_gpu_memory_gb is not None else float(profile["min_gpu_memory_gb"])
    )

    results: list[CheckResult] = []
    results.append(check_python_version(min_python))
    results.extend(check_packages(args.required_package, args.strict_optional_packages))
    results.append(check_data(args.data_dir, args.image_check_limit))
    results.append(check_disk(output_dir, min_disk_gb))
    results.append(check_gpu(require_gpu, min_gpu_memory_gb))

    rendered_command: list[str] | None = None
    if args.skip_train_command:
        results.append(warn_result("train_command", "Skipped by --skip-train-command"))
    else:
        rendered_command = render_train_command(args.train_command, args.data_dir, output_dir)
        results.append(check_train_command(rendered_command, args.timeout_seconds))

    report = {
        "profile": args.profile,
        "data_dir": args.data_dir,
        "output_dir": str(output_dir),
        "train_command": rendered_command,
        "checks": [asdict(result) for result in results],
    }

    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for result in results:
        print(f"[{result.status}] {result.name}: {result.details}")

    print(f"Report written to {report_json}")

    if has_blocking_failures(results):
        print("Training validation failed.")
        return 1

    print("Training validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
