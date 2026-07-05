from src.train_validate import CheckResult, has_blocking_failures, render_train_command


def test_render_train_command_fills_placeholders():
    command = render_train_command(
        "python -m src.train_smoke --data-dir {data_dir} --output-dir {output_dir}/smoke",
        "data",
        "outputs/train",
    )
    assert command == [
        "python",
        "-m",
        "src.train_smoke",
        "--data-dir",
        "data",
        "--output-dir",
        "outputs/train/smoke",
    ]


def test_has_blocking_failures_ignores_optional_warning():
    results = [
        CheckResult("optional", "WARN", False, "missing optional thing"),
        CheckResult("required", "PASS", True, "ok"),
    ]
    assert not has_blocking_failures(results)


def test_has_blocking_failures_detects_required_failure():
    results = [CheckResult("required", "FAIL", True, "broken")]
    assert has_blocking_failures(results)
