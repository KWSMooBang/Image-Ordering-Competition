from src.submission import (
    chronological_to_submission,
    format_answer,
    parse_answer_cell,
    parse_model_output,
    submission_to_chronological,
)


def test_chronological_to_submission_inverts_order():
    assert chronological_to_submission([4, 2, 1, 3]) == [3, 2, 4, 1]


def test_submission_to_chronological_inverts_answer():
    assert submission_to_chronological([3, 2, 4, 1]) == [4, 2, 1, 3]


def test_parse_model_output_extracts_and_inverts():
    assert parse_model_output("The answer is [4, 2, 1, 3].") == [3, 2, 4, 1]


def test_parse_model_output_uses_first_valid_repeated_list():
    output = "[1, 3, 4, 2]\n\n[1, 3, 4, 2]\n\n[1, 3, 4, 2]"

    assert parse_model_output(output) == [1, 4, 2, 3]


def test_parse_model_output_skips_invalid_lists_before_valid_permutation():
    output = "Not this [1, 2], but this [4, 2, 1, 3]."

    assert parse_model_output(output) == [3, 2, 4, 1]


def test_format_and_parse_answer_cell():
    assert format_answer([1, 2, 3, 4]) == "[1, 2, 3, 4]"
    assert parse_answer_cell("[1, 2, 3, 4]") == [1, 2, 3, 4]
