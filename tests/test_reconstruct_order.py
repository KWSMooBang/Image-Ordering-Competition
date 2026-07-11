from src.inference.reconstruct_order import (
    order_to_answer,
    reconstruct_best_order,
)


def perfect_probabilities(order):
    position = {image_index: index for index, image_index in enumerate(order)}
    probabilities = {}

    for first in range(1, 5):
        for second in range(first + 1, 5):
            probabilities[(first, second)] = (
                0.99 if position[first] < position[second] else 0.01
            )
    return probabilities


def test_reconstruct_exact_order():
    true_order = [4, 2, 1, 3]
    predicted_order, score = reconstruct_best_order(
        perfect_probabilities(true_order)
    )

    assert predicted_order == true_order
    assert isinstance(score, float)
    assert order_to_answer(predicted_order) == [3, 2, 4, 1]


def test_reverse_direction_pair_keys_are_supported():
    true_order = [2, 1, 4, 3]
    canonical = perfect_probabilities(true_order)
    reversed_pairs = {
        (second, first): 1.0 - probability
        for (first, second), probability in canonical.items()
    }

    predicted_order, _ = reconstruct_best_order(reversed_pairs)
    assert predicted_order == true_order
