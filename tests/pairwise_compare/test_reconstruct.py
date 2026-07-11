from src.pairwise_compare.reconstruct import (
    order_to_submission_answer,
    reconstruct_best_order,
)


def perfect_probabilities(order):
    position = {image_index: index for index, image_index in enumerate(order)}
    probabilities = {}
    for first in range(1, 5):
        for second in range(first + 1, 5):
            probabilities[(first, second)] = 0.99 if position[first] < position[second] else 0.01
    return probabilities


def test_reconstruct_exact_order():
    true_order = [4, 2, 1, 3]
    predicted_order, score = reconstruct_best_order(perfect_probabilities(true_order))

    assert predicted_order == true_order
    assert isinstance(score, float)
    assert order_to_submission_answer(predicted_order) == [3, 2, 4, 1]
