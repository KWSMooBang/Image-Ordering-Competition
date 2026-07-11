"""Metrics for pairwise and reconstructed full-order predictions."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Sequence

import math


def pairwise_accuracy(
    y_true: Sequence[int | float],
    y_probability: Sequence[float],
    *,
    threshold: float = 0.5,
) -> float:
    if len(y_true) != len(y_probability):
        raise ValueError("y_true and y_probability must have equal lengths.")
    if not y_true:
        return 0.0

    correct = sum(
        int((probability >= threshold) == bool(label))
        for label, probability in zip(y_true, y_probability)
    )
    return correct / len(y_true)


def binary_log_loss(
    y_true: Sequence[int | float],
    y_probability: Sequence[float],
    *,
    epsilon: float = 1e-7,
) -> float:
    if len(y_true) != len(y_probability):
        raise ValueError("y_true and y_probability must have equal lengths.")
    if not y_true:
        return 0.0

    total = 0.0
    for label, probability in zip(y_true, y_probability):
        probability = min(max(float(probability), epsilon), 1.0 - epsilon)
        label = float(label)
        total += -(label * math.log(probability) + (1.0 - label) * math.log(1.0 - probability))
    return total / len(y_true)


def exact_match_accuracy(
    true_orders: Sequence[Sequence[int]],
    predicted_orders: Sequence[Sequence[int]],
) -> float:
    if len(true_orders) != len(predicted_orders):
        raise ValueError("true_orders and predicted_orders must have equal lengths.")
    if not true_orders:
        return 0.0

    correct = sum(
        list(true_order) == list(predicted_order)
        for true_order, predicted_order in zip(true_orders, predicted_orders)
    )
    return correct / len(true_orders)


def kendall_distance(
    true_order: Sequence[int],
    predicted_order: Sequence[int],
    *,
    normalised: bool = False,
) -> float:
    if sorted(true_order) != sorted(predicted_order):
        raise ValueError("Both orders must contain the same items exactly once.")

    true_position = {item: index for index, item in enumerate(true_order)}
    predicted_position = {item: index for index, item in enumerate(predicted_order)}

    disagreements = 0
    items = list(true_order)
    for first, second in combinations(items, 2):
        true_relation = true_position[first] < true_position[second]
        predicted_relation = predicted_position[first] < predicted_position[second]
        disagreements += int(true_relation != predicted_relation)

    if not normalised:
        return float(disagreements)

    maximum = len(items) * (len(items) - 1) / 2
    return disagreements / maximum if maximum else 0.0


def has_cycle(
    pair_probabilities: Mapping[tuple[int, int], float],
    *,
    threshold: float = 0.5,
    num_items: int = 4,
) -> bool:
    graph = {item: set() for item in range(1, num_items + 1)}

    for first, second in combinations(range(1, num_items + 1), 2):
        if (first, second) in pair_probabilities:
            probability = pair_probabilities[(first, second)]
        elif (second, first) in pair_probabilities:
            probability = 1.0 - pair_probabilities[(second, first)]
        else:
            raise KeyError(f"Missing pair probability for ({first}, {second}).")

        if probability >= threshold:
            graph[first].add(second)
        else:
            graph[second].add(first)

    visiting: set[int] = set()
    visited: set[int] = set()

    def visit(node: int) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False

        visiting.add(node)
        for neighbour in graph[node]:
            if visit(neighbour):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    return any(visit(node) for node in graph)


def cycle_rate(
    probability_groups: Iterable[Mapping[tuple[int, int], float]],
    *,
    threshold: float = 0.5,
    num_items: int = 4,
) -> float:
    groups = list(probability_groups)
    if not groups:
        return 0.0

    return sum(
        has_cycle(group, threshold=threshold, num_items=num_items)
        for group in groups
    ) / len(groups)
