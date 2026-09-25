"""Standard retrieval metrics, computed from a ranked list of booleans (relevant/not).
No metric here is ever hardcoded — callers must pass real ranked results."""
from __future__ import annotations

import math


def recall_at_k(ranked_relevant: list[bool], k: int, total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(ranked_relevant[:k]) / total_relevant


def precision_at_k(ranked_relevant: list[bool], k: int) -> float:
    if k == 0:
        return 0.0
    return sum(ranked_relevant[:k]) / k


def reciprocal_rank(ranked_relevant: list[bool]) -> float:
    for i, is_rel in enumerate(ranked_relevant, start=1):
        if is_rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(ranked_relevant: list[bool], k: int) -> float:
    return sum(
        (1.0 if rel else 0.0) / math.log2(i + 1)
        for i, rel in enumerate(ranked_relevant[:k], start=1)
    )


def ndcg_at_k(ranked_relevant: list[bool], k: int, total_relevant: int) -> float:
    ideal = [True] * min(total_relevant, k) + [False] * max(0, k - total_relevant)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_relevant, k) / idcg
