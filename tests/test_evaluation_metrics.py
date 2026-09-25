import pytest

from evaluation.metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k


def test_recall_at_k_perfect():
    assert recall_at_k([True, False, False], k=3, total_relevant=1) == 1.0


def test_recall_at_k_missed_beyond_k():
    assert recall_at_k([False, False, True], k=2, total_relevant=1) == 0.0


def test_precision_at_k():
    assert precision_at_k([True, True, False, False], k=4) == 0.5


def test_reciprocal_rank_first_hit():
    assert reciprocal_rank([True, False, False]) == 1.0


def test_reciprocal_rank_third_hit():
    assert reciprocal_rank([False, False, True]) == pytest.approx(1 / 3)


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank([False, False, False]) == 0.0


def test_ndcg_at_k_ideal_ordering_scores_one():
    assert ndcg_at_k([True, False, False], k=3, total_relevant=1) == 1.0


def test_ndcg_at_k_worse_ordering_scores_less_than_one():
    score_ideal = ndcg_at_k([True, False], k=2, total_relevant=1)
    score_worse = ndcg_at_k([False, True], k=2, total_relevant=1)
    assert score_worse < score_ideal
