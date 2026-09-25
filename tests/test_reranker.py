from app.ranking.reranker import Candidate, RerankWeights, lexical_overlap, rerank


def test_lexical_overlap_identical_text():
    assert lexical_overlap("the cat sat", "the cat sat") == 1.0


def test_lexical_overlap_no_overlap():
    assert lexical_overlap("apples oranges", "cars trucks") == 0.0


def test_lexical_overlap_none_candidate_text():
    assert lexical_overlap("query", None) == 0.0


def test_rerank_boosts_lexical_match_over_pure_clip_score():
    # candidate A has a higher clip_score but zero lexical overlap;
    # candidate B has a lower clip_score but perfect lexical overlap.
    weights = RerankWeights(clip=0.3, lexical=0.7, modality_bonus=0.0, duplicate_penalty=0.5)
    candidates = [
        Candidate(chunk_id="A", modality="text", clip_score=0.9, text="unrelated content here"),
        Candidate(chunk_id="B", modality="text", clip_score=0.5, text="capital of france"),
    ]
    ranked = rerank(query_text="capital of france", query_modality="text", candidates=candidates, weights=weights)
    assert ranked[0].chunk_id == "B"


def test_rerank_modality_bonus_favors_matching_modality():
    weights = RerankWeights(clip=0.5, lexical=0.0, modality_bonus=0.5, duplicate_penalty=0.5)
    candidates = [
        Candidate(chunk_id="text_result", modality="text", clip_score=0.6, text=None),
        Candidate(chunk_id="image_result", modality="image", clip_score=0.6, text=None),
    ]
    ranked = rerank(query_text=None, query_modality="text", candidates=candidates, weights=weights)
    assert ranked[0].chunk_id == "text_result"


def test_rerank_penalizes_near_duplicates():
    weights = RerankWeights(clip=1.0, lexical=0.0, modality_bonus=0.0, duplicate_penalty=0.9)
    candidates = [
        Candidate(chunk_id="first", modality="text", clip_score=0.8, text="duplicate content example"),
        Candidate(chunk_id="dup", modality="text", clip_score=0.85, text="duplicate content example"),
    ]
    ranked = rerank(query_text=None, query_modality="text", candidates=candidates, weights=weights)
    # "dup" has higher clip_score but is flagged a duplicate (seen after "first" in insertion order... )
    assert any(c.extra["score_breakdown"]["duplicate_penalty_applied"] for c in ranked)


def test_rerank_respects_top_k():
    candidates = [Candidate(chunk_id=str(i), modality="text", clip_score=i / 10) for i in range(10)]
    ranked = rerank(query_text=None, query_modality="text", candidates=candidates, top_k=3)
    assert len(ranked) == 3
