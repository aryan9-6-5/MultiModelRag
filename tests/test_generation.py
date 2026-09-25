from app.generation.generator import generate_answer
from app.generation.llm_client import GenerationError
from app.generation.prompt import build_prompt, format_evidence
from app.retrieval.retriever import RetrievalResult


def _evidence():
    return [
        RetrievalResult(
            chunk_id="c1", modality="text", source="doc.pdf", file_path="/x/doc.pdf",
            page_number=3, text="Paris is the capital of France.", image_path=None,
            clip_score=0.9, final_score=0.9, score_breakdown={},
        ),
        RetrievalResult(
            chunk_id="c2", modality="image", source="eiffel.jpg", file_path="/x/eiffel.jpg",
            page_number=None, text="a photo of the eiffel tower", image_path="/x/eiffel.jpg",
            clip_score=0.8, final_score=0.8, score_breakdown={},
        ),
    ]


def test_format_evidence_includes_citations_and_pages():
    text = format_evidence(_evidence())
    assert "[E1]" in text and "[E2]" in text
    assert "page 3" in text


def test_format_evidence_handles_no_results():
    assert "no evidence" in format_evidence([]).lower()


def test_build_prompt_includes_grounding_instructions():
    prompt = build_prompt("Where is the Eiffel Tower?", _evidence())
    assert "do not invent" in prompt.lower() or "not invent" in prompt.lower()
    assert "Where is the Eiffel Tower?" in prompt


class _FailingClient:
    model = "unreachable-model"

    def generate(self, prompt, temperature=0.2):
        raise GenerationError("Cannot reach Ollama")


def test_generate_answer_handles_unreachable_llm_gracefully(tmp_db):
    response = generate_answer("test query", _evidence(), db=tmp_db, client=_FailingClient())
    assert response.success is False
    assert response.error is not None
    assert response.answer == ""

    failures = tmp_db.conn.execute("SELECT COUNT(*) FROM generation_log WHERE success = FALSE").fetchone()[0]
    assert failures == 1


class _StubClient:
    model = "stub-model"

    def generate(self, prompt, temperature=0.2):
        return {"text": "Paris [E1].", "model": self.model, "prompt_tokens": 42, "completion_tokens": 5}


def test_generate_answer_success_path_logs_tokens(tmp_db):
    response = generate_answer("Where is Paris?", _evidence(), db=tmp_db, client=_StubClient())
    assert response.success is True
    assert response.answer == "Paris [E1]."

    row = tmp_db.conn.execute(
        "SELECT prompt_tokens, completion_tokens, success FROM generation_log"
    ).fetchone()
    assert row == (42, 5, True)
