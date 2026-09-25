from typing import Any, cast

from financial_advisor.contracts import (
    RetrievalChannel,
    RetrievalPath,
    RetrievedEvidenceCandidate,
)
from financial_advisor.retrieval.pipeline import RetrievalPipeline


def candidate(
    canonical_candidate_id: str,
    retrieval_channel: RetrievalChannel,
    rank: int,
) -> RetrievedEvidenceCandidate:
    return RetrievedEvidenceCandidate.model_validate(
        {
            "chunk_id": canonical_candidate_id,
            "canonical_candidate_id": canonical_candidate_id,
            "source_id": "test",
            "source_title": f"{canonical_candidate_id} title",
            "publisher": "Publisher",
            "source_url": f"https://example.com/{canonical_candidate_id}",
            "topics": ["testing"],
            "heading_path": ["Test"],
            "section_position": 1,
            "chunk_position": 1,
            "token_count": 2,
            "text": f"{canonical_candidate_id} evidence",
            "retrieval_channel": retrieval_channel,
            "rank": rank,
            "vector": [1.0, float(rank)],
        }
    )


class Knowledge:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> list[RetrievedEvidenceCandidate]:
        self.queries.append(query)
        return [candidate("local", RetrievalChannel.VECTOR, 1)]


class Web:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[list[RetrievedEvidenceCandidate], list[str]]:
        self.queries.append(query)
        return [candidate("web", RetrievalChannel.WEB, 1)], ["web limitation"]


def test_pipeline_executes_only_advisor_selected_paths() -> None:
    knowledge = Knowledge()
    web = Web()
    pipeline = RetrievalPipeline(
        local_hybrid_retriever=cast(Any, knowledge),
        web_candidate_retriever=cast(Any, web),
        rerank=lambda _query, documents: [float(index) for index, _ in enumerate(documents)],
    )

    result = pipeline.retrieve(
        "question",
        [RetrievalPath.WEB],
    )

    assert knowledge.queries == []
    assert web.queries == ["question"]
    assert {item.source for item in result.evidence} == {RetrievalChannel.WEB}
    assert result.limitations == ["web limitation"]


def test_pipeline_jointly_selects_local_and_web_with_stable_evidence_ids() -> None:
    pipeline = RetrievalPipeline(
        local_hybrid_retriever=cast(Any, Knowledge()),
        web_candidate_retriever=cast(Any, Web()),
        rerank=lambda _query, documents: [float("web" in text) for text in documents],
    )
    paths = [RetrievalPath.LOCAL_HYBRID, RetrievalPath.WEB]

    first = pipeline.retrieve("question", paths)
    second = pipeline.retrieve("question", paths)

    assert {item.source for item in first.evidence} == {
        RetrievalChannel.VECTOR,
        RetrievalChannel.WEB,
    }
    assert [item.evidence_id for item in first.evidence] == [
        item.evidence_id for item in second.evidence
    ]
