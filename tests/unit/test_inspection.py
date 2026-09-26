from pathlib import Path

import lancedb

from financial_advisor.inspection import KnowledgeStoreInspector


def test_knowledge_inspection_filters_metadata_without_exposing_vectors(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "knowledge"
    database = lancedb.connect(str(database_path))
    database.create_table(
        "knowledge_chunks",
        data=[
            {
                "chunk_id": "finra:1:1",
                "canonical_candidate_id": "candidate-1",
                "source_id": "finra_basics",
                "source_title": "Investment basics",
                "publisher": "FINRA",
                "source_url": "https://www.finra.org/",
                "topics": ["investing", "risk"],
                "heading_path": ["Investing", "Diversification"],
                "section_position": 1,
                "chunk_position": 1,
                "token_count": 42,
                "text": "Diversification can help manage concentration risk.",
                "vector": [0.1, 0.2],
            },
            {
                "chunk_id": "investor:1:1",
                "canonical_candidate_id": "candidate-2",
                "source_id": "investor_savings",
                "source_title": "Savings guidance",
                "publisher": "Investor.gov",
                "source_url": "https://www.investor.gov/",
                "topics": ["savings"],
                "heading_path": ["Savings"],
                "section_position": 1,
                "chunk_position": 1,
                "token_count": 30,
                "text": "Match savings choices to the time horizon.",
                "vector": [0.3, 0.4],
            },
        ],
    )
    inspector = KnowledgeStoreInspector(database_path)

    summary = inspector.summary()
    filtered = inspector.chunks(
        publisher="finra",
        topic="risk",
        metadata_field="heading_path",
        metadata_value="diversification",
        min_token_count=40,
    )

    assert summary.chunk_count == 2
    assert summary.publishers == {"FINRA": 1, "Investor.gov": 1}
    assert filtered.total_matching_chunks == 1
    assert filtered.chunks[0].chunk_id == "finra:1:1"
    assert "vector" not in filtered.chunks[0].model_dump()
