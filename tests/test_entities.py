from __future__ import annotations

from pathlib import Path

import pytest

from data_needs_reporter.report.entities import EntityExtractionConfig, extract_entities
from data_needs_reporter.report.llm import MockProvider, RepairingLLMClient


class SequenceProvider(MockProvider):
    def __init__(self, responses):
        super().__init__(response=responses[0])
        self._responses = responses
        self._index = 0

    def json_complete(self, payload):  # type: ignore[override]
        response = self._responses[min(self._index, len(self._responses) - 1)]
        self._index += 1
        return response


def test_extract_entities_enforces_caps(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("polars")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    responses = [
        {
            "content": {
                "entities": [
                    {
                        "table": "DIM_CUSTOMER",
                        "column": "CUSTOMER_ID",
                        "confidence": 1.2,
                    },
                    {
                        "table": "dim_customer",
                        "column": "customer_id",
                        "confidence": 0.6,
                    },
                    {"table": "fact_orders", "column": "order_id", "confidence": 0.9},
                    {"table": "fact_orders", "column": "amount", "confidence": 0.8},
                    {"table": "unknown_table", "column": "foo", "confidence": 0.7},
                    {"table": "dim_customer", "column": "email", "confidence": 0.5},
                    {"table": "dim_customer", "column": "plan_id", "confidence": 0.4},
                    {
                        "table": "fact_orders",
                        "column": "customer_id",
                        "confidence": 0.65,
                    },
                    {"table": "fact_orders", "column": "status", "confidence": -0.1},
                    {"table": "fact_orders", "column": "amount", "confidence": 0.2},
                    {"table": "dim_customer", "column": "extra1", "confidence": 0.9},
                ]
            }
        },
        {"content": {"entities": []}},
    ]
    provider = SequenceProvider(responses)
    client = RepairingLLMClient(
        provider=provider,
        model="mock-model",
        api_key_env="OPENAI_API_KEY",
        timeout_s=5.0,
        max_output_tokens=64,
        cache_dir=tmp_path / "cache-entities",
        repair_attempts=0,
    )
    dictionary = {
        "dim_customer": ["customer_id", "email", "plan_id"],
        "fact_orders": ["order_id", "amount", "customer_id", "status"],
    }
    records = [
        {
            "source": "slack",
            "record_id": 1,
            "message_id": 1,
            "body": "Need revenue tables",
        },
        {
            "source": "email",
            "record_id": 2,
            "message_id": 2,
            "body": "Follow-up thread",
        },
    ]
    out_path = tmp_path / "entities.parquet"
    rows, coverage = extract_entities(
        records,
        client,
        dictionary,
        config=EntityExtractionConfig(),
        out_path=out_path,
    )
    assert out_path.exists()
    assert len(rows) == 1
    row = rows[0]
    assert len(row["tables"]) <= 5
    assert len(row["columns"]) <= 8
    column_pairs = {(col["table"], col["column"]) for col in row["columns"]}
    assert ("dim_customer", "customer_id") in column_pairs
    assert ("fact_orders", "order_id") in column_pairs
    assert all(0.0 <= col["confidence"] <= 1.0 for col in row["columns"])
    assert all(0.0 <= tbl["confidence"] <= 1.0 for tbl in row["tables"])
    assert "slack" in coverage
    assert coverage["slack"]["coverage_pct"] == 1.0
    assert coverage["email"]["coverage_pct"] == 0.0
    assert coverage["overall"]["messages"] == 2.0
