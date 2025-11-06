from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_needs_reporter.config import DEFAULT_CONFIG_PATH, load_config
from data_needs_reporter.generate.comms import generate_comms
from data_needs_reporter.report.classify import (
    classify_threads,
    pack_thread,
    save_predictions,
)
from data_needs_reporter.report.entities import EntityExtractionConfig
from data_needs_reporter.report.llm import LLMError, MockProvider, RepairingLLMClient
from data_needs_reporter.report.prefilter import prefilter_messages
from data_needs_reporter.report.run import run_entity_extraction_for_archetype
from data_needs_reporter.utils.cost_guard import CostGuard


def test_llm_cache_hits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = MockProvider(response={"content": {"theme": "latency"}})
    client = RepairingLLMClient(
        provider=provider,
        model="mock-model",
        api_key_env="OPENAI_API_KEY",
        timeout_s=5.0,
        max_output_tokens=64,
        cache_dir=tmp_path,
        repair_attempts=1,
    )

    first = client.json_complete("Prompt", temperature=0.0)
    assert first == {"theme": "latency"}
    assert provider._called == 1
    cache_files = list(tmp_path.glob("*.json"))
    assert cache_files

    second = client.json_complete("Prompt", temperature=0.0)
    assert second == first
    assert provider._called == 1  # served from cache


def test_llm_repair_attempt_then_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class BadProvider(MockProvider):
        def __init__(self) -> None:
            super().__init__(response={"content": "not-json"})

    provider = BadProvider()
    client = RepairingLLMClient(
        provider=provider,
        model="mock-model",
        api_key_env="OPENAI_API_KEY",
        timeout_s=5.0,
        max_output_tokens=64,
        cache_dir=tmp_path,
        repair_attempts=1,
    )

    with pytest.raises(LLMError):
        client.json_complete("Prompt", temperature=0.0)

    assert provider._called == 2  # initial + repair attempt
    assert not any(tmp_path.glob("*.json"))


def test_generate_comms_respects_cap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pytest.importorskip("polars")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})
    cfg.cache.dir = str(tmp_path / "cache")
    cfg.comms.slack_threads = 5
    cfg.comms.email_threads = 3
    cfg.comms.nlq = 3
    provider = MockProvider(
        response={
            "content": {
                "body": "Issue detected",
                "bucket": "data_quality",
                "subject": "Alert",
                "text": "Where are the anomalies?",
                "parsed_intent": "data_gap",
                "tokens": 2000,
            }
        }
    )
    client = RepairingLLMClient(
        provider=provider,
        model=cfg.classification.model,
        api_key_env=cfg.classification.env_key_var,
        timeout_s=5.0,
        max_output_tokens=cfg.classification.max_output_tokens,
        cache_dir=tmp_path / "cache",
        repair_attempts=1,
    )
    guard = CostGuard(cap_usd=0.5, price_per_1k_tokens=10.0)
    summary = generate_comms(
        cfg, "neobank", tmp_path / "comms", client, guard, seed=111
    )
    assert summary["slack_messages"] == 0
    assert summary["email_messages"] == 0
    assert summary["nlq"] == 0
    budget = json.loads((tmp_path / "comms" / "budget.json").read_text())
    assert budget["stopped_due_to_cap"] is True
    slack_cov = budget["coverage"]["slack"]["overall"]
    assert slack_cov["met_floor"] is False
    assert slack_cov["behavior"] == "behavior_c_continue"
    assert budget["estimate"]["total_tokens"] <= budget["token_budget"]


def test_generate_comms_hits_requested_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pl = pytest.importorskip("polars")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    cfg = load_config(DEFAULT_CONFIG_PATH, None, env={}, cli_overrides={})
    cfg.cache.dir = str(tmp_path / "cache2")
    cfg.comms.slack_threads = 20
    cfg.comms.email_threads = 12
    cfg.comms.nlq = 10
    provider = MockProvider(
        response={
            "content": {
                "body": "Resolved ingestion issue",
                "bucket": "analytics",
                "subject": "Daily sync",
                "text": "Show ingestion backlog",
                "parsed_intent": "data_health",
                "tokens": 60,
            }
        }
    )
    client = RepairingLLMClient(
        provider=provider,
        model=cfg.classification.model,
        api_key_env=cfg.classification.env_key_var,
        timeout_s=5.0,
        max_output_tokens=cfg.classification.max_output_tokens,
        cache_dir=tmp_path / "cache2",
        repair_attempts=1,
    )
    guard = CostGuard(cap_usd=5.0, price_per_1k_tokens=0.5)
    summary = generate_comms(
        cfg, "marketplace", tmp_path / "comms2", client, guard, seed=222
    )
    slack = pl.read_parquet(tmp_path / "comms2" / "slack_messages.parquet")
    email = pl.read_parquet(tmp_path / "comms2" / "email_messages.parquet")
    nlq = pl.read_parquet(tmp_path / "comms2" / "nlq.parquet")
    assert abs(slack.height - cfg.comms.slack_threads) <= max(
        1, int(0.05 * cfg.comms.slack_threads)
    )
    assert abs(email.height - cfg.comms.email_threads) <= max(
        1, int(0.05 * cfg.comms.email_threads)
    )
    assert abs(nlq.height - cfg.comms.nlq) <= max(1, int(0.05 * cfg.comms.nlq))
    budget = json.loads((tmp_path / "comms2" / "budget.json").read_text())
    assert budget["stopped_due_to_cap"] is False
    assert summary["slack_messages"] == slack.height
    assert summary["email_messages"] == email.height
    assert summary["nlq"] == nlq.height
    assert budget["estimate"]["total_tokens"] <= budget["token_budget"]
    for source in ("slack", "email", "nlq"):
        overall = budget["coverage"][source]["overall"]
        assert overall["met_floor"] is True
        per_bucket = budget["coverage"][source]["per_bucket"]
        assert isinstance(per_bucket, dict)
    assert "quotas" in budget
    quotas = budget["quotas"]
    for source in ("slack", "email", "nlq"):
        assert "bucket_totals" in quotas[source]
        assert isinstance(quotas[source]["day_bucket"], dict)


def test_prefilter_high_signal_passes_threshold():
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    users = {1: "analytics", 2: "product"}
    records = [
        {
            "message_id": 1,
            "thread_id": 1,
            "user_id": 1,
            "bucket": "data_quality",
            "body": "Missing revenue data due to pipeline backlog alert this morning.",
            "channel": "#data-quality",
            "sent_at": now,
        }
    ]
    results = prefilter_messages(records, users, threshold=0.35)
    assert results[0].selected is True
    assert results[0].score >= 0.35
    assert any(reason.startswith("bucket") for reason in results[0].reasons)
    assert any(reason.startswith("keyword") for reason in results[0].reasons)


def test_prefilter_demotes_low_signal():
    now = datetime(2024, 1, 2, tzinfo=timezone.utc)
    users = {3: "marketing"}
    records = [
        {
            "message_id": 2,
            "thread_id": 2,
            "user_id": 3,
            "bucket": "announcements",
            "body": "Quick reminder about team lunch tomorrow.",
            "channel": "#random",
            "sent_at": now,
        }
    ]
    results = prefilter_messages(
        records, users, threshold=0.35, fallback_per_day_channel=0
    )
    assert results[0].selected is False
    assert results[0].score < 0.2


def test_pack_thread_respects_caps():
    now = datetime(2024, 1, 3, tzinfo=timezone.utc)
    messages = []
    for idx in range(30):
        messages.append(
            {
                "message_id": idx + 1,
                "thread_id": 1,
                "user_id": 100 + idx,
                "body": "Message body " + ("x" * 80),
                "tokens": 60,
                "sent_at": now.replace(hour=idx % 24),
                "prefilter_score": 1.0 - (idx * 0.02),
            }
        )
    packed = pack_thread(
        messages, exec_user_ids={101, 102}, max_messages=20, max_tokens=200
    )
    assert len(packed["messages"]) <= 20
    assert packed["token_total"] <= 200
    selected_ids = {msg["message_id"] for msg in packed["messages"]}
    assert messages[0]["message_id"] in selected_ids
    assert {messages[1]["message_id"], messages[2]["message_id"]} <= selected_ids


def test_pack_thread_prioritizes_score_and_recency():
    base = datetime(2024, 2, 1, tzinfo=timezone.utc)
    messages = [
        {
            "message_id": 1,
            "thread_id": 7,
            "user_id": 10,
            "body": "Root message",
            "tokens": 40,
            "sent_at": base,
            "prefilter_score": 0.1,
        },
        {
            "message_id": 2,
            "thread_id": 7,
            "user_id": 200,
            "body": "Executive context",
            "tokens": 50,
            "sent_at": base + timedelta(minutes=1),
            "prefilter_score": 0.4,
        },
        {
            "message_id": 3,
            "thread_id": 7,
            "user_id": 11,
            "body": "Older high score",
            "tokens": 30,
            "sent_at": base + timedelta(minutes=2),
            "prefilter_score": 0.9,
        },
        {
            "message_id": 4,
            "thread_id": 7,
            "user_id": 12,
            "body": "Newer high score",
            "tokens": 30,
            "sent_at": base + timedelta(minutes=3),
            "prefilter_score": 0.9,
        },
        {
            "message_id": 5,
            "thread_id": 7,
            "user_id": 13,
            "body": "Lower score",
            "tokens": 30,
            "sent_at": base + timedelta(minutes=4),
            "prefilter_score": 0.7,
        },
    ]

    packed = pack_thread(
        messages,
        exec_user_ids={200},
        max_messages=3,
        max_tokens=300,
    )
    selected_ids = {msg["message_id"] for msg in packed["messages"]}
    assert {1, 2}.issubset(selected_ids)
    assert 4 in selected_ids
    assert 3 not in selected_ids


def test_entity_extraction_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pl = pytest.importorskip("polars")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    slack_df = pl.DataFrame(
        {
            "message_id": [1, 2],
            "body": [
                "Investigate fact_card_transaction spikes in amount_cents",
                "Unrelated chatter",
            ],
        }
    )
    email_df = pl.DataFrame(
        {
            "message_id": [10],
            "subject": ["FYI"],
            "body": ["Columns missing"],
        }
    )
    nlq_df = pl.DataFrame(
        {
            "query_id": [101],
            "text": ["Show fact_card_transaction amount_cents by merchant_id"],
        }
    )

    predictions_df = pl.DataFrame(
        {
            "thread_id": [1, 2],
            "source": ["slack", "email"],
            "relevance": [0.85, 0.2],
            "message_ids": [[1], [10]],
        }
    )

    class EntityProvider(MockProvider):
        def __init__(self) -> None:
            super().__init__(response={})

        def json_complete(self, payload):  # type: ignore[override]
            self._called += 1
            return {
                "content": {
                    "tables": [
                        {"name": "fact_card_transaction", "confidence": 0.82},
                        {"name": "unknown_table", "confidence": 0.9},
                    ],
                    "columns": [
                        {
                            "table": "fact_card_transaction",
                            "column": "amount_cents",
                            "confidence": 0.76,
                        },
                        "fact_card_transaction.merchant_id",
                        {
                            "table": "other_table",
                            "column": "foo",
                            "confidence": 0.2,
                        },
                    ],
                    "confidence": 0.9,
                    "tokens": 200,
                }
            }

    provider = EntityProvider()
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
        "fact_card_transaction": ["amount_cents", "merchant_id"],
        "dim_customer": ["customer_id"],
    }

    out_path = tmp_path / "entities.parquet"
    config = EntityExtractionConfig(cap_usd=0.05)
    results, stats = run_entity_extraction_for_archetype(
        "neobank",
        client,
        slack_messages=slack_df,
        email_messages=email_df,
        nlq_messages=nlq_df,
        predictions=predictions_df,
        out_path=out_path,
        dictionary=dictionary,
        config=config,
        relevance_threshold=0.5,
    )

    assert out_path.exists()
    df = pl.read_parquet(out_path)
    assert df.height == 2  # slack + nlq
    assert "email" not in df.select("source").to_series().to_list()

    slack_row = df.filter(pl.col("source") == "slack").to_dicts()[0]
    tables = slack_row["tables"]
    assert tables and tables[0]["name"] == "fact_card_transaction"
    assert 0.0 <= tables[0]["confidence"] <= 1.0
    columns = slack_row["columns"]
    column_names = {(col["table"], col["column"]) for col in columns}
    assert ("fact_card_transaction", "amount_cents") in column_names
    assert ("fact_card_transaction", "merchant_id") in column_names

    nlq_row = df.filter(pl.col("source") == "nlq").to_dicts()[0]
    assert nlq_row["columns"]

    assert results
    assert stats["cost_usd"] <= config.cap_usd


def test_classify_threads_and_save_predictions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pl = pytest.importorskip("polars")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    provider = MockProvider(
        response={"content": {"theme": "data_quality", "relevance": 0.87}}
    )
    client = RepairingLLMClient(
        provider=provider,
        model="mock-model",
        api_key_env="OPENAI_API_KEY",
        timeout_s=5.0,
        max_output_tokens=64,
        cache_dir=tmp_path / "cache-cls",
        repair_attempts=1,
    )
    now = datetime(2024, 1, 4, tzinfo=timezone.utc)
    thread = {
        "thread_id": 42,
        "messages": [
            {
                "message_id": 1,
                "thread_id": 42,
                "user_id": 10,
                "body": "Root message about data incidents.",
                "tokens": 120,
                "sent_at": now,
                "prefilter_score": 0.9,
            },
            {
                "message_id": 2,
                "thread_id": 42,
                "user_id": 20,
                "body": "Executive follow-up with more context.",
                "tokens": 80,
                "sent_at": now.replace(hour=2),
                "prefilter_score": 0.8,
            },
        ],
    }
    predictions = classify_threads([thread], client, source="slack", exec_user_ids={10})
    assert len(predictions) == 1
    assert predictions[0]["theme"] == "data_quality"
    assert predictions[0]["message_count"] <= 20
    assert predictions[0]["parse_error"] is False
    assert predictions[0]["include_in_demand"] is True
    out_path = tmp_path / "preds.parquet"
    save_predictions(predictions, out_path)
    df = pl.read_parquet(out_path)
    assert set(df.columns) >= {
        "thread_id",
        "source",
        "theme",
        "relevance",
        "parse_error",
        "include_in_demand",
    }
    assert df["parse_error"].sum() == 0


def test_classify_threads_parse_error_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class AlwaysBadProvider(MockProvider):
        def __init__(self) -> None:
            super().__init__(response={})
            self.calls = 0

        def json_complete(self, payload):  # type: ignore[override]
            self.calls += 1
            raise ValueError("bad response")

    provider = AlwaysBadProvider()
    client = RepairingLLMClient(
        provider=provider,
        model="mock-model",
        api_key_env="OPENAI_API_KEY",
        timeout_s=5.0,
        max_output_tokens=64,
        cache_dir=tmp_path / "cache-bad",
        repair_attempts=0,
    )

    now = datetime(2024, 1, 5, tzinfo=timezone.utc)
    threads = [
        {
            "thread_id": idx,
            "messages": [
                {
                    "message_id": idx * 10,
                    "thread_id": idx,
                    "user_id": 1,
                    "body": "Body text",
                    "tokens": 50,
                    "sent_at": now,
                    "prefilter_score": 0.9,
                }
            ],
        }
        for idx in range(1, 4)
    ]

    predictions = classify_threads(threads, client, source="slack", parse_error_limit=2)
    assert len(predictions) == 2  # stopped after hitting limit
    assert all(pred["parse_error"] for pred in predictions)
    assert all(pred["include_in_demand"] is False for pred in predictions)
    assert provider.calls == 2


runner = CliRunner()


def test_classify_threads_parse_errors_reset_on_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FlakyProvider(MockProvider):
        def __init__(self) -> None:
            super().__init__(
                response={"content": {"theme": "data_gap", "relevance": 0.7}}
            )
            self.failures = 0
            self.calls = 0

        def json_complete(self, payload):  # type: ignore[override]
            self.calls += 1
            if self.failures < 2:
                self.failures += 1
                raise ValueError("temporary failure")
            return super().json_complete(payload)

    provider = FlakyProvider()
    client = RepairingLLMClient(
        provider=provider,
        model="mock-model",
        api_key_env="OPENAI_API_KEY",
        timeout_s=5.0,
        max_output_tokens=64,
        cache_dir=tmp_path / "cache-flaky",
        repair_attempts=0,
    )

    now = datetime(2024, 1, 10, tzinfo=timezone.utc)
    threads = [
        {
            "thread_id": idx,
            "messages": [
                {
                    "message_id": idx * 10,
                    "thread_id": idx,
                    "user_id": 1,
                    "body": f"Body text {idx}",
                    "tokens": 40,
                    "sent_at": now,
                    "prefilter_score": 0.9,
                }
            ],
        }
        for idx in range(1, 4)
    ]

    predictions = classify_threads(
        threads, client, source="email", parse_error_limit=5, exec_user_ids=None
    )
    assert len(predictions) == 3
    assert sum(1 for pred in predictions if pred["parse_error"]) == 2
    assert predictions[-1]["parse_error"] is False
    assert predictions[-1]["theme"] == "data_gap"
    # ensure consecutive errors reset so we processed final thread
    assert provider.calls == 3
