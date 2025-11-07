from __future__ import annotations

from datetime import datetime, timedelta, timezone

from data_needs_reporter.report.classify import pack_thread


def test_pack_thread_respects_message_and_token_caps():
    start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    messages = []
    for idx in range(30):
        body = " ".join(f"token{idx}_{j}" for j in range(100))  # ~100 token payload
        messages.append(
            {
                "message_id": idx + 1,
                "user_id": idx % 4,
                "sent_at": start + timedelta(minutes=idx),
                "body": body,
                # Token counter is mocked to stay within the 900-token cap for 20 messages.
                "tokens": 40,
            }
        )

    packed = pack_thread(messages, max_messages=20, max_tokens=900)

    assert len(packed["messages"]) == 20
    assert packed["token_total"] <= 900
    assert packed["message_limit_hit"] is True
    assert packed["token_limit_hit"] is False
