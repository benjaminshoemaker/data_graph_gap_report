from __future__ import annotations

import logging
from io import StringIO

from data_needs_reporter.utils.logging import init_logger, run_context


def test_run_context_adds_run_id_to_log_records() -> None:
    logger = init_logger("dnr.test.logging")

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(run_id)s|%(message)s"))
    logger.addHandler(handler)

    with run_context(logger) as run_id:
        logger.info("hello world")

    output_lines = [line for line in stream.getvalue().splitlines() if line]
    assert output_lines
    assert run_id in output_lines[0]
    assert "hello world" in output_lines[0]

    stream.truncate(0)
    stream.seek(0)

    logger.info("outside context")
    outside_lines = [line for line in stream.getvalue().splitlines() if line]
    assert outside_lines
    assert outside_lines[-1].startswith("-")
    assert "outside context" in outside_lines[-1]

    logger.removeHandler(handler)
    handler.close()
