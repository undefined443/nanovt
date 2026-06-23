"""Tests for OpenAI retry classification."""

from nanovt.cli import _should_retry_openai_error


class _StatusError(Exception):
    """Test exception with an OpenAI-style status code."""

    def __init__(self, status_code: int) -> None:
        """Initialize the fake status error."""
        self.status_code = status_code


class APIConnectionError(Exception):
    """Test exception matching the OpenAI connection error class name."""


class APITimeoutError(Exception):
    """Test exception matching the OpenAI timeout error class name."""


def test_retries_transient_status_codes() -> None:
    """Retry rate-limit, timeout, conflict, and server errors."""
    assert _should_retry_openai_error(_StatusError(408))
    assert _should_retry_openai_error(_StatusError(409))
    assert _should_retry_openai_error(_StatusError(429))
    assert _should_retry_openai_error(_StatusError(500))


def test_does_not_retry_non_transient_status_code() -> None:
    """Do not retry non-transient client errors."""
    assert not _should_retry_openai_error(_StatusError(400))


def test_retries_openai_connection_error_names() -> None:
    """Retry OpenAI SDK connection and timeout error class names."""
    assert _should_retry_openai_error(APIConnectionError())
    assert _should_retry_openai_error(APITimeoutError())
