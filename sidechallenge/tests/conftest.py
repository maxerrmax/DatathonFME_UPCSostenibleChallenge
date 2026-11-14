"""Tests must never call the real Helmcode API, even if the developer's shell has a key.

Tests that need a provider set the variables themselves with a mocked transport.
"""
import pytest


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    for var in ("HELMCODE_API_KEY", "HELMCODE_MODEL", "HELMCODE_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
