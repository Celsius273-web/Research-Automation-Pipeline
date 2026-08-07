"""Unit tests for host memory helpers."""

from __future__ import annotations

import builtins

from src.tools import memory


def test_available_memory_gb_works_without_psutil(monkeypatch) -> None:
    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "psutil" or name.startswith("psutil."):
            raise ImportError("No module named 'psutil'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    value = memory.available_memory_gb()
    assert isinstance(value, float)
    assert value > 0.0


def test_available_memory_gb_prefers_psutil_when_present(monkeypatch) -> None:
    monkeypatch.setattr(memory, "_available_from_psutil", lambda: 4.0)
    assert memory.available_memory_gb() == 4.0
