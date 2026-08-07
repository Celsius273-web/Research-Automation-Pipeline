"""Host memory helpers for pre-run safety checks."""

from __future__ import annotations

import os
import platform
import re
import subprocess
from pathlib import Path


def _available_from_psutil() -> float | None:
    try:
        import psutil
    except ImportError:
        return None
    return float(psutil.virtual_memory().available) / (1024**3)


def _available_from_proc_meminfo() -> float | None:
    meminfo = Path("/proc/meminfo")
    if not meminfo.exists():
        return None
    try:
        text = meminfo.read_text(encoding="utf-8")
    except OSError:
        return None
    # Prefer MemAvailable; fall back to MemFree + Cached.
    available_match = re.search(r"^MemAvailable:\s+(\d+)\s+kB", text, re.MULTILINE)
    if available_match:
        return float(available_match.group(1)) / (1024**2)
    free_match = re.search(r"^MemFree:\s+(\d+)\s+kB", text, re.MULTILINE)
    cached_match = re.search(r"^Cached:\s+(\d+)\s+kB", text, re.MULTILINE)
    if free_match and cached_match:
        return (float(free_match.group(1)) + float(cached_match.group(1))) / (1024**2)
    return None


def _parse_vm_stat_pages(text: str, key: str) -> int | None:
    match = re.search(rf"^{re.escape(key)}:\s+(\d+)", text, re.MULTILINE)
    if not match:
        return None
    return int(match.group(1))


def _available_from_darwin_vm_stat() -> float | None:
    if platform.system() != "Darwin":
        return None
    try:
        page_size = int(
            subprocess.check_output(["pagesize"], text=True, stderr=subprocess.DEVNULL).strip()
        )
        vm_stat = subprocess.check_output(
            ["vm_stat"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None

    # Pages that can be reclaimed without swapping under memory pressure.
    free_pages = _parse_vm_stat_pages(vm_stat, "Pages free") or 0
    speculative = _parse_vm_stat_pages(vm_stat, "Pages speculative") or 0
    inactive = _parse_vm_stat_pages(vm_stat, "Pages inactive") or 0
    purgeable = _parse_vm_stat_pages(vm_stat, "Pages purgeable") or 0
    available_pages = free_pages + speculative + inactive + purgeable
    if available_pages <= 0:
        return None
    return float(available_pages * page_size) / (1024**3)


def available_memory_gb() -> float:
    """Return currently available system memory in gigabytes.

    Prefers psutil when installed, but never requires it — Engineer must not
    fail solely because an optional host dependency is missing.
    """
    for reader in (_available_from_psutil, _available_from_proc_meminfo, _available_from_darwin_vm_stat):
        value = reader()
        if value is not None:
            return value

    # Last resort: assume enough memory so the run can proceed; the caller still
    # logs the measured value.
    total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    return float(total) / (1024**3)
