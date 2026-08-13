"""
Tests for the kill-on-close Job Object.

The behaviour that matters — a child dying when its force-killed parent does — cannot
be asserted in-process, since proving it requires killing the test runner. It was
verified out-of-band with a spawned parent/child pair, and the observation is recorded
in learning.md Ch. 63. What is tested here is everything around it: the no-op path on
non-Windows so CI stays green, and that failures degrade rather than raise.
"""

from __future__ import annotations

import subprocess
import sys
import time

import pytest

from inference.win_job import KillOnCloseJob

WINDOWS_ONLY = pytest.mark.skipif(
    sys.platform != "win32", reason="Job Objects are a Windows API"
)


def test_starts_inactive() -> None:
    assert not KillOnCloseJob().is_active


def test_close_without_adopt_is_safe() -> None:
    KillOnCloseJob().close()


def test_adopt_missing_pid_returns_false_and_does_not_raise() -> None:
    """
    Orphan protection is a nicety; losing it must never stop the server serving.
    PID 0xFFFFFFF is not a live process on any sane system.
    """
    assert KillOnCloseJob().adopt(0xFFFFFFF) is False


@pytest.mark.skipif(sys.platform == "win32", reason="covers the non-Windows no-op")
def test_adopt_is_a_no_op_off_windows() -> None:
    job = KillOnCloseJob()
    assert job.adopt(1) is False
    assert not job.is_active


@WINDOWS_ONLY
def test_adopt_live_process_activates_the_job() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        job = KillOnCloseJob()
        assert job.adopt(child.pid) is True
        assert job.is_active
    finally:
        child.kill()
        child.wait()


@WINDOWS_ONLY
def test_closing_the_job_kills_its_member() -> None:
    """The kill-on-close limit, exercised directly rather than via a parent death."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        job = KillOnCloseJob()
        assert job.adopt(child.pid) is True
        assert child.poll() is None

        job.close()

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and child.poll() is None:
            time.sleep(0.1)
        assert child.poll() is not None, "closing the job must terminate its members"
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()


@WINDOWS_ONLY
def test_close_is_idempotent() -> None:
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        job = KillOnCloseJob()
        job.adopt(child.pid)
        job.close()
        job.close()
        assert not job.is_active
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
