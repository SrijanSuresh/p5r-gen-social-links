"""
Ties a child process's lifetime to ours using a Windows Job Object.

The lifespan handler stops llama-server on a clean shutdown, but nothing runs when the
parent is killed outright — closing the console window, Task Manager, or `taskkill /F`.
Observed directly: force-killing the API left llama-server alive, still holding port
8766 and ~4.9 GB of VRAM, which then failed the next start's port check.

A Job Object with JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE moves the cleanup into the
kernel. Every handle to the job closes when our process dies — however it dies — and
Windows then terminates every process in the job. This is the one mechanism that
survives SIGKILL-equivalents, because it does not require us to run any code.

No-ops on non-Windows so the module stays importable on CI (Linux).
"""

from __future__ import annotations

import ctypes
import logging
import sys
from ctypes import wintypes
from typing import Any

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

# winnt.h
_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JobObjectExtendedLimitInformation = 9

# processthreadsapi.h — the rights needed to put a process into a job and end it.
_PROCESS_SET_QUOTA = 0x0100
_PROCESS_TERMINATE = 0x0001


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.POINTER(wintypes.ULONG)),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class KillOnCloseJob:
    """
    A Job Object that terminates its members when this object is closed or the
    process exits.

    The handle must stay open for as long as the child should live — releasing it is
    precisely what triggers the kill — so callers hold this for the process lifetime.
    """

    def __init__(self) -> None:
        self._job: int | None = None

    @property
    def is_active(self) -> bool:
        return self._job is not None

    def _kernel32(self) -> Any:
        return ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]

    def adopt(self, pid: int) -> bool:
        """
        Put `pid` into a kill-on-close job. Returns False if that was not possible.

        Failure is never fatal: it costs orphan protection, not functionality, so the
        caller logs and continues rather than refusing to serve.
        """
        if not IS_WINDOWS:
            return False

        kernel32 = self._kernel32()
        try:
            if self._job is None:
                job = kernel32.CreateJobObjectW(None, None)
                if not job:
                    log.warning(
                        "CreateJobObject failed (%s); llama-server will not be "
                        "auto-killed if this process is force-terminated.",
                        ctypes.get_last_error(),
                    )
                    return False

                info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
                info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                if not kernel32.SetInformationJobObject(
                    job,
                    _JobObjectExtendedLimitInformation,
                    ctypes.byref(info),
                    ctypes.sizeof(info),
                ):
                    log.warning(
                        "SetInformationJobObject failed (%s).", ctypes.get_last_error()
                    )
                    kernel32.CloseHandle(job)
                    return False
                self._job = job

            # OpenProcess by pid rather than reaching for Popen._handle, which is a
            # private attribute of the subprocess implementation.
            handle = kernel32.OpenProcess(
                _PROCESS_SET_QUOTA | _PROCESS_TERMINATE, False, pid
            )
            if not handle:
                log.warning(
                    "OpenProcess(%s) failed (%s).", pid, ctypes.get_last_error()
                )
                return False
            try:
                if not kernel32.AssignProcessToJobObject(self._job, handle):
                    # Nested jobs are supported from Windows 8 on, so this is rare;
                    # some sandboxes still forbid it.
                    log.warning(
                        "AssignProcessToJobObject failed (%s); the child may outlive "
                        "a force-kill of this process.",
                        ctypes.get_last_error(),
                    )
                    return False
            finally:
                kernel32.CloseHandle(handle)

            log.info("llama-server (pid %s) adopted into kill-on-close job.", pid)
            return True
        except OSError as exc:
            log.warning("Job Object setup failed (%s).", exc)
            return False

    def close(self) -> None:
        """Close the job handle, terminating any surviving members."""
        if self._job is None:
            return
        try:
            self._kernel32().CloseHandle(self._job)
        except OSError as exc:
            log.warning("Closing job handle failed (%s).", exc)
        finally:
            self._job = None
