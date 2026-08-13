"""
Tests for the llama-server subprocess supervisor.

These never spawn llama-server itself — CI has neither the binary nor a GPU. The
command builder is checked directly, and the startup/teardown logic is driven with
stub processes, so the failure paths (missing binary, child dies during load, never
becomes healthy) are exercised deterministically rather than by luck.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from inference.config import LlamaServerConfig, ModelConfig
from inference.llama_process import LlamaServerProcess, LlamaServerStartupError


@pytest.fixture
def installed(tmp_path: Path) -> tuple[LlamaServerConfig, ModelConfig]:
    """Configs pointing at an existing fake binary and fake GGUF."""
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"MZ")
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    return (
        LlamaServerConfig(binary_path=str(binary), startup_timeout_s=1.0, startup_poll_s=0.01),
        ModelConfig(model_path=str(model)),
    )


class _StubProc:
    """Stands in for subprocess.Popen: `codes` is the poll() sequence to replay."""

    def __init__(self, codes: list[int | None]) -> None:
        self._codes = codes
        self.pid = 4242
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self._codes.pop(0) if len(self._codes) > 1 else self._codes[0]

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        return 0


# --- command construction ----------------------------------------------------


def test_command_includes_model_host_port_and_offload(
    installed: tuple[LlamaServerConfig, ModelConfig],
) -> None:
    server_cfg, model_cfg = installed
    command = LlamaServerProcess(server_cfg, model_cfg)._build_command()

    assert "--model" in command
    assert command[command.index("--host") + 1] == server_cfg.host
    assert command[command.index("--port") + 1] == str(server_cfg.port)
    assert command[command.index("--n-gpu-layers") + 1] == str(model_cfg.n_gpu_layers)
    assert command[command.index("--ctx-size") + 1] == str(model_cfg.n_ctx)


def test_command_respects_configured_gpu_layers(
    installed: tuple[LlamaServerConfig, ModelConfig],
) -> None:
    """Partial offload must reach the child — it is the knob for VRAM pressure."""
    server_cfg, model_cfg = installed
    partial = ModelConfig(model_path=model_cfg.model_path, n_gpu_layers=20)
    command = LlamaServerProcess(server_cfg, partial)._build_command()
    assert command[command.index("--n-gpu-layers") + 1] == "20"


def test_missing_binary_names_the_fetch_script(tmp_path: Path) -> None:
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    proc = LlamaServerProcess(
        LlamaServerConfig(binary_path=str(tmp_path / "absent.exe")),
        ModelConfig(model_path=str(model)),
    )
    with pytest.raises(LlamaServerStartupError, match="fetch-llama-server"):
        proc._build_command()


def test_missing_model_is_reported(tmp_path: Path) -> None:
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"MZ")
    proc = LlamaServerProcess(
        LlamaServerConfig(binary_path=str(binary)),
        ModelConfig(model_path=str(tmp_path / "absent.gguf")),
    )
    with pytest.raises(LlamaServerStartupError, match="GGUF model not found"):
        proc._build_command()


# --- startup -----------------------------------------------------------------


def test_start_returns_once_healthy(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    server_cfg, model_cfg = installed
    proc = LlamaServerProcess(server_cfg, model_cfg)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _StubProc([None]))
    monkeypatch.setattr(
        "inference.llama_process.LlamaServerClient.is_healthy", lambda self: True
    )
    proc.start()
    assert proc.is_running


def test_child_exiting_during_load_raises_with_exit_code(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CUDA OOM shows up as an early exit; the code and log must reach the caller."""
    server_cfg, model_cfg = installed
    proc = LlamaServerProcess(server_cfg, model_cfg)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _StubProc([1]))
    monkeypatch.setattr(
        "inference.llama_process.LlamaServerClient.is_healthy", lambda self: False
    )
    with pytest.raises(LlamaServerStartupError, match="exited with code 1"):
        proc.start()


def test_never_healthy_times_out(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    server_cfg, model_cfg = installed
    proc = LlamaServerProcess(server_cfg, model_cfg)
    stub = _StubProc([None])
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: stub)
    monkeypatch.setattr(
        "inference.llama_process.LlamaServerClient.is_healthy", lambda self: False
    )
    with pytest.raises(LlamaServerStartupError, match="did not become healthy"):
        proc.start()
    assert stub.terminated, "a child that never came up must not be left running"


def test_spawn_failure_is_wrapped(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    server_cfg, model_cfg = installed
    proc = LlamaServerProcess(server_cfg, model_cfg)

    def boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("access denied")

    monkeypatch.setattr(subprocess, "Popen", boom)
    with pytest.raises(LlamaServerStartupError, match="failed to spawn"):
        proc.start()


def test_start_is_idempotent_while_running(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second start must not spawn a second child fighting for the same port."""
    server_cfg, model_cfg = installed
    proc = LlamaServerProcess(server_cfg, model_cfg)
    spawns = []

    def spawn(*args: Any, **kwargs: Any) -> _StubProc:
        spawns.append(1)
        return _StubProc([None])

    monkeypatch.setattr(subprocess, "Popen", spawn)
    monkeypatch.setattr(
        "inference.llama_process.LlamaServerClient.is_healthy", lambda self: True
    )
    proc.start()
    proc.start()
    assert len(spawns) == 1


# --- teardown ----------------------------------------------------------------


def test_stop_terminates_child(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    server_cfg, model_cfg = installed
    proc = LlamaServerProcess(server_cfg, model_cfg)
    stub = _StubProc([None])
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: stub)
    monkeypatch.setattr(
        "inference.llama_process.LlamaServerClient.is_healthy", lambda self: True
    )
    proc.start()
    proc.stop()
    assert stub.terminated
    assert not proc.is_running


def test_stop_escalates_to_kill_on_timeout(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    server_cfg, model_cfg = installed
    proc = LlamaServerProcess(server_cfg, model_cfg)
    stub = _StubProc([None])

    def stubborn(timeout: float | None = None) -> int:
        if not stub.killed:
            raise subprocess.TimeoutExpired(cmd="llama-server", timeout=timeout or 0)
        return 0

    stub.wait = stubborn  # type: ignore[method-assign]
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: stub)
    monkeypatch.setattr(
        "inference.llama_process.LlamaServerClient.is_healthy", lambda self: True
    )
    proc.start()
    proc.stop()
    assert stub.killed


def test_stop_without_start_is_safe(
    installed: tuple[LlamaServerConfig, ModelConfig],
) -> None:
    server_cfg, model_cfg = installed
    LlamaServerProcess(server_cfg, model_cfg).stop()


# --- port conflict -----------------------------------------------------------


def test_occupied_port_is_reported_before_spawning(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A foreign listener answers /health with its own 404, which reads as a half-started
    model rather than a port conflict. The pre-flight check must name the real cause,
    and must do so without launching a doomed child.
    """
    import socket as socket_module

    server_cfg, model_cfg = installed
    with socket_module.socket(socket_module.AF_INET, socket_module.SOCK_STREAM) as squatter:
        squatter.bind((server_cfg.host, 0))
        squatter.listen(1)
        taken_port = squatter.getsockname()[1]

        conflicted = LlamaServerConfig(
            binary_path=server_cfg.binary_path, port=taken_port
        )
        proc = LlamaServerProcess(conflicted, model_cfg)

        spawns: list[int] = []
        monkeypatch.setattr(
            subprocess, "Popen", lambda *a, **k: spawns.append(1) or _StubProc([None])
        )

        with pytest.raises(LlamaServerStartupError, match="already in use"):
            proc.start()
        assert not spawns, "must not spawn a child that cannot bind"


def test_free_port_passes_the_preflight_check(
    installed: tuple[LlamaServerConfig, ModelConfig], monkeypatch: pytest.MonkeyPatch
) -> None:
    server_cfg, model_cfg = installed
    proc = LlamaServerProcess(server_cfg, model_cfg)
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: _StubProc([None]))
    monkeypatch.setattr(
        "inference.llama_process.LlamaServerClient.is_healthy", lambda self: True
    )
    proc.start()
    assert proc.is_running
