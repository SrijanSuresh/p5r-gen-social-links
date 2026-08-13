"""
Supervises the llama-server child process.

Owning the model as a subprocess rather than an in-process binding decouples three
things that used to be fused: the API cannot be blocked by a 15s model load, a CUDA
OOM no longer takes down the process the game is talking to, and swapping the GGUF
does not mean restarting the API. See learning.md Ch. 62.
"""

from __future__ import annotations

import logging
import subprocess
import time
from pathlib import Path

from .config import LlamaServerConfig, ModelConfig
from .llama_client import LlamaServerClient

log = logging.getLogger(__name__)

SERVER_DIR = Path(__file__).parent.parent


class LlamaServerStartupError(RuntimeError):
    """The child process could not be started, or never became healthy."""


class LlamaServerProcess:
    """Starts, waits for, and tears down a llama-server child process."""

    def __init__(self, server_cfg: LlamaServerConfig, model_cfg: ModelConfig) -> None:
        self._cfg = server_cfg
        self._model_cfg = model_cfg
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_file: Path = SERVER_DIR / "logs" / "llama-server.log"

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def _resolve(self, relative: str) -> Path:
        """Resolve a config path against server/, so cwd never changes the outcome."""
        path = Path(relative)
        return path if path.is_absolute() else SERVER_DIR / path

    def _build_command(self) -> list[str]:
        binary = self._resolve(self._cfg.binary_path)
        model = self._resolve(self._model_cfg.model_path)

        if not binary.exists():
            raise LlamaServerStartupError(
                f"llama-server.exe not found at {binary}. "
                "Fetch it with: scripts/fetch-llama-server.ps1"
            )
        if not model.exists():
            raise LlamaServerStartupError(
                f"GGUF model not found at {model}. See server/models/README.txt"
            )

        return [
            str(binary),
            "--model", str(model),
            "--host", self._cfg.host,
            "--port", str(self._cfg.port),
            # -1 offloads every layer to the GPU. On an 8 GB card this is the setting
            # most likely to need lowering: P5R holds its own render targets in VRAM,
            # so the budget while playing is smaller than the free figure reported by
            # nvidia-smi with the game closed.
            "--n-gpu-layers", str(self._model_cfg.n_gpu_layers),
            "--ctx-size", str(self._model_cfg.n_ctx),
        ]

    def start(self) -> None:
        """
        Spawn the child and block until it reports healthy.

        Returning normally means the weights are resident and /v1/chat/completions
        will answer — callers need no further readiness check.
        """
        if self.is_running:
            log.info("llama-server already running (pid %s)", self._proc.pid)  # type: ignore[union-attr]
            return

        command = self._build_command()
        self._log_file.parent.mkdir(parents=True, exist_ok=True)

        log.info("Starting llama-server on %s …", self._cfg.base_url)
        log.info("  %s", " ".join(command))
        log.info("  child output -> %s", self._log_file)

        # Redirect to a file rather than PIPE. llama.cpp logs steadily while loading,
        # and an unread PIPE fills its buffer and blocks the child mid-startup — the
        # deadlock would look exactly like a slow model load.
        handle = self._log_file.open("wb")
        try:
            self._proc = subprocess.Popen(
                command,
                stdout=handle,
                stderr=subprocess.STDOUT,
                cwd=SERVER_DIR,
            )
        except OSError as exc:
            handle.close()
            raise LlamaServerStartupError(
                f"failed to spawn {command[0]}: {exc}"
            ) from exc

        self._await_health()

    def _await_health(self) -> None:
        """Poll /health until ready, the child dies, or the timeout expires."""
        client = LlamaServerClient(self._cfg, self._model_cfg)
        deadline = time.monotonic() + self._cfg.startup_timeout_s
        try:
            while time.monotonic() < deadline:
                # Check liveness before health: a child that already exited will never
                # answer, and its exit code plus log tail is the useful error.
                exit_code = self._proc.poll()  # type: ignore[union-attr]
                if exit_code is not None:
                    raise LlamaServerStartupError(
                        f"llama-server exited with code {exit_code} during startup.\n"
                        f"{self._log_tail()}"
                    )
                if client.is_healthy():
                    log.info("llama-server ready at %s", self._cfg.base_url)
                    return
                time.sleep(self._cfg.startup_poll_s)
        finally:
            client.close()

        self.stop()
        raise LlamaServerStartupError(
            f"llama-server did not become healthy within "
            f"{self._cfg.startup_timeout_s}s.\n{self._log_tail()}"
        )

    def _log_tail(self, lines: int = 20) -> str:
        """Last few lines of the child's log — the actual reason for most failures."""
        try:
            content = self._log_file.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return f"(could not read {self._log_file}: {exc})"
        tail = content.splitlines()[-lines:]
        return "--- llama-server log tail ---\n" + "\n".join(tail)

    def stop(self, timeout_s: float = 10.0) -> None:
        """Terminate the child, escalating to kill if it does not exit."""
        if self._proc is None:
            return
        if self._proc.poll() is not None:
            self._proc = None
            return

        log.info("Stopping llama-server (pid %s) …", self._proc.pid)
        self._proc.terminate()
        try:
            self._proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            log.warning("llama-server ignored terminate; killing.")
            self._proc.kill()
            self._proc.wait()
        finally:
            self._proc = None
