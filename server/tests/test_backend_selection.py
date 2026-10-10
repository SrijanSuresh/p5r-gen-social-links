"""
Picking an inference engine, and what happens when you pick the unfinished one.

The project has always planned two backends. Only one works, and the useful property of
the seam is that selecting the other one says exactly what is missing rather than failing
somewhere deeper with a confusing error.
"""

from __future__ import annotations

import pytest

from inference.backend import (
    DEFAULT_BACKEND,
    TRITON_MISSING_PIECES,
    Backend,
    BackendNotBuiltError,
    UnknownBackendError,
    require_available,
    resolve,
)


# --- resolving a name ---------------------------------------------------------


def test_the_default_is_the_backend_that_works() -> None:
    assert DEFAULT_BACKEND is Backend.LLAMA_SERVER
    assert resolve("") is Backend.LLAMA_SERVER


def test_an_unset_environment_variable_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    assert resolve() is DEFAULT_BACKEND


def test_the_environment_variable_selects_the_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFERENCE_BACKEND", "triton")
    assert resolve() is Backend.TRITON


def test_names_arriving_from_a_shell_are_forgiven() -> None:
    # Whitespace and case both survive a copy-paste into a .bat file.
    assert resolve("  TRITON  ") is Backend.TRITON
    assert resolve("Llama") is Backend.LLAMA_SERVER
    assert resolve("   ") is DEFAULT_BACKEND


def test_an_unknown_name_names_the_valid_ones() -> None:
    with pytest.raises(UnknownBackendError) as caught:
        resolve("vllm")

    message = str(caught.value)
    assert "vllm" in message
    # A rejection that does not say what would have worked costs another round trip.
    for name in Backend.names():
        assert name in message


# --- availability -------------------------------------------------------------


def test_the_llama_backend_is_available() -> None:
    require_available(Backend.LLAMA_SERVER)   # must not raise


def test_selecting_triton_explains_what_is_missing() -> None:
    with pytest.raises(BackendNotBuiltError) as caught:
        require_available(Backend.TRITON)

    message = str(caught.value)
    # The kernel's existence makes the backend look closer than it is, so the error has
    # to enumerate the distance rather than just refusing.
    for piece in TRITON_MISSING_PIECES:
        assert piece in message
    # And it has to say how to get working again.
    assert "INFERENCE_BACKEND=llama" in message


def test_not_built_is_distinct_from_not_recognised() -> None:
    # Different remedies: one is fixed by the person who set the variable, the other by
    # writing the missing code. Catching them together would conflate the two.
    assert not issubclass(BackendNotBuiltError, UnknownBackendError)
    assert issubclass(BackendNotBuiltError, NotImplementedError)
    assert issubclass(UnknownBackendError, ValueError)


def test_the_missing_pieces_list_is_not_empty() -> None:
    # If the Triton backend is ever finished this list empties and the guard above should
    # go with it; an empty list today would mean the error says nothing.
    assert len(TRITON_MISSING_PIECES) >= 1
    assert all(piece.strip() for piece in TRITON_MISSING_PIECES)
