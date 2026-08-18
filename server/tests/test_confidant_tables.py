"""Pins the mod's confidant table to the server's.

The two live in different languages and are edited months apart, and they had already
drifted: ConfidantNames.cs was missing Justine and Caroline (12) and Lavenza (22), which
shifted every later id by one and made the mod send "Sadayo Kawakami" as the character
name for a Tae Takemi hang-out.

That was invisible because the server keys off ``confidant_id`` and never trusted the
name. Speaker attribution changes the stakes: the mod compares its own name for the
confidant against the speaker table inside the scene's BMD, so a shifted table stops it
recognising the confidant's own lines.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from social_link.arcana import CONFIDANTS

MOD_TABLE = (
    Path(__file__).resolve().parents[2]
    / "mod"
    / "P5RGenSocialLinks"
    / "Memory"
    / "ConfidantNames.cs"
)

# { 14, "Tae Takemi" }, — id and quoted name, ignoring the alignment padding.
ENTRY = re.compile(r"\{\s*(\d+)\s*,\s*\"([^\"]+)\"\s*\}")


def _mod_names() -> dict[int, str]:
    source = MOD_TABLE.read_text(encoding="utf-8")
    return {int(cid): name for cid, name in ENTRY.findall(source)}


def test_mod_table_is_present_and_parsed() -> None:
    # A regex that silently matches nothing would make every test below pass vacuously.
    names = _mod_names()
    assert len(names) >= 20, f"parsed only {len(names)} entries from {MOD_TABLE}"


def test_every_mod_id_exists_on_the_server() -> None:
    unknown = sorted(set(_mod_names()) - set(CONFIDANTS))
    assert not unknown, f"mod knows ids the server does not: {unknown}"


def test_every_server_id_exists_in_the_mod() -> None:
    missing = sorted(set(CONFIDANTS) - set(_mod_names()))
    assert not missing, f"server knows ids the mod does not: {missing}"


@pytest.mark.parametrize("confidant_id", sorted(CONFIDANTS))
def test_names_agree(confidant_id: int) -> None:
    names = _mod_names()
    assert names.get(confidant_id) == CONFIDANTS[confidant_id].name


def test_the_two_ids_confirmed_from_game_memory() -> None:
    # Ryuji=8 and Takemi=14 were read out of the live CMM session struct. Everything else
    # in both tables is cross-referenced rather than observed, so these two are the anchor
    # that says the whole numbering is aligned to the game and not just to itself.
    names = _mod_names()
    assert names[8] == "Ryuji Sakamoto"
    assert names[14] == "Tae Takemi"
    assert CONFIDANTS[8].name == "Ryuji Sakamoto"
    assert CONFIDANTS[14].name == "Tae Takemi"
