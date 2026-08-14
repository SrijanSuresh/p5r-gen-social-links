"""
Check a SigScan pattern against p5r.exe on disk, before trusting it at runtime.

A signature taken from a debugger is a sample of one: it matched the address you were
looking at. Whether it also matches somewhere else in a 378 MB executable is a separate
question, and the runtime scanner will not tell you - Reloaded's FindPattern returns the
first hit and says nothing about the second. A pattern with two occurrences resolves
happily and hooks the wrong instruction.

This reads the PE on disk and answers three things a live scan cannot:

  * how many times the pattern occurs (1 is the only acceptable answer),
  * the RVA of each hit, to compare against `module+offset` in a disassembler,
  * whether a given expected RVA actually holds those bytes.

Usage:
    python scripts/verify-signature.py "48 63 53 30 48 8B 43 20 0F B6 3C 02 85 FF 0F 85"
    python scripts/verify-signature.py "48 8B 05 ?? ?? ?? ?? 48 85 C0" --expect 17A3D1F
    python scripts/verify-signature.py <pattern> --exe "D:/Games/P5R/P5R.exe"

Exit code is 0 when the pattern occurs exactly once (and matches --expect, if given),
1 otherwise - so it can gate a build.
"""

from __future__ import annotations

import argparse
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_EXE = Path(r"C:\Games\Persona 5 Royal\P5R.exe")

PE32PLUS_MAGIC = 0x20B


@dataclass(frozen=True)
class Section:
    name: str
    virtual_address: int
    virtual_size: int
    raw_pointer: int
    raw_size: int


@dataclass(frozen=True)
class Image:
    data: bytes
    image_base: int
    size_of_image: int
    sections: tuple[Section, ...]

    def offset_to_rva(self, offset: int) -> tuple[int, str] | None:
        for section in self.sections:
            if section.raw_pointer <= offset < section.raw_pointer + section.raw_size:
                return section.virtual_address + (offset - section.raw_pointer), section.name
        return None

    def rva_to_offset(self, rva: int) -> tuple[int, str] | None:
        for section in self.sections:
            span = max(section.virtual_size, section.raw_size)
            if section.virtual_address <= rva < section.virtual_address + span:
                return section.raw_pointer + (rva - section.virtual_address), section.name
        return None


class PeFormatError(ValueError):
    """The file is not a PE32+ image we can read."""


def load_image(path: Path) -> Image:
    data = path.read_bytes()
    if data[:2] != b"MZ":
        raise PeFormatError(f"{path} does not start with 'MZ'")

    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe : pe + 4] != b"PE\0\0":
        raise PeFormatError(f"{path} has no PE header at 0x{pe:X}")

    section_count = struct.unpack_from("<H", data, pe + 6)[0]
    optional_size = struct.unpack_from("<H", data, pe + 20)[0]
    optional = pe + 24

    magic = struct.unpack_from("<H", data, optional)[0]
    if magic != PE32PLUS_MAGIC:
        raise PeFormatError(f"{path} is not PE32+ (optional header magic 0x{magic:X})")

    image_base = struct.unpack_from("<Q", data, optional + 24)[0]
    size_of_image = struct.unpack_from("<I", data, optional + 56)[0]

    table = optional + optional_size
    sections: list[Section] = []
    for index in range(section_count):
        entry = table + index * 40
        name = data[entry : entry + 8].rstrip(b"\0").decode("ascii", "replace")
        virtual_size, virtual_address, raw_size, raw_pointer = struct.unpack_from(
            "<IIII", data, entry + 8
        )
        sections.append(Section(name, virtual_address, virtual_size, raw_pointer, raw_size))

    return Image(data, image_base, size_of_image, tuple(sections))


def pattern_to_regex(pattern: str) -> re.Pattern[bytes]:
    """
    Translate a Reloaded-style pattern ("48 8B ?? C0") into a byte regex.

    Wildcards have to survive the translation as single-byte matches rather than being
    dropped, or the pattern silently gets shorter and matches far more than intended.
    """
    parts = pattern.replace("??", "?").split()
    if not parts:
        raise ValueError("empty pattern")

    chunks: list[bytes] = []
    for part in parts:
        if part == "?":
            chunks.append(b".")
            continue
        if len(part) != 2:
            raise ValueError(f"token {part!r} is neither a byte nor a wildcard")
        chunks.append(re.escape(bytes([int(part, 16)])))
    return re.compile(b"".join(chunks), re.DOTALL)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pattern", help='byte pattern, e.g. "48 8B 05 ?? ?? ?? ??"')
    parser.add_argument("--exe", type=Path, default=DEFAULT_EXE, help="path to p5r.exe")
    parser.add_argument(
        "--expect", default=None, help="RVA the pattern should resolve to, in hex"
    )
    args = parser.parse_args()

    if not args.exe.is_file():
        print(f"FAIL  no such file: {args.exe}", file=sys.stderr)
        return 1

    try:
        image = load_image(args.exe)
        regex = pattern_to_regex(args.pattern)
    except (PeFormatError, ValueError, OSError) as error:
        print(f"FAIL  {error}", file=sys.stderr)
        return 1

    print(f"exe          {args.exe}")
    print(f"size         {len(image.data):,} bytes (0x{len(image.data):X})")
    print(f"image base   0x{image.image_base:X}")
    print(f"SizeOfImage  0x{image.size_of_image:X}")
    print(f"pattern      {args.pattern}")

    hits = [match.start() for match in regex.finditer(image.data)]
    print(f"occurrences  {len(hits)}")
    for offset in hits:
        located = image.offset_to_rva(offset)
        if located is None:
            print(f"  file 0x{offset:X} -> outside every section (not loaded)")
            continue
        rva, section = located
        print(f"  file 0x{offset:X} -> p5r.exe+{rva:X}  (section {section})")

    ok = len(hits) == 1
    if not ok:
        print("\nFAIL  a signature must occur exactly once; the runtime scanner takes")
        print("      the first hit and cannot warn you about the others.")

    if args.expect is not None:
        expected = int(args.expect, 16)
        located = image.rva_to_offset(expected)
        if located is None:
            print(f"\nFAIL  RVA 0x{expected:X} is not inside any section")
            ok = False
        else:
            offset, section = located
            found = offset in hits
            width = len(args.pattern.split())
            print(f"\nexpected     p5r.exe+{expected:X} (section {section})")
            print(f"bytes there  {image.data[offset:offset + width].hex(' ').upper()}")
            print(f"match        {found}")
            ok = ok and found

    print("\nOK" if ok else "")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
