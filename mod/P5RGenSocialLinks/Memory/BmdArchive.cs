using System;
using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// The MSG1 file wrapped around a set of <see cref="BmdMessage"/> records.
///
/// <code>
/// offset  size  field
/// 0x00       4  FileType / flags
/// 0x04       4  FileSize                  bytes, including the relocation table
/// 0x08       4  Magic                     "MSG1"
/// 0x0C       4  ExtSize
/// 0x10       4  RelocationTable           plain file offset — see below
/// 0x14       4  RelocationTableSize
/// 0x18       4  DialogCount
/// 0x1C       2  IsRelocated
/// 0x1E       2  Version
/// 0x20     8*n  DialogEntry[n]            { int Kind; int Address; }   n = DialogCount
///  ...      16  SpeakerTable              { int ArrayAddress; int Count; int Ext; int _ }
/// </code>
///
/// **Every stored address is relative to the position of the field that stores it.** A
/// dialogue entry's address field at 0x24 holding 0x18C means the message is at 0x1B0;
/// the speaker array field at 0x1A0 holding 0x1FE0 means the array is at 0x2180; each
/// name address inside that array is relative to its own slot. That is what
/// <c>IsRelocated = 1</c> means for this file — the fields named by the relocation table
/// have been rewritten in place as self-relative deltas.
///
/// <c>RelocationTable</c> is the single exception, and it has to be: it is the one field
/// that cannot be listed in the table it points at. It is a plain file offset, confirmed
/// on two files by the table ending exactly on the declared file size.
///
/// Nothing here searches. DialogCount says how many messages exist and the dialogue table
/// says where each one is, so the whole scene enumerates in a loop — including the
/// messages the player has not reached, which is what pre-generation needs.
///
/// See learning.md Ch. 79.
/// </summary>
internal readonly struct BmdArchive
{
    /// Distance from the file start to the "MSG1" magic.
    private const int MagicOffset = 0x08;

    private const int FileSizeOffset    = 0x04;
    private const int RelocTableOffset  = 0x10;
    private const int DialogCountOffset = 0x18;
    private const int IsRelocatedOffset = 0x1C;
    private const int DialogTableOffset = 0x20;
    private const int DialogEntryBytes  = 8;
    private const int SpeakerTableBytes = 16;

    /// Sanity bounds. Every one of these sizes a table walk over memory that may have been
    /// recycled, so an implausible value has to fail here rather than downstream.
    private const int MaxDialogs  = 4096;
    private const int MaxSpeakers = 1024;

    /// Bounds on the declared file size, which decides how much is copied per read.
    private const int MinFileBytes = 0x40;
    private const int MaxFileBytes = 16 * 1024 * 1024;

    /// <summary>Longest speaker name accepted, in bytes, before the string is truncated.</summary>
    private const int MaxNameBytes = 64;

    /// <summary>One message, located and decoded.</summary>
    internal readonly struct Entry
    {
        internal int    Index      { get; }
        internal int    Offset     { get; }   // file offset of the record header
        internal int    TextOffset { get; }   // file offset of its text buffer
        internal int    SpeakerId  { get; }
        internal string Name       { get; }
        internal bool   IsSelection { get; }

        internal Entry(int index, int offset, int textOffset, int speakerId,
                       string name, bool isSelection)
        {
            Index       = index;
            Offset      = offset;
            TextOffset  = textOffset;
            SpeakerId   = speakerId;
            Name        = name;
            IsSelection = isSelection;
        }
    }

    private readonly byte[]  _file;
    private readonly Entry[] _entries;
    private readonly int     _speakerArray;

    internal int    SpeakerCount  { get; }
    internal int    DialogCount   => _entries.Length;

    /// <summary>
    /// File offset where dialogue data stops and the speaker names begin.
    ///
    /// Everything past it is names and the relocation table. "Girl's Father" is thirteen
    /// printable characters sitting inside an armed region, and the pool scanner has been
    /// finding it — knowing where the dialogue ends is what makes not overwriting it a
    /// decision rather than luck.
    /// </summary>
    internal int DialogueEnd { get; }

    internal IReadOnlyList<Entry> Entries => _entries;

    private BmdArchive(byte[] file, Entry[] entries, int speakerArray, int speakerCount,
                       int dialogueEnd)
    {
        _file         = file;
        _entries      = entries;
        _speakerArray = speakerArray;
        SpeakerCount  = speakerCount;
        DialogueEnd   = dialogueEnd;
    }

    internal bool HasSpeakerTable => SpeakerCount > 0 && _speakerArray > 0;

    /// <summary>
    /// Locate the "MSG1" magic at or before <paramref name="limit"/>, searching backwards.
    ///
    /// Backwards because the caller knows where a record is, not where its file begins.
    /// Nearest match wins: several BMDs can be resident at once, and a record belongs to
    /// the file immediately behind it.
    /// </summary>
    internal static bool TryFindMagicBefore(byte[] window, int limit, out int magicIndex)
    {
        magicIndex = -1;
        if (window is null) return false;

        int start = Math.Min(limit, window.Length - 4);
        for (int i = start; i >= 0; i--)
        {
            if (window[i]     == (byte)'M' && window[i + 1] == (byte)'S' &&
                window[i + 2] == (byte)'G' && window[i + 3] == (byte)'1')
            {
                magicIndex = i;
                return true;
            }
        }
        return false;
    }

    /// <summary>
    /// Read the file's declared size from the four bytes at file+0x04.
    ///
    /// The caller needs this before it can read the archive: speaker names sit past the end
    /// of every message, so a window sized to reach the records falls short of the table
    /// that names them.
    /// </summary>
    internal static bool TryReadFileSize(byte[] window, int magicIndex, out int fileSize)
    {
        fileSize = 0;

        int fileStart = magicIndex - MagicOffset;
        if (window is null || fileStart < 0 || fileStart + MagicOffset > window.Length) return false;

        int size = ReadInt32(window, fileStart + FileSizeOffset);
        if (size < MinFileBytes || size > MaxFileBytes) return false;

        fileSize = size;
        return true;
    }

    /// <summary>
    /// Parse a complete archive out of <paramref name="file"/>, whose index 0 is the file's
    /// first byte.
    ///
    /// Every message is decoded up front. There is no partial success: a dialogue entry
    /// that does not resolve to a parseable header means the addressing is not what this
    /// struct believes, and half a scene attributed under a wrong rule is worse than none.
    /// </summary>
    internal static bool TryParse(byte[] file, out BmdArchive archive)
    {
        archive = default;
        if (file is null || file.Length < DialogTableOffset + DialogEntryBytes) return false;

        if (file[MagicOffset]     != (byte)'M' || file[MagicOffset + 1] != (byte)'S' ||
            file[MagicOffset + 2] != (byte)'G' || file[MagicOffset + 3] != (byte)'1') return false;

        int dialogCount = ReadInt32(file, DialogCountOffset);
        if (dialogCount < 1 || dialogCount > MaxDialogs) return false;

        int speakerTable = DialogTableOffset + DialogEntryBytes * dialogCount;
        if (speakerTable < 0 || speakerTable + SpeakerTableBytes > file.Length) return false;

        // Self-relative: the array lives at the address field's own position plus its value.
        int speakerCount = ReadInt32(file, speakerTable + 4);
        if (speakerCount < 0 || speakerCount > MaxSpeakers) return false;

        int speakerArray = 0;
        if (speakerCount > 0)
        {
            speakerArray = speakerTable + ReadInt32(file, speakerTable);
            if (speakerArray < 0 || speakerArray + 4 * speakerCount > file.Length)
                speakerArray = 0;
        }

        var entries = new Entry[dialogCount];
        for (int i = 0; i < dialogCount; i++)
        {
            int field  = DialogTableOffset + DialogEntryBytes * i + 4;
            int kind   = ReadInt32(file, field - 4);
            int offset = field + ReadInt32(file, field);

            if (offset < 0 || offset + BmdMessage.HeaderBytes > file.Length) return false;

            var header = new byte[BmdMessage.HeaderBytes];
            Array.Copy(file, offset, header, 0, BmdMessage.HeaderBytes);
            if (!BmdMessage.TryParse(header, out BmdMessage message)) return false;

            int textOffset = offset + message.TextOffset;
            if (textOffset > file.Length) return false;

            entries[i] = new Entry(i, offset, textOffset, message.SpeakerId,
                                   message.Name, kind != 0 || message.IsSelection);
        }

        // Dialogue stops at whichever comes first: the name array or the relocation table.
        // Both are file offsets past the last text buffer, and either one bounds the region
        // the writer may touch.
        int reloc      = ReadInt32(file, RelocTableOffset);
        int dialogEnd  = file.Length;
        if (speakerArray > 0)                       dialogEnd = Math.Min(dialogEnd, speakerArray);
        if (reloc > 0 && reloc <= file.Length)      dialogEnd = Math.Min(dialogEnd, reloc);

        archive = new BmdArchive(file, entries, speakerArray, speakerCount, dialogEnd);
        return true;
    }

    /// <summary>True when the file declares its addresses have been relocated in place.</summary>
    internal static bool IsRelocated(byte[] file) =>
        file is not null && file.Length > IsRelocatedOffset + 1 &&
        (file[IsRelocatedOffset] | (file[IsRelocatedOffset + 1] << 8)) != 0;

    /// <summary>
    /// The message whose text buffer contains <paramref name="fileOffset"/>, or false.
    ///
    /// Matching is by containment against the next message's text rather than by the
    /// declared buffer size, because the runs being matched are what the pool scanner
    /// found — printable ASCII somewhere inside a buffer that starts with control codes.
    /// The distance from the buffer start to the first readable character varies per
    /// message, which is the whole reason searching backwards from a run never worked.
    /// </summary>
    internal bool TryFindByTextOffset(int fileOffset, out Entry entry)
    {
        entry = default;
        if (fileOffset < 0 || fileOffset >= DialogueEnd) return false;

        bool found = false;
        foreach (Entry candidate in _entries)
        {
            if (candidate.TextOffset > fileOffset) continue;
            if (found && candidate.TextOffset <= entry.TextOffset) continue;

            entry = candidate;
            found = true;
        }
        return found;
    }

    /// <summary>
    /// Resolve a speaker id to its name, following the self-relative address in the array.
    ///
    /// Names carry the format's inline control bytes; those are dropped rather than
    /// rendered, and a name left empty by that is treated as no name at all.
    /// </summary>
    internal bool TryGetSpeakerName(int speakerId, out string name)
    {
        name = string.Empty;
        if (!HasSpeakerTable) return false;
        if (speakerId < 0 || speakerId >= SpeakerCount) return false;

        int field = _speakerArray + 4 * speakerId;
        if (field + 4 > _file.Length) return false;

        int start = field + ReadInt32(_file, field);
        if (start < 0 || start >= _file.Length) return false;

        var chars = new System.Text.StringBuilder(MaxNameBytes);
        for (int i = start; i < _file.Length && i - start < MaxNameBytes; i++)
        {
            byte b = _file[i];
            if (b == 0) break;
            if (b >= 0x20 && b <= 0x7E) chars.Append((char)b);
        }

        name = chars.ToString().Trim();
        return name.Length > 0;
    }

    private static int ReadInt32(byte[] buf, int index) =>
        buf[index] | (buf[index + 1] << 8) | (buf[index + 2] << 16) | (buf[index + 3] << 24);
}
