using System;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// The MSG1 file wrapped around a set of <see cref="BmdMessage"/> records.
///
/// A message header carries a <see cref="BmdMessage.SpeakerId"/>, which is an index and
/// therefore useless on its own. The table it indexes lives in the file header, ahead of
/// the dialogue data:
///
/// <code>
/// offset  size  field
/// 0x00       1  FileType
/// 0x01       1  Format
/// 0x02       2  UserId
/// 0x04       4  FileSize
/// 0x08       4  Magic                "MSG1"
/// 0x0C       4  ExtSize
/// 0x10       4  RelocationTable
/// 0x14       4  RelocationTableSize
/// 0x18       4  DialogCount
/// 0x1C       2  IsRelocated
/// 0x1E       2  Version
/// 0x20     8*n  DialogEntry[n]        { int Kind; int Address; }   n = DialogCount
///  ...      16  SpeakerTable          { int ArrayAddress; int Count; int ExtAddress; int _ }
/// </code>
///
/// Stored addresses are relative to a base rather than to the file start, and which base
/// is a detail this mod has no way to look up. So it does not assume one: both candidates
/// are tried and the one whose dialogue entries land on parseable message headers wins.
/// That check is cheap, it is decisive, and it is the same discipline as
/// <see cref="BmdMessage.TryFindHeader"/> — accept the interpretation that predicts data
/// we can already verify.
///
/// See learning.md Ch. 76.
/// </summary>
internal readonly struct BmdArchive
{
    /// Distance from the file start to the "MSG1" magic.
    private const int MagicOffset = 0x08;

    private const int FileSizeOffset    = 0x04;
    private const int DialogCountOffset = 0x18;
    private const int DialogTableOffset = 0x20;
    private const int DialogEntryBytes  = 8;
    private const int SpeakerTableBytes = 16;

    /// Sanity bounds. Both fields are read from memory that may have been recycled, and
    /// both size a table walk, so an implausible value has to fail here.
    private const int MaxDialogs  = 4096;
    private const int MaxSpeakers = 1024;

    /// Bounds on the declared file size, which decides how much is copied per read.
    private const int MinFileBytes = 0x40;
    private const int MaxFileBytes = 16 * 1024 * 1024;

    /// <summary>Longest speaker name accepted, in bytes, before the string is truncated.</summary>
    private const int MaxNameBytes = 64;

    /// <summary>Index of file byte 0 within the window this was parsed from.</summary>
    internal int FileStart { get; }

    /// <summary>Add this to a stored address to get a window index.</summary>
    internal int AddressBase { get; }

    internal int DialogCount  { get; }
    internal int SpeakerCount { get; }

    /// <summary>Window index of the int32 array of speaker-name addresses.</summary>
    internal int SpeakerArrayIndex { get; }

    private BmdArchive(
        int fileStart, int addressBase, int dialogCount, int speakerCount, int speakerArrayIndex)
    {
        FileStart         = fileStart;
        AddressBase       = addressBase;
        DialogCount       = dialogCount;
        SpeakerCount      = speakerCount;
        SpeakerArrayIndex = speakerArrayIndex;
    }

    internal bool HasSpeakerTable => SpeakerCount > 0 && SpeakerArrayIndex > 0;

    /// <summary>
    /// Locate the "MSG1" magic at or before <paramref name="limit"/>, searching backwards.
    ///
    /// Backwards because the caller knows where a record is, not where its file begins,
    /// and the header always precedes the dialogue data. Nearest match wins: several BMDs
    /// can be resident at once and the one immediately behind a record is its own.
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
    /// The caller needs this before it can read the archive, because the speaker names sit
    /// past the end of every message — a window sized to reach the records is nowhere near
    /// long enough to reach the table that names them.
    ///
    /// Bounded because the value decides how much memory gets copied out of the game on a
    /// poll tick. Sixteen megabytes is far past any scene script and far short of anything
    /// that would be felt.
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
    /// Parse the file header found at <paramref name="magicIndex"/>, choosing the address
    /// base by which one makes the dialogue table resolve.
    /// </summary>
    internal static bool TryParse(byte[] window, int magicIndex, out BmdArchive archive)
    {
        archive = default;
        if (window is null) return false;

        int fileStart = magicIndex - MagicOffset;
        if (fileStart < 0) return false;
        if (fileStart + DialogTableOffset + DialogEntryBytes > window.Length) return false;

        int dialogCount = ReadInt32(window, fileStart + DialogCountOffset);
        if (dialogCount < 1 || dialogCount > MaxDialogs) return false;

        int speakerTable = fileStart + DialogTableOffset + DialogEntryBytes * dialogCount;
        if (speakerTable < 0 || speakerTable + SpeakerTableBytes > window.Length) return false;

        int speakerArrayAddress = ReadInt32(window, speakerTable);
        int speakerCount        = ReadInt32(window, speakerTable + 4);
        if (speakerCount < 0 || speakerCount > MaxSpeakers) return false;

        // The two bases the format is known to use. Trying both and keeping the one that
        // resolves is deliberate: guessing wrong here would not fail loudly, it would
        // produce plausible names for the wrong speakers.
        int[] bases = { fileStart + 0x10, fileStart };
        foreach (int addressBase in bases)
        {
            if (!DialogTableResolves(window, fileStart, addressBase, dialogCount)) continue;

            int speakerArrayIndex = speakerCount > 0 ? addressBase + speakerArrayAddress : 0;
            if (speakerArrayIndex < 0 ||
                speakerArrayIndex + 4 * speakerCount > window.Length) speakerArrayIndex = 0;

            archive = new BmdArchive(
                fileStart, addressBase, dialogCount, speakerCount, speakerArrayIndex);
            return true;
        }
        return false;
    }

    /// <summary>
    /// True when the dialogue entries point at things that parse as message headers.
    ///
    /// Only the first few are checked. The table is in script order and a wrong base is
    /// wrong for all of them, so testing more buys nothing and costs a walk over a table
    /// that can hold four thousand entries.
    /// </summary>
    private static bool DialogTableResolves(
        byte[] window, int fileStart, int addressBase, int dialogCount)
    {
        int examined = 0;

        for (int i = 0; i < dialogCount && examined < 4; i++)
        {
            int entry = fileStart + DialogTableOffset + DialogEntryBytes * i;
            if (entry + DialogEntryBytes > window.Length) return false;

            int address = ReadInt32(window, entry + 4);
            int index   = addressBase + address;
            if (index < 0 || index + BmdMessage.HeaderBytes > window.Length) return false;

            var slice = new byte[BmdMessage.HeaderBytes];
            Array.Copy(window, index, slice, 0, BmdMessage.HeaderBytes);
            if (!BmdMessage.TryParse(slice, out _)) return false;

            examined++;
        }
        return examined > 0;
    }

    /// <summary>
    /// Resolve a speaker id to the name string the table points at, or return false.
    ///
    /// Names may carry the format's inline control bytes; those are dropped rather than
    /// rendered, and a name left empty by that is treated as no name at all.
    /// </summary>
    internal bool TryGetSpeakerName(byte[] window, int speakerId, out string name)
    {
        name = string.Empty;
        if (!HasSpeakerTable) return false;
        if (speakerId < 0 || speakerId >= SpeakerCount) return false;

        int index = SpeakerArrayIndex + 4 * speakerId;
        if (index + 4 > window.Length) return false;

        int start = AddressBase + ReadInt32(window, index);
        if (start < 0 || start >= window.Length) return false;

        var chars = new System.Text.StringBuilder(MaxNameBytes);
        for (int i = start; i < window.Length && i - start < MaxNameBytes; i++)
        {
            byte b = window[i];
            if (b == 0) break;
            if (b >= 0x20 && b <= 0x7E) chars.Append((char)b);
        }

        name = chars.ToString().Trim();
        return name.Length > 0;
    }

    private static int ReadInt32(byte[] buf, int index) =>
        buf[index] | (buf[index + 1] << 8) | (buf[index + 2] << 16) | (buf[index + 3] << 24);
}
