using System;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// The fixed header at the front of a BMD dialogue message, as the interpreter sees it.
///
/// Atlus ships dialogue in MSG1 (.BMD) archives, and a message inside one is a struct
/// rather than a string. The pointer the hook captures in RAX is the base of that struct,
/// not the base of its text — which is why <see cref="Mod.ReadRecordPreview"/> has always
/// had to skip forward looking for the first English-looking run.
///
/// <code>
/// offset  size  field
/// 0x00      24  Name                 NUL-padded ASCII, e.g. "MSG_001_5_0"
/// 0x18       2  PageCount            int16, speech bubbles in this message
/// 0x1A       2  SpeakerId            uint16, index into the file speaker table
/// 0x1C    4*n   PageStartAddresses   int32 each, n = PageCount
///  ...       4  TextBufferSize
///  ...       ?  TextBuffer
/// </code>
///
/// The layout predicts the one offset measured live: Ch. 69 found a record's first
/// character at +0x28, and 0x1C + 4*2 + 4 is 0x28 for a two-page message.
///
/// See learning.md Ch. 75.
/// </summary>
internal readonly struct BmdMessage
{
    /// <summary>Bytes that must be readable before <see cref="TryParse"/> can be called.</summary>
    internal const int HeaderBytes = 0x20;

    internal const int NameBytes       = 0x18;
    internal const int PageCountOffset = 0x18;
    internal const int SpeakerIdOffset = 0x1A;
    internal const int PageTableOffset = 0x1C;

    /// <summary>SpeakerId used by narration and by any message with nobody attached.</summary>
    internal const int NoSpeaker = 0xFFFF;

    /// <summary>
    /// A message with more pages than this is not a message. The cap exists because
    /// PageCount is read from game memory that may have been freed between the hook
    /// capturing the pointer and the poll tick dereferencing it, and a garbage count
    /// would otherwise size a page-table walk.
    /// </summary>
    private const int MaxPages = 64;

    internal string Name      { get; }
    internal int    PageCount { get; }
    internal int    SpeakerId { get; }

    private BmdMessage(string name, int pageCount, int speakerId)
    {
        Name      = name;
        PageCount = pageCount;
        SpeakerId = speakerId;
    }

    /// <summary>True when this message carries no speaker — narration, or a system prompt.</summary>
    internal bool HasSpeaker => SpeakerId != NoSpeaker;

    /// <summary>
    /// True when the name looks like a selection prompt rather than a line of dialogue.
    ///
    /// The interpreter serves both. A SEL record is the list of replies the player picks
    /// from, and rewriting one is worse than rewriting the wrong speaker: it changes what
    /// the buttons say without changing what they do.
    /// </summary>
    internal bool IsSelection =>
        Name.StartsWith("SEL", StringComparison.Ordinal);

    /// <summary>
    /// Byte offset of the text buffer within the record, derived from the page count.
    ///
    /// This is where the cursor starts for page 0, and it replaces the "scan forward for
    /// something printable" heuristic with arithmetic.
    /// </summary>
    internal int TextOffset => PageTableOffset + 4 * PageCount + 4;

    /// <summary>
    /// Parse the header out of <paramref name="buf"/>, or return false if it does not
    /// look like a message.
    ///
    /// Rejection is the point of this method. The hook fires for every string the
    /// interpreter touches, menu labels and item descriptions included, and the buffer
    /// may hold whatever the allocator put there after the message was freed. A parser
    /// that always succeeds would hand the rest of the mod a name made of line noise.
    /// </summary>
    internal static bool TryParse(byte[] buf, out BmdMessage message)
    {
        message = default;
        if (buf is null || buf.Length < HeaderBytes) return false;

        if (!TryReadName(buf, out string name)) return false;

        int pageCount = buf[PageCountOffset] | (buf[PageCountOffset + 1] << 8);
        int speakerId = buf[SpeakerIdOffset] | (buf[SpeakerIdOffset + 1] << 8);

        if (pageCount < 1 || pageCount > MaxPages) return false;

        message = new BmdMessage(name, pageCount, speakerId);
        return true;
    }

    /// <summary>
    /// Read the 24-byte name field: printable ASCII, then NUL padding to the end.
    ///
    /// Anything else means we are not looking at a message header. The check is strict on
    /// purpose — the padding has to be NUL rather than merely unprintable, because that
    /// is the one part of the struct whose contents are fully specified, and a strict
    /// test on a specified field is the cheapest way to reject a freed buffer.
    /// </summary>
    private static bool TryReadName(byte[] buf, out string name)
    {
        name = string.Empty;

        int len = 0;
        while (len < NameBytes && buf[len] != 0)
        {
            byte b = buf[len];
            if (b < 0x20 || b > 0x7E) return false;
            len++;
        }
        if (len == 0) return false;

        for (int i = len; i < NameBytes; i++)
            if (buf[i] != 0) return false;

        var chars = new char[len];
        for (int i = 0; i < len; i++) chars[i] = (char)buf[i];
        name = new string(chars);
        return true;
    }

    public override string ToString() =>
        $"{Name} pages={PageCount} speaker={(HasSpeaker ? SpeakerId.ToString() : "none")}";
}
