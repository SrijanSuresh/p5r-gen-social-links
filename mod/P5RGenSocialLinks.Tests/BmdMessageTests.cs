using System;
using System.Text;
using P5RGenSocialLinks.Memory;
using Xunit;

namespace P5RGenSocialLinks.Tests;

/// <summary>
/// Layout tests for the MSG1 dialogue header (learning.md Ch. 75).
///
/// These are byte-level rather than behavioural because that is where the risk is. The
/// struct is read out of live game memory that may have been freed a millisecond earlier,
/// so the parser is as much a rejection filter as a decoder, and the cases worth pinning
/// are the ones where it must say no.
/// </summary>
public class BmdMessageTests
{
    /// <summary>Build a header: 24-byte NUL-padded name, int16 page count, uint16 speaker.</summary>
    private static byte[] Header(string name, int pages, int speaker)
    {
        var buf = new byte[BmdMessage.HeaderBytes];
        Encoding.ASCII.GetBytes(name).CopyTo(buf, 0);
        buf[BmdMessage.PageCountOffset]     = (byte)(pages & 0xFF);
        buf[BmdMessage.PageCountOffset + 1] = (byte)(pages >> 8);
        buf[BmdMessage.SpeakerIdOffset]     = (byte)(speaker & 0xFF);
        buf[BmdMessage.SpeakerIdOffset + 1] = (byte)(speaker >> 8);
        return buf;
    }

    [Fact]
    public void Parses_name_page_count_and_speaker()
    {
        Assert.True(BmdMessage.TryParse(Header("MSG_001_5_0", 2, 7), out BmdMessage msg));

        Assert.Equal("MSG_001_5_0", msg.Name);
        Assert.Equal(2, msg.PageCount);
        Assert.Equal(7, msg.SpeakerId);
        Assert.True(msg.HasSpeaker);
    }

    /// <summary>
    /// The offset that ties the layout to reality: Ch. 69 measured a live record's first
    /// character at +0x28, and that is what a two-page message predicts.
    /// </summary>
    [Theory]
    [InlineData(1, 0x24)]
    [InlineData(2, 0x28)]
    [InlineData(3, 0x2C)]
    public void Text_offset_follows_the_page_table(int pages, int expected)
    {
        Assert.True(BmdMessage.TryParse(Header("MSG_001_5_0", pages, 0), out BmdMessage msg));
        Assert.Equal(expected, msg.TextOffset);
    }

    [Fact]
    public void Speaker_0xFFFF_means_narration()
    {
        Assert.True(BmdMessage.TryParse(Header("MSG_010_0_0", 1, 0xFFFF), out BmdMessage msg));
        Assert.False(msg.HasSpeaker);
        Assert.Equal(BmdMessage.NoSpeaker, msg.SpeakerId);
    }

    [Fact]
    public void Selection_records_are_flagged()
    {
        Assert.True(BmdMessage.TryParse(Header("SEL_003", 1, 0xFFFF), out BmdMessage sel));
        Assert.True(sel.IsSelection);

        Assert.True(BmdMessage.TryParse(Header("MSG_003", 1, 0xFFFF), out BmdMessage msg));
        Assert.False(msg.IsSelection);
    }

    [Fact]
    public void Rejects_a_name_field_that_is_not_NUL_padded()
    {
        // A freed buffer reads as plausible ASCII far more often than it reads as ASCII
        // followed by exactly the right run of zeroes. This is the cheap half of the
        // filter and it is why the padding check is strict.
        byte[] buf = Header("MSG_001_5_0", 1, 0);
        buf[20] = (byte)'X';

        Assert.False(BmdMessage.TryParse(buf, out _));
    }

    [Fact]
    public void Rejects_unprintable_bytes_inside_the_name()
    {
        byte[] buf = Header("MSG_001_5_0", 1, 0);
        buf[3] = 0x01;

        Assert.False(BmdMessage.TryParse(buf, out _));
    }

    [Fact]
    public void Rejects_an_empty_name()
    {
        Assert.False(BmdMessage.TryParse(new byte[BmdMessage.HeaderBytes], out _));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(-1)]
    [InlineData(1000)]
    public void Rejects_an_implausible_page_count(int pages)
    {
        // PageCount sizes a page-table walk. A garbage count from a freed record has to
        // fail here rather than downstream where it would become an offset.
        Assert.False(BmdMessage.TryParse(Header("MSG_001_5_0", pages & 0xFFFF, 0), out _));
    }

    [Fact]
    public void Rejects_a_short_buffer()
    {
        Assert.False(BmdMessage.TryParse(new byte[BmdMessage.HeaderBytes - 1], out _));
        Assert.False(BmdMessage.TryParse(null!, out _));
    }

    [Fact]
    public void Name_stops_at_the_first_NUL()
    {
        byte[] buf = Header("MSG_A", 1, 3);
        Assert.True(BmdMessage.TryParse(buf, out BmdMessage msg));
        Assert.Equal("MSG_A", msg.Name);
        Assert.Equal(5, msg.Name.Length);
    }

    // --- backwards search -----------------------------------------------------------

    /// <summary>
    /// A pool fragment: <paramref name="lead"/> bytes of whatever came before, then a
    /// complete record — header, page table, text size, text.
    /// Returns the buffer and the index its text starts at.
    /// </summary>
    private static (byte[] Pool, int TextIndex) Record(
        string name, int pages, int speaker, string text, int lead = 0)
    {
        byte[] header = Header(name, pages, speaker);
        int    span   = BmdMessage.PageTableOffset + 4 * pages + 4;

        var pool = new byte[lead + span + text.Length];
        for (int i = 0; i < lead; i++) pool[i] = (byte)'x';   // previous message's text
        header.CopyTo(pool, lead);
        // Page table and text size are left as zeroes: the search never reads them, it
        // only needs to know how many bytes they occupy.
        Encoding.ASCII.GetBytes(text).CopyTo(pool, lead + span);

        return (pool, lead + span);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(5)]
    public void Finds_the_header_behind_a_text_run(int pages)
    {
        (byte[] pool, int textIndex) = Record("MSG_002_0_0", pages, 12, "Hey, Joker.", lead: 40);

        Assert.True(BmdMessage.TryFindHeader(pool, textIndex, out BmdMessage msg, out int at));

        Assert.Equal("MSG_002_0_0", msg.Name);
        Assert.Equal(pages, msg.PageCount);
        Assert.Equal(12, msg.SpeakerId);
        Assert.Equal(textIndex - at, msg.TextOffset);   // the record is self-consistent
    }

    [Fact]
    public void Finds_nothing_when_the_bytes_behind_are_just_text()
    {
        var pool = Encoding.ASCII.GetBytes(new string('a', 200));

        Assert.False(BmdMessage.TryFindHeader(pool, 150, out _, out int at));
        Assert.Equal(-1, at);
    }

    [Fact]
    public void A_header_that_does_not_predict_its_own_position_is_rejected()
    {
        // Well-formed in isolation — name padded, page count sane — but it sits at the
        // distance a 2-page record would use while claiming to hold 4 pages. This is the
        // false positive the self-consistency check exists to kill, and without it the
        // mod would attribute a line to whoever the stray bytes named.
        (byte[] pool, int textIndex) = Record("MSG_002_0_0", 2, 12, "Hey, Joker.", lead: 40);
        pool[40 + BmdMessage.PageCountOffset] = 4;

        Assert.False(BmdMessage.TryFindHeader(pool, textIndex, out _, out _));
    }

    [Fact]
    public void Resolves_the_second_of_two_packed_records()
    {
        // The realistic shape: a record whose header is preceded not by padding but by the
        // tail of the previous message's text. Candidate positions are four bytes apart
        // while a header is thirty-two bytes long, so the windows overlap heavily and the
        // search has to land on the one that agrees with itself.
        (byte[] first,  int firstText)  = Record("MSG_001_0_0", 2, 8, "Yo, what's up?", lead: 16);
        (byte[] second, int secondText) = Record("MSG_002_0_0", 1, 14, "Nothing much.");

        var pool = new byte[first.Length + second.Length];
        first.CopyTo(pool, 0);
        second.CopyTo(pool, first.Length);

        Assert.True(BmdMessage.TryFindHeader(pool, firstText, out BmdMessage a, out _));
        Assert.Equal("MSG_001_0_0", a.Name);
        Assert.Equal(8, a.SpeakerId);

        Assert.True(BmdMessage.TryFindHeader(
            pool, first.Length + secondText, out BmdMessage b, out _));
        Assert.Equal("MSG_002_0_0", b.Name);
        Assert.Equal(14, b.SpeakerId);
    }

    [Fact]
    public void Does_not_run_off_the_front_of_the_buffer()
    {
        var pool = new byte[8];

        Assert.False(BmdMessage.TryFindHeader(pool, 4, out _, out _));
        Assert.False(BmdMessage.TryFindHeader(pool, -1, out _, out _));
        Assert.False(BmdMessage.TryFindHeader(null!, 4, out _, out _));
    }
}
