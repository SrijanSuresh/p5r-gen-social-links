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
}
