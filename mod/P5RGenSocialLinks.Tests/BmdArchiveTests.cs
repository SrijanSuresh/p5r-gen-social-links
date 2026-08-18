using System;
using System.Collections.Generic;
using System.Text;
using P5RGenSocialLinks.Memory;
using Xunit;

namespace P5RGenSocialLinks.Tests;

/// <summary>
/// Tests for the MSG1 file header, its dialogue table and its speaker table
/// (learning.md Ch. 79).
///
/// The fixture builds a whole miniature archive, self-relative addresses and all, because
/// what is under test is agreement between three structures. It also puts a control-code
/// prefix in front of every text buffer, which is not decoration: that prefix is why
/// searching backwards from a run of printable ASCII never found a header, and a fixture
/// without it would let the same mistake pass unnoticed.
/// </summary>
public class BmdArchiveTests
{
    private sealed record Message(string Name, int Pages, int Speaker, string Text);

    /// Bytes the game puts between a text buffer's start and its first readable character.
    private static readonly byte[] ControlPrefix =
        { 0xF2, 0x05, 0xFF, 0xFF, 0xF1, 0x41, 0xF7, 0x61 };

    /// <summary>
    /// Build a complete MSG1 file. Returns the bytes and, for each message, the file offset
    /// of its first printable character — what the pool scanner would report.
    /// </summary>
    private static (byte[] File, int[] AsciiOffsets) Archive(
        IReadOnlyList<Message> messages, IReadOnlyList<string> speakers)
    {
        var f = new List<byte>();

        f.AddRange(new byte[0x20]);
        WriteAscii(f, 0x08, "MSG1");
        WriteInt32(f, 0x18, messages.Count);
        WriteInt16(f, 0x1C, 1);                    // IsRelocated
        WriteInt16(f, 0x1E, 2);                    // Version

        int dialogTable = f.Count;
        f.AddRange(new byte[8 * messages.Count]);
        int speakerTable = f.Count;
        f.AddRange(new byte[16]);

        var messageOffsets = new List<int>();
        var asciiOffsets   = new List<int>();

        foreach (Message m in messages)
        {
            messageOffsets.Add(f.Count);

            var name = new byte[BmdMessage.NameBytes];
            Encoding.ASCII.GetBytes(m.Name).CopyTo(name, 0);
            f.AddRange(name);
            AppendInt16(f, m.Pages);
            AppendInt16(f, m.Speaker);

            int pageTable = f.Count;
            f.AddRange(new byte[4 * m.Pages]);
            f.AddRange(new byte[4]);               // TextBufferSize, unread by the parser

            int textStart = f.Count;
            // Page 0 starts at the buffer; the field is self-relative like everything else.
            WriteInt32(f, pageTable, textStart - pageTable);

            f.AddRange(ControlPrefix);
            asciiOffsets.Add(f.Count);
            f.AddRange(Encoding.ASCII.GetBytes(m.Text));
            f.Add(0x0A);
        }

        int speakerArray = f.Count;
        f.AddRange(new byte[4 * speakers.Count]);

        var nameOffsets = new List<int>();
        foreach (string s in speakers)
        {
            nameOffsets.Add(f.Count);
            f.AddRange(Encoding.ASCII.GetBytes(s));
            f.Add(0);
        }

        int reloc = f.Count;
        f.AddRange(new byte[4]);                   // a stand-in relocation table

        // --- self-relative back-fill --------------------------------------------
        for (int i = 0; i < messages.Count; i++)
        {
            int field = dialogTable + 8 * i + 4;
            WriteInt32(f, field - 4, 0);                          // Kind = message
            WriteInt32(f, field, messageOffsets[i] - field);
        }
        WriteInt32(f, speakerTable,     speakerArray - speakerTable);
        WriteInt32(f, speakerTable + 4, speakers.Count);
        for (int i = 0; i < speakers.Count; i++)
            WriteInt32(f, speakerArray + 4 * i, nameOffsets[i] - (speakerArray + 4 * i));

        WriteInt32(f, 0x10, reloc);                // plain file offset, not self-relative
        WriteInt32(f, 0x14, f.Count - reloc);
        WriteInt32(f, 0x04, f.Count);              // FileSize

        return (f.ToArray(), asciiOffsets.ToArray());
    }

    private static void WriteInt32(List<byte> buf, int index, int value)
    {
        buf[index]     = (byte)(value & 0xFF);
        buf[index + 1] = (byte)((value >> 8) & 0xFF);
        buf[index + 2] = (byte)((value >> 16) & 0xFF);
        buf[index + 3] = (byte)((value >> 24) & 0xFF);
    }

    private static void WriteInt16(List<byte> buf, int index, int value)
    {
        buf[index]     = (byte)(value & 0xFF);
        buf[index + 1] = (byte)((value >> 8) & 0xFF);
    }

    private static void AppendInt16(List<byte> buf, int value)
    {
        buf.Add((byte)(value & 0xFF));
        buf.Add((byte)((value >> 8) & 0xFF));
    }

    private static void WriteAscii(List<byte> buf, int index, string text)
    {
        for (int i = 0; i < text.Length; i++) buf[index + i] = (byte)text[i];
    }

    // The shape of the scene that produced the dump: Takemi, the girl's father, and a
    // system line with no speaker at all.
    private static readonly Message[] Scene =
    {
        new("MSG_000_0_0", 1, 0,      "...So, why come here?"),
        new("MSG_001_0_0", 2, 1,      "She has a fever that won't go away."),
        new("MSG_002_0_0", 1, 0,      "Cash only, it's safer that way."),
        new("MSG_003_0_0", 3, 0xFFFF, "You've unlocked the Death Confidant."),
    };

    private static readonly string[] Speakers = { "Takemi", "Girl's Father", "Sick Girl" };

    private static string[] Names(BmdArchive archive)
    {
        var names = new List<string>();
        foreach (BmdArchive.Entry e in archive.Entries) names.Add(e.Name);
        return names.ToArray();
    }

    // --- header ---------------------------------------------------------------------

    [Fact]
    public void Parses_the_dialogue_and_speaker_counts()
    {
        (byte[] file, _) = Archive(Scene, Speakers);

        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));
        Assert.Equal(4, archive.DialogCount);
        Assert.Equal(3, archive.SpeakerCount);
        Assert.True(archive.HasSpeakerTable);
        Assert.True(BmdArchive.IsRelocated(file));
    }

    [Fact]
    public void Enumerates_every_message_in_script_order()
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        Assert.Equal(
            new[] { "MSG_000_0_0", "MSG_001_0_0", "MSG_002_0_0", "MSG_003_0_0" },
            Names(archive));

        // Ordered and non-overlapping, which is what makes containment matching sound.
        for (int i = 1; i < archive.DialogCount; i++)
            Assert.True(archive.Entries[i].Offset > archive.Entries[i - 1].Offset);
    }

    [Fact]
    public void Reads_the_speaker_id_off_each_message()
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        Assert.Equal(0, archive.Entries[0].SpeakerId);
        Assert.Equal(1, archive.Entries[1].SpeakerId);
        Assert.Equal(BmdMessage.NoSpeaker, archive.Entries[3].SpeakerId);
    }

    // --- self-relative addressing ---------------------------------------------------

    [Fact]
    public void Resolves_speaker_names_through_self_relative_addresses()
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        Assert.True(archive.TryGetSpeakerName(0, out string a));
        Assert.Equal("Takemi", a);
        Assert.True(archive.TryGetSpeakerName(1, out string b));
        Assert.Equal("Girl's Father", b);
        Assert.True(archive.TryGetSpeakerName(2, out string c));
        Assert.Equal("Sick Girl", c);
    }

    [Fact]
    public void An_address_read_as_file_relative_resolves_to_the_wrong_place()
    {
        // The bug this rewrite fixes, pinned. Under the old rule a dialogue entry's value
        // was treated as an offset from the file start; the format means an offset from the
        // field's own position, which for the first entry differs by 0x24.
        (byte[] file, _) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        int field  = 0x20 + 4;
        int stored = file[field] | (file[field + 1] << 8) |
                     (file[field + 2] << 16) | (file[field + 3] << 24);

        Assert.NotEqual(stored, archive.Entries[0].Offset);
        Assert.Equal(field + stored, archive.Entries[0].Offset);
    }

    // --- matching a scanner's text run ----------------------------------------------

    [Fact]
    public void Matches_a_run_of_ascii_to_the_message_that_contains_it()
    {
        // The scanner reports where printable text starts, which is past a control-code
        // prefix whose length varies per message. Containment handles that; searching
        // backwards from the run does not.
        (byte[] file, int[] ascii) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        for (int i = 0; i < ascii.Length; i++)
        {
            Assert.True(archive.TryFindByTextOffset(ascii[i], out BmdArchive.Entry entry));
            Assert.Equal(Scene[i].Name, entry.Name);
        }
    }

    [Fact]
    public void Matches_a_run_partway_through_a_message()
    {
        // Second and later rows of a bubble are their own runs to the scanner.
        (byte[] file, int[] ascii) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        Assert.True(archive.TryFindByTextOffset(ascii[1] + 5, out BmdArchive.Entry entry));
        Assert.Equal("MSG_001_0_0", entry.Name);
    }

    [Fact]
    public void Refuses_to_match_anything_past_the_end_of_the_dialogue()
    {
        // "Girl's Father" is thirteen printable characters inside an armed region, and the
        // pool scanner finds it. Attributing it to the last message would also make it a
        // write target.
        (byte[] file, _) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        Assert.False(archive.TryFindByTextOffset(archive.DialogueEnd + 4, out _));
        Assert.False(archive.TryFindByTextOffset(-1, out _));
    }

    [Fact]
    public void Dialogue_ends_before_the_speaker_names()
    {
        (byte[] file, int[] ascii) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        Assert.True(archive.DialogueEnd > ascii[ascii.Length - 1]);
        Assert.True(archive.DialogueEnd < file.Length);
    }

    // --- rejection ------------------------------------------------------------------

    [Fact]
    public void Rejects_a_dialogue_table_that_points_at_nothing()
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        file[0x24] ^= 0x5A;

        Assert.False(BmdArchive.TryParse(file, out _));
    }

    [Fact]
    public void Rejects_a_file_whose_later_entries_do_not_resolve()
    {
        // All or nothing. Half a scene attributed under a wrong rule is worse than none,
        // because it looks like it worked.
        (byte[] file, _) = Archive(Scene, Speakers);
        file[0x20 + 8 * 2 + 4] ^= 0x33;

        Assert.False(BmdArchive.TryParse(file, out _));
    }

    [Fact]
    public void Rejects_an_implausible_dialogue_count()
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        file[0x18] = file[0x19] = file[0x1A] = file[0x1B] = 0;

        Assert.False(BmdArchive.TryParse(file, out _));
    }

    [Fact]
    public void Rejects_a_buffer_that_is_not_an_MSG1_file()
    {
        Assert.False(BmdArchive.TryParse(null!, out _));
        Assert.False(BmdArchive.TryParse(new byte[16], out _));
        Assert.False(BmdArchive.TryParse(new byte[256], out _));   // zeroed: no magic
    }

    [Fact]
    public void Rejects_a_speaker_id_outside_the_table()
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));

        Assert.False(archive.TryGetSpeakerName(3, out _));
        Assert.False(archive.TryGetSpeakerName(-1, out _));

        // Narration's sentinel is not an index and must never be treated as one.
        Assert.False(archive.TryGetSpeakerName(BmdMessage.NoSpeaker, out _));
    }

    [Fact]
    public void Handles_a_file_with_no_speakers_at_all()
    {
        // The rank-up notification file: messages, and no speaker table at all.
        (byte[] file, _) = Archive(new[] { Scene[3] }, Array.Empty<string>());

        Assert.True(BmdArchive.TryParse(file, out BmdArchive archive));
        Assert.Equal(0, archive.SpeakerCount);
        Assert.False(archive.HasSpeakerTable);
        Assert.False(archive.TryGetSpeakerName(0, out _));
    }

    // --- locating the file ----------------------------------------------------------

    [Fact]
    public void Finds_the_magic_by_searching_backwards()
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        var window = new byte[200 + file.Length];
        file.CopyTo(window, 200);

        Assert.True(BmdArchive.TryFindMagicBefore(window, window.Length - 1, out int found));
        Assert.Equal(200 + 0x08, found);
    }

    [Fact]
    public void Finds_the_nearest_magic_when_two_files_are_resident()
    {
        (byte[] first, _)  = Archive(Scene, Speakers);
        (byte[] second, _) = Archive(Scene, Speakers);

        var window = new byte[first.Length + second.Length];
        first.CopyTo(window, 0);
        second.CopyTo(window, first.Length);

        Assert.True(BmdArchive.TryFindMagicBefore(window, window.Length - 1, out int found));
        Assert.Equal(first.Length + 0x08, found);
    }

    [Fact]
    public void Reads_the_declared_file_size()
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        var window = new byte[96 + file.Length];
        file.CopyTo(window, 96);

        Assert.True(BmdArchive.TryReadFileSize(window, 96 + 0x08, out int size));
        Assert.Equal(file.Length, size);
    }

    [Theory]
    [InlineData(0)]
    [InlineData(0x20)]
    [InlineData(int.MaxValue)]
    [InlineData(-4096)]
    public void Rejects_an_implausible_file_size(int declared)
    {
        (byte[] file, _) = Archive(Scene, Speakers);
        var window = new byte[96 + file.Length];
        file.CopyTo(window, 96);

        window[96 + 4] = (byte)(declared & 0xFF);
        window[96 + 5] = (byte)((declared >> 8) & 0xFF);
        window[96 + 6] = (byte)((declared >> 16) & 0xFF);
        window[96 + 7] = (byte)((declared >> 24) & 0xFF);

        Assert.False(BmdArchive.TryReadFileSize(window, 96 + 0x08, out _));
    }

    [Fact]
    public void Does_not_run_off_the_ends_of_the_window()
    {
        Assert.False(BmdArchive.TryFindMagicBefore(null!, 4, out _));
        Assert.False(BmdArchive.TryFindMagicBefore(new byte[2], 1, out _));
        Assert.False(BmdArchive.TryReadFileSize(new byte[4], 8, out _));
        Assert.False(BmdArchive.TryReadFileSize(null!, 8, out _));
    }
}
