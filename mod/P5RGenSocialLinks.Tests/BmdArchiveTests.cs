using System;
using System.Collections.Generic;
using System.Text;
using P5RGenSocialLinks.Memory;
using Xunit;

namespace P5RGenSocialLinks.Tests;

/// <summary>
/// Tests for the MSG1 file header and its speaker table (learning.md Ch. 76).
///
/// The fixture builds a whole miniature archive rather than a hand-written byte array,
/// because what is being tested is agreement between three structures — the dialogue
/// table, the speaker table and the message headers — and a fixture that cannot express
/// that agreement cannot test the check that relies on it.
/// </summary>
public class BmdArchiveTests
{
    /// <summary>
    /// The address base the real format uses: stored addresses are relative to file
    /// start + 0x10, not to the file start.
    /// </summary>
    private const int RealAddressBase = 0x10;

    private sealed record Message(string Name, int Pages, int Speaker, string Text);

    /// <summary>
    /// Build a window containing <paramref name="lead"/> bytes of unrelated memory
    /// followed by one MSG1 file. Returns the window and the index of the magic.
    /// </summary>
    private static (byte[] Window, int MagicIndex) Archive(
        IReadOnlyList<Message> messages,
        IReadOnlyList<string> speakers,
        int lead = 0,
        int addressBase = RealAddressBase)
    {
        var body = new List<byte>();

        // --- fixed header --------------------------------------------------------
        body.AddRange(new byte[0x20]);
        WriteAscii(body, 0x08, "MSG1");
        WriteInt32(body, 0x18, messages.Count);

        // --- dialogue table, then speaker table ---------------------------------
        int dialogTable  = body.Count;
        body.AddRange(new byte[8 * messages.Count]);
        int speakerTable = body.Count;
        body.AddRange(new byte[16]);

        // --- message records -----------------------------------------------------
        var messageIndices = new List<int>();
        foreach (Message m in messages)
        {
            messageIndices.Add(body.Count);

            var name = new byte[BmdMessage.NameBytes];
            Encoding.ASCII.GetBytes(m.Name).CopyTo(name, 0);
            body.AddRange(name);
            body.Add((byte)(m.Pages & 0xFF));   body.Add((byte)(m.Pages >> 8));
            body.Add((byte)(m.Speaker & 0xFF)); body.Add((byte)(m.Speaker >> 8));
            body.AddRange(new byte[4 * m.Pages]);       // page table
            body.AddRange(new byte[4]);                 // text buffer size
            body.AddRange(Encoding.ASCII.GetBytes(m.Text));
            body.Add(0);
        }

        // --- speaker name strings, then the array of addresses to them -----------
        var nameIndices = new List<int>();
        foreach (string s in speakers)
        {
            nameIndices.Add(body.Count);
            body.AddRange(Encoding.ASCII.GetBytes(s));
            body.Add(0);
        }

        int speakerArray = body.Count;
        body.AddRange(new byte[4 * speakers.Count]);

        // --- back-fill every address --------------------------------------------
        for (int i = 0; i < messages.Count; i++)
        {
            WriteInt32(body, dialogTable + 8 * i, 0);                             // Kind
            WriteInt32(body, dialogTable + 8 * i + 4, messageIndices[i] - addressBase);
        }
        WriteInt32(body, speakerTable,     speakerArray - addressBase);
        WriteInt32(body, speakerTable + 4, speakers.Count);
        for (int i = 0; i < speakers.Count; i++)
            WriteInt32(body, speakerArray + 4 * i, nameIndices[i] - addressBase);

        var window = new byte[lead + body.Count];
        for (int i = 0; i < lead; i++) window[i] = 0xCC;
        body.CopyTo(window, lead);

        return (window, lead + 0x08);
    }

    private static void WriteInt32(List<byte> buf, int index, int value)
    {
        buf[index]     = (byte)(value & 0xFF);
        buf[index + 1] = (byte)((value >> 8) & 0xFF);
        buf[index + 2] = (byte)((value >> 16) & 0xFF);
        buf[index + 3] = (byte)((value >> 24) & 0xFF);
    }

    private static void WriteAscii(List<byte> buf, int index, string text)
    {
        for (int i = 0; i < text.Length; i++) buf[index + i] = (byte)text[i];
    }

    private static readonly Message[] Scene =
    {
        new("MSG_001_0_0", 2, 0,      "Yo, Joker!"),
        new("MSG_002_0_0", 1, 1,      "Good afternoon."),
        new("MSG_003_0_0", 1, 0xFFFF, "The room falls quiet."),
    };

    private static readonly string[] Speakers = { "Ryuji", "Tae Takemi" };

    [Fact]
    public void Parses_the_dialogue_and_speaker_counts()
    {
        (byte[] window, int magic) = Archive(Scene, Speakers, lead: 96);

        Assert.True(BmdArchive.TryParse(window, magic, out BmdArchive archive));
        Assert.Equal(3, archive.DialogCount);
        Assert.Equal(2, archive.SpeakerCount);
        Assert.True(archive.HasSpeakerTable);
    }

    [Fact]
    public void Recovers_the_address_base_from_the_data()
    {
        // The point of the whole struct: the base is not assumed, it is the candidate that
        // makes the dialogue entries land on parseable message headers.
        (byte[] window, int magic) = Archive(Scene, Speakers, lead: 96);
        int fileStart = magic - 0x08;

        Assert.True(BmdArchive.TryParse(window, magic, out BmdArchive archive));
        Assert.Equal(fileStart + RealAddressBase, archive.AddressBase);
    }

    [Fact]
    public void Resolves_speaker_names()
    {
        (byte[] window, int magic) = Archive(Scene, Speakers, lead: 96);
        Assert.True(BmdArchive.TryParse(window, magic, out BmdArchive archive));

        Assert.True(archive.TryGetSpeakerName(window, 0, out string first));
        Assert.Equal("Ryuji", first);

        Assert.True(archive.TryGetSpeakerName(window, 1, out string second));
        Assert.Equal("Tae Takemi", second);
    }

    [Fact]
    public void Rejects_a_speaker_id_outside_the_table()
    {
        (byte[] window, int magic) = Archive(Scene, Speakers, lead: 96);
        Assert.True(BmdArchive.TryParse(window, magic, out BmdArchive archive));

        Assert.False(archive.TryGetSpeakerName(window, 2, out _));
        Assert.False(archive.TryGetSpeakerName(window, -1, out _));

        // Narration's sentinel is not an index and must never be treated as one.
        Assert.False(archive.TryGetSpeakerName(window, BmdMessage.NoSpeaker, out _));
    }

    [Fact]
    public void Drops_control_bytes_from_a_name()
    {
        // P5R name strings carry inline formatting. Rendering those as characters would
        // put line noise in the prompt, and the prompt is what the model reasons about.
        (byte[] window, int magic) = Archive(
            new[] { Scene[0] }, new[] { "Ry\u0001uji\u000B" }, lead: 32);

        Assert.True(BmdArchive.TryParse(window, magic, out BmdArchive archive));
        Assert.True(archive.TryGetSpeakerName(window, 0, out string name));
        Assert.Equal("Ryuji", name);
    }

    [Fact]
    public void Finds_the_magic_by_searching_backwards()
    {
        (byte[] window, int magic) = Archive(Scene, Speakers, lead: 200);

        Assert.True(BmdArchive.TryFindMagicBefore(window, window.Length - 1, out int found));
        Assert.Equal(magic, found);
    }

    [Fact]
    public void Finds_the_nearest_magic_when_two_files_are_resident()
    {
        // Several BMDs can be in memory at once. A record belongs to the file immediately
        // behind it, so the search has to stop at the first match going backwards rather
        // than run to the start of the region.
        (byte[] first, _)      = Archive(Scene, Speakers);
        (byte[] second, int m) = Archive(Scene, Speakers);

        var window = new byte[first.Length + second.Length];
        first.CopyTo(window, 0);
        second.CopyTo(window, first.Length);

        Assert.True(BmdArchive.TryFindMagicBefore(window, window.Length - 1, out int found));
        Assert.Equal(first.Length + m, found);
    }

    [Fact]
    public void Rejects_a_magic_with_nothing_behind_it()
    {
        var window = Encoding.ASCII.GetBytes("MSG1padding-but-no-header");
        Assert.False(BmdArchive.TryParse(window, 0, out _));
    }

    [Fact]
    public void Rejects_a_dialogue_table_that_points_at_nothing()
    {
        (byte[] window, int magic) = Archive(Scene, Speakers, lead: 96);

        // Corrupt the first entry's address. A base that cannot resolve the table is not
        // the file's base, and a struct that accepted it would attribute lines to whatever
        // the stray bytes happened to name.
        int entry = magic - 0x08 + 0x20 + 4;
        window[entry] ^= 0x5A;

        Assert.False(BmdArchive.TryParse(window, magic, out _));
    }

    [Fact]
    public void Rejects_an_implausible_dialogue_count()
    {
        (byte[] window, int magic) = Archive(Scene, Speakers, lead: 96);
        int fileStart = magic - 0x08;

        window[fileStart + 0x18] = 0x00;
        window[fileStart + 0x19] = 0x00;
        window[fileStart + 0x1A] = 0x00;
        window[fileStart + 0x1B] = 0x00;

        Assert.False(BmdArchive.TryParse(window, magic, out _));
    }

    [Fact]
    public void Handles_a_file_with_no_speakers_at_all()
    {
        (byte[] window, int magic) = Archive(new[] { Scene[2] }, Array.Empty<string>(), lead: 16);

        Assert.True(BmdArchive.TryParse(window, magic, out BmdArchive archive));
        Assert.Equal(0, archive.SpeakerCount);
        Assert.False(archive.HasSpeakerTable);
        Assert.False(archive.TryGetSpeakerName(window, 0, out _));
    }

    [Fact]
    public void Does_not_run_off_the_ends_of_the_window()
    {
        Assert.False(BmdArchive.TryParse(null!, 8, out _));
        Assert.False(BmdArchive.TryParse(new byte[64], 4, out _));      // magic before file start
        Assert.False(BmdArchive.TryParse(new byte[16], 8, out _));      // truncated header
        Assert.False(BmdArchive.TryFindMagicBefore(null!, 4, out _));
        Assert.False(BmdArchive.TryFindMagicBefore(new byte[2], 1, out _));
    }
}
