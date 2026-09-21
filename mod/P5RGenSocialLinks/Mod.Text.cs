using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Reloaded.Hooks.Definitions;
using Reloaded.Hooks.Definitions.X64;
using Reloaded.Mod.Interfaces;
using Reloaded.Mod.Interfaces.Internal;
using P5RGenSocialLinks.Memory;
using P5RGenSocialLinks.Server;

namespace P5RGenSocialLinks;

/// <summary>
/// Deciding whether a run of bytes is English dialogue.
///
/// These are heuristics, and the project's history is largely the history of them being
/// wrong: scoring regions for sentence punctuation and second-person address is how the
/// pool was found for thirty-three seconds a scene, and it ranked a data table
/// ("ternTableOffset: -1") above the dialogue it was looking for (Ch. 59).
///
/// The interpreter hook retired them as a way of *locating* the pool. They survive as a
/// way of *recognising* a line once an address is already in hand: arming on a record
/// requires a dialogue-shaped preview, because the interpreter serves menus and item names
/// through the same loop.
/// </summary>
public partial class Mod
{
    /// <summary>
    /// Largest length up to <paramref name="want"/> that is fully readable from
    /// <paramref name="addr"/>, resolved in page steps. A mapped file's tail pages may
    /// not be resident, so committing to the header's declared size would fail the read.
    /// </summary>
    private static int ReadableLen(nuint addr, int want)
    {
        const int page = 0x1000;
        if (MemoryGuard.IsReadable(addr, want)) return want;
        int len = 0;
        while (len + page <= want && MemoryGuard.IsReadable(addr, len + page)) len += page;
        return len;
    }

    /// <summary>
    /// Same ratio test as <see cref="IsEnglishSentence"/> but over a decoded string, so
    /// UTF-16 runs can reuse it without duplicating the pointer walk.
    /// </summary>
    private static bool IsEnglishString(string s)
    {
        if (s.Length < 10 || s.Length > 400) return false;
        int letters = 0, lower = 0, vowels = 0, digits = 0, spaces = 0, other = 0;
        foreach (char ch in s)
        {
            if (ch >= 'a' && ch <= 'z')      { letters++; lower++; if (IsVowel((byte)ch)) vowels++; }
            else if (ch >= 'A' && ch <= 'Z') { letters++; if (IsVowel((byte)(ch | 0x20))) vowels++; }
            else if (ch == ' ')              spaces++;
            else if (ch >= '0' && ch <= '9') digits++;
            else if (ch == '.' || ch == ',' || ch == '!' || ch == '?' || ch == '\'' ||
                     ch == '"' || ch == '-' || ch == ':' || ch == ';') { }
            else other++;
        }
        if (spaces < 1)                       return false;
        if (letters * 100 < s.Length * 55)    return false;
        if (lower   * 100 < letters * 40)     return false;
        if (vowels  * 100 < letters * 25)     return false;
        if (vowels  * 100 > letters * 60)     return false;
        if (digits  * 100 > s.Length * 10)    return false;
        if (other   * 100 > s.Length * 5)     return false;
        return true;
    }

    /// <summary>
    /// Allocation-free approximation of the letter/lowercase/space ratios in
    /// <see cref="IsEnglishString"/>. Deliberately looser — it only has to reject the
    /// obvious non-prose cheaply so the real test runs on a small remainder.
    /// </summary>
    private static bool QuickEnglishBytes(byte[] buf, int begin, int len)
    {
        int letters = 0, lower = 0, spaces = 0;
        int end = begin + len;
        for (int i = begin; i < end; i++)
        {
            byte c = buf[i];
            if      (c >= 'a' && c <= 'z') { letters++; lower++; }
            else if (c >= 'A' && c <= 'Z') letters++;
            else if (c == ' ')             spaces++;
        }
        return spaces >= 1
               && letters * 100 >= len * 55
               && lower   * 100 >= letters * 40;
    }

    private static unsafe string AsciiPreview(nuint addr, int maxChars)
    {
        if (!MemoryGuard.IsReadable(addr, maxChars)) return "";
        byte* p = (byte*)addr;
        int begin = 0;
        while (begin < maxChars && !IsPrintable(p[begin])) begin++;
        int end = begin;
        while (end < maxChars && IsPrintable(p[end])) end++;
        if (end - begin < 8) return "";
        var sb = new System.Text.StringBuilder(end - begin);
        for (int i = begin; i < end; i++) sb.Append((char)p[i]);
        return IsEnglishString(sb.ToString()) ? sb.ToString() : "";
    }

    /// <summary>
    /// Rejects Windows runtime strings. The heap sweep spent its whole budget reporting
    /// DirectInput names, font copyright blocks and impersonation flags — all genuine
    /// UTF-16, none of it game text.
    /// </summary>
    private static bool IsSystemString(string s) =>
        s.Length == 0 ||
        s.Contains("Microsoft") || s.Contains("Windows") || s.Contains("Corporation") ||
        s.Contains("Copyright")  || s.Contains("http")    || s.Contains(".dll") ||
        s.Contains("OpenType")   || s.Contains("License") || s.Contains("reserved") ||
        s.Contains("Impersonation") || s.Contains("DirectInput") || s.Contains("Version") ||
        // Shader and engine identifiers. The top-ranked region was full of "float3
        // position" — valid English by every ratio test, and not remotely dialogue.
        s.Contains("float")  || s.Contains("_")      || s.Contains("()") ||
        s.Contains("vec")    || s.Contains("Matrix") || s.Contains("Buffer") ||
        s.Contains("Shader") || s.Contains("Texture");

    /// <summary>
    /// Scores a string as conversational rather than merely English. Item names, skill
    /// labels and shader identifiers all pass the ratio test; what separates dialogue is
    /// sentence punctuation and second-person address.
    ///
    /// Weighted rather than boolean because P5R splits lines on embedded function codes,
    /// so many genuine fragments ("? You got any big") end mid-sentence and would fail
    /// any single hard requirement.
    /// </summary>
    private static int DialogueScore(string s)
    {
        if (!IsEnglishString(s) || IsSystemString(s)) return 0;

        int score = 0;
        char last = s[^1];
        if (last == '.' || last == '!' || last == '?') score += 3;
        else if (last == ',' || last == '"')           score += 1;

        if (s.Contains(" you") || s.Contains("You ") || s.Contains(" I ") ||
            s.Contains("I'm")  || s.Contains(" me")   || s.Contains(" we ") ||
            s.Contains("Ryuji"))                       score += 3;

        if (s.Contains('\'')) score += 1;   // contractions
        if (s.Contains(' '))  score += 1;

        return score;
    }

    private static bool IsVowel(byte c) =>
        c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';

    /// <summary>
    /// True only for a byte run that reads as an English sentence. The earlier
    /// "≥10 printable, ≥2 spaces" test passed compressed texture data — binary is full
    /// of 0x20 bytes. These ratios encode what actually separates prose from binary:
    /// mostly letters, mostly lowercase, a plausible vowel share, few digits, almost
    /// no symbol junk.
    /// </summary>
    private static unsafe bool IsEnglishSentence(byte* p, int len)
    {
        if (len < 12 || len > 400) return false;

        int letters = 0, lower = 0, vowels = 0, digits = 0, spaces = 0, other = 0;
        for (int i = 0; i < len; i++)
        {
            byte c = p[i];
            if (c >= 'a' && c <= 'z')      { letters++; lower++; if (IsVowel(c)) vowels++; }
            else if (c >= 'A' && c <= 'Z') { letters++; if (IsVowel((byte)(c | 0x20))) vowels++; }
            else if (c == ' ')             spaces++;
            else if (c >= '0' && c <= '9') digits++;
            else if (c == '.' || c == ',' || c == '!' || c == '?' || c == '\'' ||
                     c == '"' || c == '-' || c == ':' || c == ';') { /* sentence punctuation */ }
            else other++;
        }

        if (spaces < 2)                     return false;
        if (letters * 100 < len * 55)       return false; // ≥55% letters
        if (lower   * 100 < letters * 40)   return false; // ≥40% of letters lowercase
        if (vowels  * 100 < letters * 25)   return false; // English runs ~38% vowels
        if (vowels  * 100 > letters * 60)   return false;
        if (digits  * 100 > len * 10)       return false; // ≤10% digits
        if (other   * 100 > len * 5)        return false; // ≤5% symbol junk
        return true;
    }

    private static bool IsPrintable(byte c) => c >= 0x20 && c <= 0x7E;

    /// <summary>
    /// Counts maximal runs of printable ASCII that read as English. Runs are delimited by
    /// ANY non-printable byte, not just NUL. BMD text carries embedded function codes —
    /// voice cues, speaker-name substitution, line breaks — as control and high bytes, so
    /// walking only to the next NUL yielded one enormous run per file, which failed the
    /// len>400 check and scored every message file zero.
    /// </summary>
    private static unsafe int CountEnglishSentences(byte* buf, int maxBytes)
    {
        int count = 0, pos = 0;
        while (pos < maxBytes)
        {
            while (pos < maxBytes && !IsPrintable(buf[pos])) pos++;
            int begin = pos;
            while (pos < maxBytes && IsPrintable(buf[pos])) pos++;
            if (pos > begin && IsEnglishSentence(buf + begin, pos - begin)) count++;
        }
        return count;
    }

    private static unsafe string PreviewSentences(nuint page, int maxBytes, int maxChars)
    {
        var sb = new System.Text.StringBuilder(maxChars + 8);
        byte* buf = (byte*)page;
        int pos = 0;
        while (pos < maxBytes && sb.Length < maxChars)
        {
            while (pos < maxBytes && !IsPrintable(buf[pos])) pos++;
            int begin = pos;
            while (pos < maxBytes && IsPrintable(buf[pos])) pos++;
            if (pos > begin && IsEnglishSentence(buf + begin, pos - begin))
            {
                if (sb.Length > 0) sb.Append(" | ");
                for (int i = begin; i < pos && sb.Length < maxChars; i++) sb.Append((char)buf[i]);
            }
        }
        return sb.ToString();
    }

    private static unsafe int CountNullTermStrings(byte* buf, int maxBytes, int minPrintable)
    {
        int count = 0, pos = 0;
        while (pos < maxBytes)
        {
            int start = pos, printable = 0;
            while (pos < maxBytes && buf[pos] != 0)
            {
                if (buf[pos] >= 0x20 && buf[pos] <= 0x7E) printable++;
                pos++;
            }
            int len = pos - start;
            if (len > 0 && printable >= minPrintable) count++;
            if (pos >= maxBytes) break;
            pos++; // skip '\0'
        }
        return count;
    }

    private static unsafe int CountPrintableRuns(nuint regionBase, int scanBytes)
    {
        byte* p = (byte*)regionBase;
        int runs = 0;
        int i = 0;

        while (i < scanBytes)
        {
            byte c = p[i];
            if (c < 0x20 || c >= 0x7F) { i++; continue; }

            int start = i;
            int letters = 0, spaces = 0;
            while (i < scanBytes)
            {
                c = p[i];
                if (c < 0x20 || c >= 0x7F) break;
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) letters++;
                if (c == ' ') spaces++;
                i++;
            }
            int runLen = i - start;
            if (runLen >= 12 && spaces >= 1 && letters * 10 >= runLen * 6)
                runs++;
        }

        return runs;
    }
}
