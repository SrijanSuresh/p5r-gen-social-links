using System;
using System.Text;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Writes LLM-generated text into a specific slot of P5R's pre-loaded script text pool.
///
/// The pool is a contiguous sequence of null-terminated UTF-8 strings:
///   [str0]\0[str1]\0[str2]\0...
///
/// Write strategy: navigate to the target line by counting null terminators, read the
/// original string's length, then overwrite with LLM text truncated to that length.
/// Truncation is mandatory — writing more bytes than the original string would corrupt
/// the next string in the pool.
/// </summary>
internal static class DialogueWriter
{
    // How far into the pool to search for a line offset (safety cap).
    private const int MaxPoolScan = 32768;

    /// <summary>
    /// Overwrites line <paramref name="lineIndex"/> (0-based) in the text pool at
    /// <paramref name="poolBase"/> with <paramref name="text"/>.
    /// Returns false if the line could not be located or the page is not writable.
    /// </summary>
    internal static unsafe bool WriteAtLineIndex(nuint poolBase, int lineIndex, string text)
    {
        if (poolBase == 0 || lineIndex < 0 || string.IsNullOrEmpty(text)) return false;

        // Walk past 'lineIndex' null terminators to reach the target slot.
        int offset = FindLineOffset((byte*)poolBase, lineIndex);
        if (offset < 0) return false;

        // Measure the original string at this slot — we must not exceed its length.
        int origLen = MeasureString((byte*)poolBase + offset, MaxPoolScan - offset);
        if (origLen <= 0) return false;

        nuint writeAddr = poolBase + (nuint)offset;
        if (!MemoryGuard.IsWritable(writeAddr, origLen + 1)) return false;

        // Encode as UTF-8 (ASCII-range for English P5R PC) and truncate to fit.
        byte[] encoded = Encoding.UTF8.GetBytes(text);
        int writeLen   = Math.Min(encoded.Length, origLen);

        byte* dest = (byte*)writeAddr;
        fixed (byte* src = encoded)
            Buffer.MemoryCopy(src, dest, origLen, writeLen);

        dest[writeLen] = 0; // null-terminate

        return true;
    }

    // ── Helpers ──────────────────────────────────────────────────────────

    private static unsafe int FindLineOffset(byte* pool, int lineIndex)
    {
        int linesSkipped = 0;
        for (int i = 0; i < MaxPoolScan; i++)
        {
            if (pool[i] != 0) continue;

            // Found a null terminator — the next string starts at i+1.
            linesSkipped++;
            if (linesSkipped == lineIndex) return i + 1;
        }
        return -1; // lineIndex beyond pool scan range
    }

    private static unsafe int MeasureString(byte* p, int maxLen)
    {
        for (int i = 0; i < maxLen; i++)
            if (p[i] == 0) return i;
        return -1; // no null terminator found within maxLen
    }
}
