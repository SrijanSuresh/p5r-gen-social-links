using System;
using System.Text;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Writes LLM-generated text back into the game''s dialogue buffer.
/// P5R stores dialogue as null-terminated UTF-16LE strings in a heap buffer
/// pointed to by SocialLinkSession.DialogueBuffer.
/// </summary>
internal static class DialogueWriter
{
    // Maximum characters we will write — stays within P5R''s expected buffer bounds.
    // Actual buffer size must be confirmed via Ghidra; 256 wchars is a safe conservative limit.
    private const int MaxChars = 256;

    /// <summary>
    /// Overwrites the dialogue buffer at <paramref name="bufferPtr"/> with
    /// <paramref name="text"/>, truncating to <see cref="MaxChars"/> if needed.
    /// </summary>
    internal static unsafe void Write(nuint bufferPtr, string text)
    {
        if (bufferPtr == 0)
            throw new ArgumentException("Null dialogue buffer pointer.");

        ReadOnlySpan<char> chars = text.AsSpan(0, Math.Min(text.Length, MaxChars - 1));
        char* dest = (char*)bufferPtr;

        for (int i = 0; i < chars.Length; i++)
            dest[i] = chars[i];

        dest[chars.Length] = '\0';  // null-terminate
    }
}