namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Reads the scripted NPC dialogue currently in the game buffer and formats
/// it as a context string for the LLM prompt.
///
/// Must be called AFTER OriginalFunction() — the game writes the scripted
/// line into the buffer during that call, so the buffer is populated by the
/// time we read it.
/// </summary>
internal static class ContextBuilder
{
    private const int MaxReadChars = 512;

    /// <summary>
    /// Formats a context string for the LLM from a pre-read dialogue string.
    /// Pure method — no unsafe code, fully testable without game memory.
    /// </summary>
    internal static string Build(int dialogueIndex, string rawDialogue)
    {
        string trimmed = rawDialogue.Trim();
        if (string.IsNullOrEmpty(trimmed))
            return $"Dialogue line {dialogueIndex}";

        return $"[Line {dialogueIndex}] NPC says: \"{trimmed}\"";
    }

    /// <summary>
    /// Reads the null-terminated UTF-16LE string from the game's dialogue
    /// buffer and delegates to <see cref="Build"/>.
    /// </summary>
    internal static unsafe string ReadAndBuild(SocialLinkSnapshot snap)
    {
        char* ptr = (char*)(snap.SessionBase + (nuint)P5ROffsets.DIALOGUE_BUFFER);

        // Walk until null terminator or safety cap — a missing terminator
        // would otherwise cause us to read arbitrarily far into VRAM.
        int len = 0;
        while (len < MaxReadChars && ptr[len] != '\0')
            len++;

        // new string(char*, start, length) copies from unmanaged → managed memory.
        // The GC owns the copy; the game can overwrite the buffer freely after this.
        string raw = len == 0 ? string.Empty : new string(ptr, 0, len);
        return Build(snap.DialogueIndex, raw);
    }
}
