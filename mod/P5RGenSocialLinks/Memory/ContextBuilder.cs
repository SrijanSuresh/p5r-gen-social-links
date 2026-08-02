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
    ///
    /// +0x10 in the session struct stores a char* (pointer to the string),
    /// not the string itself. We need two unsafe reads: first read the 8-byte
    /// pointer value stored at sessionBase+0x10, then dereference that pointer
    /// to reach the actual UTF-16LE text.
    /// </summary>
    internal static unsafe string ReadAndBuild(SocialLinkSnapshot snap)
    {
        // Step 1: validate the address that holds the pointer field
        nuint ptrFieldAddr = snap.SessionBase + (nuint)P5ROffsets.DIALOGUE_BUFFER;
        if (!MemoryGuard.IsReadable(ptrFieldAddr, sizeof(nuint)))
            return Build(snap.DialogueIndex, string.Empty);

        // Step 2: read the pointer value (the actual string address)
        nuint strAddr = *(nuint*)ptrFieldAddr;
        if (strAddr == 0)
            return Build(snap.DialogueIndex, string.Empty);

        // Step 3: validate the string address itself (need at least 2 bytes for one char)
        if (!MemoryGuard.IsReadable(strAddr, 2))
            return Build(snap.DialogueIndex, string.Empty);

        char* ptr = (char*)strAddr;

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

    /// <summary>
    /// Returns the string address stored at sessionBase+DIALOGUE_BUFFER, or 0 if
    /// the pointer field is unreadable. Used by the poll loop for offset discovery.
    /// </summary>
    internal static unsafe nuint PeekDialoguePtr(nuint sessionBase)
    {
        nuint ptrFieldAddr = sessionBase + (nuint)P5ROffsets.DIALOGUE_BUFFER;
        if (!MemoryGuard.IsReadable(ptrFieldAddr, sizeof(nuint))) return 0;
        return *(nuint*)ptrFieldAddr;
    }
}
