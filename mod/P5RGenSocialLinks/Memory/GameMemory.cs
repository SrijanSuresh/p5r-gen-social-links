namespace P5RGenSocialLinks.Memory;

// Offsets discovered via Cheat Engine / ghidra analysis of P5R PC binary.
// All values are relative to the module base address of p5r.exe.
// These are placeholder offsets — they MUST be verified against the actual binary.
internal static class P5ROffsets
{
    // Pointer chain to the active Social Link session object:
    // [[[BaseModule + SL_STATIC_PTR] + 0x18] + 0x08] = SocialLinkSession*
    internal const int SL_STATIC_PTR   = 0x01_E8_00_00;

    // Offsets within SocialLinkSession struct
    internal const int CONFIDANT_ID    = 0x00;  // int32 — which arcana/character
    internal const int RANK_LEVEL      = 0x04;  // int32 — current rank 1-10
    internal const int DIALOGUE_INDEX  = 0x08;  // int32 — script line index
    internal const int DIALOGUE_BUFFER = 0x10;  // wchar_t* — pointer to UTF-16 text buffer
}

// Memory layout of the Social Link session struct (mirrors C++ object in P5R)
[System.Runtime.InteropServices.StructLayout(
    System.Runtime.InteropServices.LayoutKind.Explicit, Size = 0x20)]
internal unsafe struct SocialLinkSession
{
    [System.Runtime.InteropServices.FieldOffset(0x00)] public int  ConfidantId;
    [System.Runtime.InteropServices.FieldOffset(0x04)] public int  RankLevel;
    [System.Runtime.InteropServices.FieldOffset(0x08)] public int  DialogueIndex;
    [System.Runtime.InteropServices.FieldOffset(0x10)] public char* DialogueBuffer;
}
