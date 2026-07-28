namespace P5RGenSocialLinks.Memory;

// Offsets discovered via Ghidra analysis of CMM_EXEC_EVENT native function.
// All values are relative to the module base address of p5r.exe.
internal static class P5ROffsets
{
    // Static pointer to the CommunityManager object.
    // Found via Ghidra: CMM_EXEC_EVENT at 0x140E0D0B0 opens with
    //   MOV RAX, [DAT_142a63ef0]  →  offset 0x2A63EF0 from image base.
    internal const int SL_STATIC_PTR = 0x02_A6_3E_F0;

    // Chain offset: [CMM + 0x48] = active community event/session sub-object.
    // Discovered from:  MOV RCX, [RAX + 0x48]  in CMM_EXEC_EVENT.
    internal const int CMM_SESSION_OFFSET = 0x48;

    // Field offsets within the session sub-object — byte at +0xA observed in
    // CMM_EXEC_EVENT: MOVZX EAX, byte ptr [RCX + 0xA].
    // Remaining offsets (rank, dialogue) are candidates pending hex-dump confirmation.
    internal const int CONFIDANT_ID    = 0x0A;  // byte — cmmId 1-21 (arcana index)
    internal const int RANK_LEVEL      = 0x0B;  // byte candidate — needs verification
    internal const int DIALOGUE_INDEX  = 0x0C;  // byte candidate — needs verification
    internal const int DIALOGUE_BUFFER = 0x10;  // ptr candidate — needs verification
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
