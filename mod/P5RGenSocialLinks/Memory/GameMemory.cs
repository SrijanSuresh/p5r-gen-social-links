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

    // Field offsets confirmed via hex dump (session=0x41D7156660, Ryuji conversation):
    //   +000: 08 00 00 00  → int32 = 8 (Ryuji cmmId) ✓
    //   +004: 00 00 00 00  → int32 = 0 (rank level — 0 in field, non-zero in hang-out)
    //   +00A: 02           → byte event-type (what CMM_EXEC_EVENT reads; NOT confidant id)
    //   +00C: 33 00        → int16 candidate (dialogue index?)
    //   +010: ptr 0x70B8C058 (unknown pointer)
    // RANK_LEVEL and DIALOGUE_INDEX still need hang-out session to confirm values.
    internal const int CONFIDANT_ID    = 0x00;  // int32 — cmmId confirmed by hex dump
    internal const int RANK_LEVEL      = 0x04;  // int32 candidate — 0 in field dialogue
    internal const int DIALOGUE_INDEX  = 0x0C;  // int16 candidate — needs verification
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
