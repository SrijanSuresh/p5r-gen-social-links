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

    // Field offsets confirmed via hex dump during real Ryuji gym hang-out
    // (session=0x41D7156660):
    //   +000: 08 00 00 00  → int32 = 8  (Ryuji cmmId)           CONFIRMED
    //   +004: 00 00 00 00  → int32 = 0  (dialogue line index, starts at 0)  CONFIRMED
    //   +008: 08           → byte  = 8  (rank level, Ryuji rank 8 mid-game) CONFIRMED
    //   +009: 00
    //   +00A: 02           → byte  = 2  (event-type, read by CMM_EXEC_EVENT)
    //   +00C: 33 00        → int16 = 51 (unknown — scene/event number?)
    //   +010: ptr (unknown)
    internal const int CONFIDANT_ID    = 0x00;  // int32
    internal const int DIALOGUE_INDEX  = 0x04;  // int32 — line 0 = first dialogue
    internal const int RANK_LEVEL      = 0x08;  // byte  — 1-10
    internal const int DIALOGUE_BUFFER = 0x10;  // ptr candidate — unverified
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
