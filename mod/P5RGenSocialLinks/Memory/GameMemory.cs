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
    //   +000: 08 00 00 00  → int32 = 8  (Ryuji cmmId)                CONFIRMED
    //   +004: 00 00 00 00  → int32 = 0  (dialogue line, starts at 0) CONFIRMED
    //   +008: 08           → byte  = 8  (arcana/cmmId repeated — NOT rank)
    //   +009: 00
    //   +00A: 02           → byte  = 2  (event-type, read by CMM_EXEC_EVENT)
    //   +00B: 04           → byte  = 4  (rank before this session; Ryuji went 4→5) CONFIRMED
    //   +00C: 33 00        → int16 = 51 (unknown — scene/event number)
    //   +010: ptr (unknown)
    internal const int CONFIDANT_ID    = 0x00;  // int32
    internal const int DIALOGUE_INDEX  = 0x04;  // int32 — increments with each line advance
    internal const int RANK_LEVEL      = 0x0B;  // byte  — rank before this session (1-10)
    internal const int SCENE_NUMBER    = 0x0C;  // int16 — specific hang-out scene id (e.g. 51)
}

// Memory layout of the Social Link session struct (mirrors C++ object in P5R)
// All fields confirmed from live hex dump during Ryuji gym hang-out (session=0x41D7156660).
// DIALOGUE_BUFFER at +0x10 is a pointer candidate — see ContextBuilder.ReadAndBuild.
[System.Runtime.InteropServices.StructLayout(
    System.Runtime.InteropServices.LayoutKind.Explicit, Size = 0x20)]
internal unsafe struct SocialLinkSession
{
    [System.Runtime.InteropServices.FieldOffset(0x00)] public int   ConfidantId;    // cmmId (8=Ryuji)
    [System.Runtime.InteropServices.FieldOffset(0x04)] public int   DialogueIndex;  // line 0 = conversation start
    [System.Runtime.InteropServices.FieldOffset(0x08)] public byte  CmmIdRepeat;    // arcana/cmmId repeated
    [System.Runtime.InteropServices.FieldOffset(0x0A)] public byte  EventType;      // read by CMM_EXEC_EVENT
    [System.Runtime.InteropServices.FieldOffset(0x0B)] public byte  RankLevel;      // rank before this session
    [System.Runtime.InteropServices.FieldOffset(0x0C)] public short SceneNumber;    // e.g. 51 — scene/event id
    // +0x10 onwards: internal CMM pointers — no dialogue text accessible from here.
}
