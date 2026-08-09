namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Byte signatures for P5R functions located via SigScan.
/// Each pattern targets a unique instruction sequence in the function prologue.
/// Wildcard bytes (??) mark compiler-generated stack offsets that vary between builds.
///
/// HOW TO UPDATE: attach Cheat Engine → "Find what writes to" ConfidantId field →
/// copy the function address → open in Ghidra → copy first ~20 bytes → wildcard
/// any displacement/offset bytes.
/// </summary>
internal static class Signatures
{
    // void __fastcall SocialLink_BeginConversation(SocialLinkSession* session)
    // Captured via Cheat Engine "Find what writes to" ConfidantId field.
    // 48 89 5C 24 ?? = MOV [RSP+??], RBX  (stack offset wildcarded — compiler-chosen)
    // 57             = PUSH RDI
    // 48 83 EC 20    = SUB RSP, 0x20       (shadow space allocation)
    // 48 89 74 24 ?? = MOV [RSP+??], RSI  (stack offset wildcarded)
    // 48 8D 99 B8 62 00 00 = LEA RBX, [RCX+0x62B8]  (session field offset — unique)
    // CMM_EXEC_EVENT — fires when the game executes a Social Link community event.
    // Discovered via Ghidra at 0x140E0D0B0. Wildcards cover RIP-relative displacements
    // and short-jump offsets that shift between builds.
    //   MOV RAX,[CMM_global]   48 8B 05 ?? ?? ?? ??
    //   TEST RAX,RAX           48 85 C0
    //   JZ end                 74 ??
    //   MOV RCX,[RAX+0x48]     48 8B 48 48          ← unique session sub-object load
    //   TEST RCX,RCX           48 85 C9
    //   JZ end                 74 ??
    //   MOVZX EAX,byte[RCX+A]  0F B6 41 0A          ← confidant-id byte read
    internal const string CmmExecEvent =
        "48 8B 05 ?? ?? ?? ?? 48 85 C0 74 ?? 48 8B 48 48 48 85 C9 74 ?? 0F B6 41 0A";

    // void __fastcall SocialLink_AdvanceLine(SocialLinkSession* session, int lineIndex)
    // PLACEHOLDER
    internal const string AdvanceLine =
        "48 83 EC ?? 48 8B 05 ?? ?? ?? ?? 48 85 C0 74 ??";
}