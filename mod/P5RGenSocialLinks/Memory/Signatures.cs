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

    // The BMD message interpreter's byte-fetch loop — the instruction that pulls the
    // next character out of the message currently being displayed.
    //
    // Found by putting a hardware read watchpoint on a line while it was on screen
    // (learning.md Ch. 65). Five call sites read that byte; this one is the only one
    // that loads BOTH the record pointer and the cursor from memory, so a single hook
    // on it yields the whole interpreter state from one register.
    //
    //   MOVSXD RDX, dword [RBX+0x30]   48 63 53 30   <- byte cursor into the record
    //   MOV    RAX, [RBX+0x20]         48 8B 43 20   <- pointer to the message record
    //   MOVZX  EDI, byte [RDX+RAX]     0F B6 3C 02   <- the character
    //   TEST   EDI, EDI                85 FF
    //   JNE    loop                    0F 85         <- displacement deliberately excluded
    //
    // Observed live: RAX=0x424F054798, RDX=0x28, RDI=0x4F ('O'), with "Oh yeah! You
    // bring your stuff?" in the bubble.
    //
    // Verified unique by scanning the 378 MB P5R.exe on disk: exactly one occurrence,
    // at RVA 0x17A3D1F in section .sdata. Nothing is wildcarded, and that is deliberate
    // - the struct offsets 0x20 and 0x30 are the payload, not incidental encoding, so a
    // build that moves them has to fail the scan loudly instead of matching and reading
    // the wrong field.
    internal const string MsgByteFetch =
        "48 63 53 30 48 8B 43 20 0F B6 3C 02 85 FF 0F 85";

    /// Offset from the match to the MOVZX itself, for hooking the read rather than the
    /// two loads that feed it.
    internal const int MsgByteFetchToMovzx = 8;

    /// Field offsets inside the interpreter state struct that RBX points at.
    internal const int MsgRecordPtrOffset = 0x20;
    internal const int MsgCursorOffset    = 0x30;
}