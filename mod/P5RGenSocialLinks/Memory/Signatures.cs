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
    // PLACEHOLDER — must be verified against your p5r.exe build via Ghidra.
    internal const string BeginConversation =
        "48 89 5C 24 ?? 48 89 6C 24 ?? 48 89 74 24 ?? 57 41 56 41 57";

    // void __fastcall SocialLink_AdvanceLine(SocialLinkSession* session, int lineIndex)
    // PLACEHOLDER
    internal const string AdvanceLine =
        "48 83 EC ?? 48 8B 05 ?? ?? ?? ?? 48 85 C0 74 ??";
}