using System;
using System.Diagnostics;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// One-shot diagnostic: identifies which loaded module contains the BF script
/// line-counter write instruction found via Cheat Engine (0x7FFA995C2928).
/// Logs the module name and RVA so we can load the right DLL in Ghidra.
/// </summary>
internal static class ModuleProbe
{
    // Absolute address of the `mov [rcx+18],eax` instruction that writes the
    // dialogue line counter. Found via CE "Find what writes" on 2026-08-09.
    // This address is ASLR-relative; the module base changes each boot, but the
    // RVA (offset within the DLL) is stable and is what Ghidra needs.
    private static readonly nuint WriteInstrAddr = unchecked((nuint)0x7FFA995C2928UL);

    internal static void LogWriteInstructionModule(Action<string> log)
    {
        try
        {
            foreach (ProcessModule mod in Process.GetCurrentProcess().Modules)
            {
                nuint modBase = (nuint)mod.BaseAddress;
                nuint modEnd  = modBase + (nuint)mod.ModuleMemorySize;

                if (WriteInstrAddr >= modBase && WriteInstrAddr < modEnd)
                {
                    nuint rva = WriteInstrAddr - modBase;
                    log(
                        $"[ModuleProbe] Write instr 0x{WriteInstrAddr:X} → " +
                        $"{mod.ModuleName} base=0x{modBase:X} RVA=0x{rva:X}");
                    return;
                }
            }

            log($"[ModuleProbe] Write instr 0x{WriteInstrAddr:X} not found in any loaded module " +
                          $"— address may have changed since CE session.");
        }
        catch (Exception ex)
        {
            log($"[ModuleProbe] Module enumeration failed: {ex.Message}");
        }
    }
}
