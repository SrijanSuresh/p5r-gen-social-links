using System;
using System.Diagnostics;
using Reloaded.Memory.Sigscan;
using Reloaded.Memory.Sigscan.Definitions.Structs;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Thin wrapper around Reloaded.Memory.SigScan that resolves a byte pattern
/// to an absolute virtual address within p5r.exe''s module region.
/// </summary>
internal sealed class FunctionScanner : IDisposable
{
    private readonly Scanner _scanner;
    private readonly nuint   _moduleBase;

    internal FunctionScanner()
    {
        Process process = Process.GetCurrentProcess();
        ProcessModule module = process.MainModule!;
        _moduleBase = (nuint)module.BaseAddress;
        _scanner = new Scanner(process, module);
    }

    /// <summary>
    /// Scans for <paramref name="pattern"/> and returns its absolute address.
    /// Throws <see cref="InvalidOperationException"/> when the pattern is not found.
    /// </summary>
    internal nuint FindOrThrow(string pattern)
    {
        PatternScanResult result = _scanner.FindPattern(pattern);
        if (!result.Found)
            throw new InvalidOperationException(
                $"SigScan failed for pattern [{pattern}]. " +
                "Update Signatures.cs against the current p5r.exe build.");

        return _moduleBase + (nuint)result.Offset;
    }

    /// <summary>
    /// Returns the offset of the first occurrence of <paramref name="pattern"/>,
    /// or null if not found. Used for discovery logging without hooking.
    /// </summary>
    internal nuint? TryFindFirst(string pattern)
    {
        PatternScanResult result = _scanner.FindPattern(pattern);
        return result.Found ? _moduleBase + (nuint)result.Offset : null;
    }

    public void Dispose() => _scanner.Dispose();
}