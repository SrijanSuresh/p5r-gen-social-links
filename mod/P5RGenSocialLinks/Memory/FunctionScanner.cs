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

    internal unsafe FunctionScanner()
    {
        ProcessModule module = Process.GetCurrentProcess().MainModule!;
        _moduleBase = (nuint)module.BaseAddress;
        // Pin the module memory region for scanning
        _scanner = new Scanner((byte*)_moduleBase, module.ModuleMemorySize);
    }

    /// <summary>
    /// Scans for <paramref name="pattern"/> and returns its absolute address.
    /// Throws <see cref="InvalidOperationException"/> if not found.
    /// </summary>
    internal nuint FindOrThrow(string pattern)
    {
        PatternScanResult result = _scanner.CompiledFindPattern(pattern);
        if (!result.Found)
            throw new InvalidOperationException(
                $"SigScan failed for pattern [{pattern}]. " +
                "Update Signatures.cs against the current p5r.exe build.");

        return _moduleBase + (nuint)result.Offset;
    }

    public void Dispose() => _scanner.Dispose();
}