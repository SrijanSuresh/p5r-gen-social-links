using System;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Reloaded.Hooks.Definitions;
using Reloaded.Hooks.Definitions.X64;
using Reloaded.Mod.Interfaces;
using Reloaded.Mod.Interfaces.Internal;
using P5RGenSocialLinks.Memory;
using P5RGenSocialLinks.Server;

namespace P5RGenSocialLinks;

public class Mod : IModV1
{
    private ILoggerV2?        _logger;
    private ModLogger?        _modLog;
    private LLMClient?        _llmClient;
    private DialogueBridge?   _bridge;
    private SocialLinkReader? _reader;

    private sealed class LoggerAdapter : DialogueBridge.ILogger
    {
        private readonly ILoggerV2 _inner;
        internal LoggerAdapter(ILoggerV2 inner) => _inner = inner;
        public void WriteLine(string msg) => _inner.WriteLine(msg);
    }

    private PeriodicTimer?          _timer;
    private Task?                   _pollTask;
    private CancellationTokenSource _cts = new();

    // CmmExecEvent fires once at hang-out init (not per-line, kept for session context).
    private IHook<CmmExecEventDelegate>? _conversationHook;
    private IReloadedHooks?              _hooks;
    private int                          _cmmExecFireCount;

    [Function(CallingConventions.Microsoft)]
    public delegate nint CmmExecEventDelegate();

    // Per-line hook: intercepts FUN_1405a8570 (the REP MOVSB inner copy function).
    // The dialogue system calls this DIRECTLY via function pointer (0x158131be0),
    // bypassing the outer dispatcher FUN_1405a8590 that uses standard registers.
    // Custom convention: RCX=dst, R10=src (non-standard!), R8=count.
    [Function(
        new FunctionAttribute.Register[] {
            FunctionAttribute.Register.rcx,
            FunctionAttribute.Register.r10,
            FunctionAttribute.Register.r8 },
        FunctionAttribute.Register.rax, false)]
    private delegate void MemcpyInnerDelegate(nuint dst, nuint src, nuint count);

    private IHook<MemcpyInnerDelegate>? _memcpyHook;

    // BF opcode dispatcher hook: FUN_14024EE00.
    // Called for every BF instruction. RCX=channel(0x2B), RDX=opcode type byte,
    // R8=opcode_struct[+0x08], R9=opcode_struct[+0x10].
    // When RDX==5 this is a dialogue instruction — log R8/R9 to find the text pointer.
    // Return type is nuint (not void) — the dispatcher may return a status value
    // that callers (e.g. FUN_141844f20) check. Declaring void leaves RAX garbage.
    [Function(CallingConventions.Microsoft)]
    private delegate nuint BfOpcodeDispatchDelegate(nuint channel, nuint typeAndFlags,
                                                     nuint arg2, nuint arg3);

    private IHook<BfOpcodeDispatchDelegate>? _bfDispatchHook;

    // Dialogue heap sits above 256 GB; CLR/runtime copies are all below 4 GB.
    private static readonly nuint HeapLow = unchecked((nuint)0x4000000000UL);

    // Cached BF script buffer address — found by TryFindBfBuffer() on any tick,
    // then used by ProbeBfLine() for the rest of the session.
    private nuint  _bfBufferBase;
    private int    _bfBufferOff;   // session struct offset where we found it (for logging)

    // BMD text pool: the first page in the ±512KB bfBase vicinity that has ≥5 English
    // dialogue strings (null-terminated, ≥2 spaces, ≥10 printable chars). Found once
    // per session after bfBase is confirmed; written via VirtualProtect(PAGE_WRITECOPY).
    private nuint _bmdTextPool;
    private bool  _bmdScanDoneV2;
    private int   _bmdScanAttempts;

    // (offset, original length) of every dialogue-looking string in _bmdTextPool.
    // Captured once at discovery, BEFORE any write — see CapturePoolSlots for why.
    private (int Off, int Len)[]? _poolSlots;

    // Byte length of the pool region — the MSG1 file size when found via magic,
    // otherwise a single page. Used to size the VirtualProtect call.
    private int _bmdPoolLen = 0x1000;

    // Most recent LLM response. Written on the async LLM thread; read volatilely on the
    // game thread inside OnGameMemcpy so no lock is needed (reference read/write is
    // atomic on x64, and we only need eventual visibility, not strict ordering).
    private volatile string? _lastLlmText;

    // Large-copy log: OnGameMemcpy records the dst of every heap-to-heap copy
    // ≥ 500 bytes. TryFindBfBuffer probes these at session start to locate the
    // BF script buffer, which is loaded in one bulk copy before session detection.
    private readonly System.Collections.Generic.List<nuint> _largeCopyDsts = new();
    private readonly object _largeCopyLock = new();

    private GenConfig _cfg = new();

    public void Start(IModLoaderV1 loader)
    {
        _logger = (ILoggerV2)loader.GetLogger();

        string modDir = System.IO.Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? ".";
        _cfg    = GenConfig.Load(modDir);
        _modLog = new ModLogger(_logger, _cfg.LogLevel);
        _modLog.Always($"[P5RGenSocialLinks] Config: throttle={_cfg.ThrottleSeconds}s timeout={_cfg.TimeoutSeconds}s url={_cfg.ServerUrl} logLevel={_cfg.LogLevel}");

        _llmClient = new LLMClient(_cfg.ServerUrl);
        ServerHealthChecker.CheckAsync(_cfg.ServerUrl, msg => _logger.WriteLine(msg));
        _logger.WriteLine("[P5RGenSocialLinks] Post-health — entering setup.");

        try
        {
            nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
            _logger.WriteLine($"[P5RGenSocialLinks] Base: 0x{moduleBase:X}");

            _reader = new SocialLinkReader(moduleBase, _cfg.VerboseChain,
                msg => _logger!.WriteLine(msg));
            _logger.WriteLine("[P5RGenSocialLinks] SocialLinkReader OK.");

            _bridge = new DialogueBridge(_llmClient!, new LoggerAdapter(_logger!), _cfg);
            _logger.WriteLine("[P5RGenSocialLinks] DialogueBridge OK.");

            loader.GetController<IReloadedHooks>()?.TryGetTarget(out _hooks);
            _logger.WriteLine($"[P5RGenSocialLinks] IReloadedHooks: {(_hooks is not null ? "OK" : "null")}");

            TryActivateHook();
            SetupMemcpyHook();
            // BfDispatch hook crashes regardless of handler — abandoned, hunting text ptr via CE instead
            StartPollLoop();

            _logger.WriteLine($"[P5RGenSocialLinks] Started — hook:{(_hookActive ? "ON" : "OFF")} poll:ON");
        }
        catch (Exception ex)
        {
            _logger.WriteLine($"[P5RGenSocialLinks] STARTUP CRASH: {ex.GetType().Name}: {ex.Message}");
            _logger.WriteLine(ex.StackTrace ?? "(no stack trace)");
        }
    }

    private bool _hookActive;

    private unsafe void SetupBfDispatchHook()
    {
        if (_hooks is null) return;
        try
        {
            nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
            nuint addr = moduleBase + 0x24EE00;

            // Sanity-check: log first 8 bytes so we can verify this is a real function
            // prologue and not a data section or already-patched trampoline.
            byte* p = (byte*)addr;
            string byteDump = $"{p[0]:X2} {p[1]:X2} {p[2]:X2} {p[3]:X2} {p[4]:X2} {p[5]:X2} {p[6]:X2} {p[7]:X2}";
            _logger!.WriteLine($"[P5RGenSocialLinks] BfDispatch target 0x{addr:X} bytes: {byteDump}");

            _bfDispatchHook = _hooks.CreateHook<BfOpcodeDispatchDelegate>(
                OnBfOpcodeDispatch, (long)addr).Activate();
            _logger!.WriteLine($"[P5RGenSocialLinks] BfDispatch hook ACTIVE at 0x{addr:X}");
        }
        catch (Exception ex)
        {
            _logger!.WriteLine($"[P5RGenSocialLinks] BfDispatch hook FAILED: {ex.GetType().Name}: {ex.Message}");
        }
    }

    // Diagnostic: bare pass-through — if this crashes, the hook address/convention is wrong.
    // If this is stable, the crash is in our logging/memory-read code above.
    private static int _bfCallCount;
    private unsafe nuint OnBfOpcodeDispatch(nuint channel, nuint typeAndFlags,
                                             nuint arg2, nuint arg3)
    {
        nuint result = _bfDispatchHook!.OriginalFunction(channel, typeAndFlags, arg2, arg3);
        System.Threading.Interlocked.Increment(ref _bfCallCount);
        return result;
    }

    // Windows x64 user-mode addresses are capped at 128 TB (bit 47).
    // Values above this are data misread as pointers — reject them.
    private static readonly nuint UserAddrMax = unchecked((nuint)0x7FFFFFFFFFFFUL);

    // Applies the 3-hop chain: descriptor → descriptor+0x18 → textObj → *(textObj) → charPtr.
    // Final gate: charPtr must be in valid heap range AND look like English dialogue text.
    private static unsafe nuint FollowTextObjChain(nuint descriptor)
    {
        if (!MemoryGuard.IsReadable(descriptor + 0x18, 8)) return 0;
        nuint textObj = *(nuint*)(descriptor + 0x18);
        if (textObj == 0 || !MemoryGuard.IsReadable(textObj, 8)) return 0;
        nuint charPtr = *(nuint*)textObj;
        if (charPtr < HeapLow || charPtr > UserAddrMax) return 0;
        return LooksLikeText(charPtr) ? charPtr : 0;
    }

    // Returns true if ≥8 printable ASCII bytes and ≥1 space in first 48 bytes — looks
    // like English dialogue rather than binary data, a filepath, or a null region.
    private static unsafe bool LooksLikeText(nuint addr)
    {
        if (!MemoryGuard.IsReadable(addr, 48)) return false;
        byte* p = (byte*)addr;
        int printable = 0, spaces = 0;
        for (int i = 0; i < 48 && p[i] != 0; i++)
        {
            if (p[i] >= 0x20 && p[i] <= 0x7E) { printable++; if (p[i] == ' ') spaces++; }
        }
        return printable >= 8 && spaces >= 1;
    }

    // Finds the actual character buffer. Tries three paths in order:
    //   A) session+0xD0 → heap descriptor chain (4-hop)
    //   B) session+0xD0 value is a direct mapped-file pointer (< HeapLow, looks like text)
    //   C) Fallback heap scan of session[0x00..0xC8) — external ptrs probed as descriptors
    private static unsafe nuint TryReadTextAddr(nuint session)
    {
        if (MemoryGuard.IsReadable(session + 0xD0, 8))
        {
            nuint d = *(nuint*)((byte*)session + 0xD0);
            if (d != 0)
            {
                // Path A: d is a heap descriptor → follow chain
                if (d >= HeapLow && d <= UserAddrMax)
                {
                    nuint cp = FollowTextObjChain(d);
                    if (cp != 0) return cp;
                }
                // Path B: d is a mapped-file address → treat as direct text pointer
                else if (d >= 0x1000 && d < HeapLow && LooksLikeText(d))
                {
                    return d;
                }
            }
        }

        // Path C: scan session[0x00..0x100) per slot for external heap pointers.
        // Per-slot IsReadable handles VirtualQuery region boundaries — the bulk check
        // fails at 0xC8 but individual slots at 0xD8/0xE0 can still be readable.
        for (int off = 0; off < 0x100; off += 8)
        {
            nuint slotAddr = session + (nuint)off;
            if (!MemoryGuard.IsReadable(slotAddr, 8)) continue;
            nuint ptr = *(nuint*)(byte*)slotAddr;
            if (ptr < HeapLow || ptr > UserAddrMax) continue;
            if (ptr >= session && ptr < session + 0x1000) continue; // self-referential
            nuint cp = FollowTextObjChain(ptr);
            if (cp != 0) return cp;
        }
        return 0;
    }

    private static unsafe string TryReadString(nuint addr)
    {
        if (addr < unchecked((nuint)0x1000000UL)) return "";
        if (!Memory.MemoryGuard.IsReadable(addr, 32)) return "?";
        byte* p = (byte*)addr;
        var sb = new System.Text.StringBuilder(32);
        for (int i = 0; i < 32 && p[i] != 0; i++)
            sb.Append(p[i] >= 0x20 && p[i] <= 0x7E ? (char)p[i] : '·');
        return sb.ToString();
    }

    private void SetupMemcpyHook()
    {
        if (_hooks is null)
        {
            _logger!.WriteLine("[P5RGenSocialLinks] Memcpy inner hook skipped — IReloadedHooks null.");
            return;
        }
        try
        {
            nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
            nuint addr = moduleBase + 0x5A8570;
            _memcpyHook = _hooks.CreateHook<MemcpyInnerDelegate>(OnGameMemcpy, (long)addr).Activate();
            _logger!.WriteLine($"[P5RGenSocialLinks] Memcpy inner hook ACTIVE at 0x{addr:X}");
        }
        catch (Exception ex)
        {
            _logger!.WriteLine($"[P5RGenSocialLinks] Memcpy hook FAILED: {ex.GetType().Name}: {ex.Message}");
        }
    }

    private unsafe void OnGameMemcpy(nuint dst, nuint src, nuint count)
    {
        _memcpyHook!.OriginalFunction(dst, src, count);

        // Large-copy tracking: heap-to-heap only, for BF script discovery.
        if (dst >= HeapLow && src >= HeapLow && count >= 500 && count <= 500_000)
        {
            lock (_largeCopyLock)
            {
                if (_largeCopyDsts.Count < 150)
                    _largeCopyDsts.Add(dst);
            }
        }

        // No small-copy text inspection here. The [MemcpyText] probe established that
        // dialogue never flows through this function — every capture that passed an
        // "English text" filter turned out to be an array of heap pointers whose 0x20
        // bytes happened to look like spaces. The renderer walks the BMD in place, so
        // the write target is the pool itself (see WritePoolStrings). Scanning bytes on
        // every memcpy the game makes was pure overhead in a very hot path.
    }

    private void TryActivateHook()
    {
        _logger!.WriteLine("[P5RGenSocialLinks] TryActivateHook: begin.");
        _logger!.WriteLine($"[P5RGenSocialLinks] IReloadedHooks available: {_hooks is not null}");

        try
        {
            using var scanner = new FunctionScanner();
            nuint funcAddr = scanner.FindOrThrow(Signatures.CmmExecEvent);
            _logger!.WriteLine($"[P5RGenSocialLinks] CmmExecEvent sig scan OK: 0x{funcAddr:X}");

            if (_hooks is null)
            {
                _logger!.WriteLine("[P5RGenSocialLinks] Hook skipped — IReloadedHooks null.");
                return;
            }

            _conversationHook = _hooks
                .CreateHook<CmmExecEventDelegate>(OnCmmExecEvent, (long)funcAddr)
                .Activate();

            _hookActive = true;
            _logger.WriteLine("[P5RGenSocialLinks] CmmExecEvent hook ACTIVE.");
        }
        catch (InvalidOperationException ex)
        {
            _logger!.WriteLine($"[P5RGenSocialLinks] Sig scan FAILED: {ex.Message}");
            _logger.WriteLine("[P5RGenSocialLinks] Falling back to poll loop only.");
        }
    }

    private nint OnCmmExecEvent()
    {
        nint result    = _conversationHook!.OriginalFunction();
        int  fireCount = Interlocked.Increment(ref _cmmExecFireCount);

        try
        {
            if (!_reader!.TryResolve(out nuint session))
            {
                _logger?.WriteLine($"[P5RGenSocialLinks] CmmExecEvent #{fireCount}: session chain unresolved.");
                return result;
            }

            SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
            if (snap is null) return result;

            _modLog!.Info(
                $"[P5RGenSocialLinks] CmmExec #{fireCount}: Confidant={snap.ConfidantId} Rank={snap.RankLevel} Scene={snap.SceneNumber}");

            // Retry text pool discovery on first few fires — the BF interpreter may not have
            // decompressed the script into heap memory at raw session-detection time, but it
            // is guaranteed to be loaded by the time CmmExecEvent fires for the first line.
            if (_bridge!.PoolBase == 0 && fireCount <= 5)
            {
                nuint pool = Memory.DialogueTextPoolFinder.Find(session, msg => _modLog!.Info(msg));
                if (pool != 0)
                {
                    _bridge!.SetPoolBase(pool);
                    _modLog!.Info($"[P5RGenSocialLinks] Text pool found on fire #{fireCount}: 0x{pool:X}");
                }
            }

            bool dispatched = _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap), lineIndex: fireCount);
            if (!dispatched)
                _modLog!.Info($"[P5RGenSocialLinks] CmmExec #{fireCount}: throttled.");
        }
        catch (Exception ex)
        {
            _logger?.WriteLine($"[P5RGenSocialLinks] OnCmmExecEvent error: {ex.Message}");
        }

        return result;
    }

    // ── Poll loop — session lifecycle + text pool discovery ───────────────

    private readonly StructDiffScanner _diffScanner = new();
    private uint   _lastBfPc;        // change-detection: fires only when PC moves (32-bit per mov [rbx+20],eax)
    private ushort _msgIdCandidate;  // current candidate from the last BF instruction window
    private int    _msgIdStreak;     // consecutive windows with the same candidate
    private ushort _currentMsgId;    // last confirmed msg_id (3+ consecutive windows)
    private nuint  _capturedSession; // session address when _currentMsgId was last confirmed
    private nuint  _confirmedBfBase; // real BF script base (session+0x18, memory-mapped <4 GB)
    private nuint  _bmdBase;         // BMD text table found by TryScanForBmd(); 0 until found
    private bool   _bmdScanDone;     // true after the one-shot scan fires this session
    private nuint  _currentMsgTextAddr; // session+0xD0 snapshot at msgId confirmation — direct write target

    /// <summary>
    /// Runs every poll tick while in a session. Scans the first 1024 bytes of the
    /// session struct for heap pointers whose targets contain ≥20 consecutive printable
    /// ASCII bytes AND are NOT self-referential (i.e., do not point back into the session
    /// struct itself — those are embedded metadata/vtable pointers, not BF script buffers).
    ///
    /// Among all qualifying candidates, the one with the most null-terminated strings of
    /// ≥4 printable chars is selected as the BF script buffer — because a real BF dialogue
    /// script has many null-separated lines, while embedded metadata has one or two.
    /// </summary>
    /// <summary>
    /// Returns true if <paramref name="ptr"/> looks like a BF dialogue script:
    /// has a long printable run AND at least 3 null-terminated strings that each
    /// contain ≥ 2 lowercase vowels (English sentences, not mesh/texture IDs).
    /// </summary>
    private static unsafe bool ProbeForBfContent(nuint ptr)
    {
        const int probeSize = 2048;
        if (!Memory.MemoryGuard.IsReadable(ptr, probeSize)) return false;
        byte* b = (byte*)ptr;

        int maxRun = 0, run = 0;
        for (int i = 0; i < probeSize; i++)
        {
            if (b[i] >= 0x20 && b[i] <= 0x7E) { if (++run > maxRun) maxRun = run; }
            else run = 0;
        }
        if (maxRun < 12) return false;

        // Require ≥ 3 null-terminated strings with ≥ 2 vowels each.
        // Mesh names ("mesh_920", "Ryuji_Hair") fail; English sentences pass.
        int goodStrings = 0, pos = 0;
        while (pos < probeSize)
        {
            int start = pos, printable = 0, vowels = 0;
            while (pos < probeSize && b[pos] != 0)
            {
                byte c = b[pos++];
                if (c >= 0x20 && c <= 0x7E) printable++;
                if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') vowels++;
            }
            int len = pos - start;
            if (len >= 4 && printable >= 3 && vowels >= 2) goodStrings++;
            if (goodStrings >= 3) return true;
            if (pos < probeSize) pos++; // skip '\0'
        }
        return false;
    }

    private unsafe void TryFindBfBuffer(nuint session)
    {
        if (_bfBufferBase != 0) return;   // already cached

        // Phase 0: probe large-copy destinations recorded by OnGameMemcpy.
        // The BF script is bulk-loaded (several KB) before session detection;
        // scan in reverse so the most-recent copy is tested first.
        nuint[] copySnapshot;
        lock (_largeCopyLock) copySnapshot = _largeCopyDsts.ToArray();

        for (int i = copySnapshot.Length - 1; i >= 0; i--)
        {
            nuint ptr = copySnapshot[i];
            if (!ProbeForBfContent(ptr)) continue;

            byte* b = (byte*)ptr;
            int strings = CountNullTermStrings(b, 2048, minPrintable: 4);
            var preview = new System.Text.StringBuilder(64);
            for (int j = 0; j < 512 && preview.Length < 64; j++)
                if (b[j] >= 0x20 && b[j] <= 0x7E) preview.Append((char)b[j]);

            _bfBufferBase = ptr;
            _bfBufferOff  = -1;
            _modLog!.Info($"[BFBuffer] Phase0 copy-log 0x{ptr:X} strings={strings}: \"{preview}\"");
            return;
        }

        // Phase 1: session-struct pointer scan (fallback — BF buffer is ~2 GB away
        // from the session struct, so this rarely succeeds, but costs little).
        // BF interpreter pointer may be at any offset in the session struct.
        // 4096 bytes = 512 pointer slots; cheap at 200ms poll interval.
        const int sessionScan = 4096;
        // 2048-byte probe: BF files start with a 32-byte binary header before
        // the first dialogue instruction — need more window to accumulate strings.
        const int probeScan   = 2048;
        // 12-char threshold: "gym over in Shibuya" = 19 chars, but the header
        // region can suppress long runs. 12 passes any English sentence and still
        // excludes pure-binary objects that rarely have 12 consecutive printable bytes.
        const int minRun      = 12;

        if (!Memory.MemoryGuard.IsReadable(session, sessionScan)) return;
        byte* sp = (byte*)session;

        nuint bestPtr     = 0;
        int   bestOff     = 0;
        int   bestStrings = -1;

        for (int off = 0; off + 8 <= sessionScan; off += 8)
        {
            nuint ptr = *(nuint*)(sp + off);
            if (ptr < HeapLow) continue;
            // Self-referential: points back into the session struct itself.
            // These are embedded sub-objects (ability descriptions, vtable pointers),
            // never a separately-allocated BF script buffer.
            if (ptr >= session && ptr < session + sessionScan) continue;
            if (!Memory.MemoryGuard.IsReadable(ptr, probeScan)) continue;
            byte* b = (byte*)ptr;

            int maxRun = 0, run = 0;
            for (int i = 0; i < probeScan; i++)
            {
                if (b[i] >= 0x20 && b[i] <= 0x7E) { if (++run > maxRun) maxRun = run; }
                else run = 0;
            }
            if (maxRun < minRun) continue;

            // Count null-terminated strings with ≥4 printable chars each.
            // BF dialogue script → many strings (one per line).
            // Embedded metadata  → one or two long strings.
            int strings = CountNullTermStrings(b, probeScan, minPrintable: 4);

            var preview = new System.Text.StringBuilder(64);
            for (int i = 0; i < probeScan && preview.Length < 64; i++)
                if (b[i] >= 0x20 && b[i] <= 0x7E) preview.Append((char)b[i]);

            _modLog!.Info(
                $"[BFBuffer] CAND sess+0x{off:X3} → 0x{ptr:X} (maxRun={maxRun} strings={strings}): \"{preview}\"");

            if (strings > bestStrings)
            {
                bestStrings = strings;
                bestPtr     = ptr;
                bestOff     = off;
            }
        }

        if (bestPtr == 0) return;

        _bfBufferBase = bestPtr;
        _bfBufferOff  = bestOff;
        _modLog!.Info($"[BFBuffer] SELECTED sess+0x{bestOff:X3} → 0x{bestPtr:X} (strings={bestStrings})");
    }

    /// <summary>
    /// Probes heap addresses captured in the StructDiff snapshot rather than live session
    /// memory. Catches transient pointers (e.g. BF script ptr at session+0x60) that are
    /// set and cleared within a single poll interval — Diff() captures them in _previous[]
    /// even after live memory has already been cleared.
    /// </summary>
    private unsafe void TryFindBfBufferFromSnapshot(nuint session)
    {
        if (_bfBufferBase != 0) return;
        const int probeScan   = 512;
        const int minRun      = 20;
        const int sessionScan = 1024;

        nuint bestPtr     = 0;
        int   bestOff     = 0;
        int   bestStrings = -1;

        foreach ((int off, nuint ptr) in _diffScanner.SnapshotHeapPointers(HeapLow))
        {
            if (ptr >= session && ptr < session + sessionScan) continue;
            if (!Memory.MemoryGuard.IsReadable(ptr, probeScan)) continue;
            byte* b = (byte*)ptr;

            int maxRun = 0, run = 0;
            for (int i = 0; i < probeScan; i++)
            {
                if (b[i] >= 0x20 && b[i] <= 0x7E) { if (++run > maxRun) maxRun = run; }
                else run = 0;
            }
            if (maxRun < minRun) continue;

            int strings = CountNullTermStrings(b, probeScan, minPrintable: 4);

            var preview = new System.Text.StringBuilder(64);
            for (int i = 0; i < probeScan && preview.Length < 64; i++)
                if (b[i] >= 0x20 && b[i] <= 0x7E) preview.Append((char)b[i]);

            _modLog!.Info(
                $"[BFBuffer] SNAP sess+0x{off:X3} → 0x{ptr:X} (maxRun={maxRun} strings={strings}): \"{preview}\"");

            if (strings > bestStrings) { bestStrings = strings; bestPtr = ptr; bestOff = off; }
        }

        if (bestPtr == 0) return;

        _bfBufferBase = bestPtr;
        _bfBufferOff  = bestOff;
        _modLog!.Info($"[BFBuffer] SNAP-SELECTED sess+0x{bestOff:X3} → 0x{bestPtr:X} (strings={bestStrings})");
    }

    // Scans ±512KB around bfBase for a page whose string content looks like dialogue.
    // Skips the first 0x100 bytes of each candidate (binary header). Counts strings
    // that are null-terminated, ≥10 printable ASCII chars, ≥2 spaces. Caches the
    // first page with ≥5 such strings as _bmdTextPool.
    private unsafe void TryScanBmdVicinity()
    {
        // Retry across ticks rather than one-shot: the BMD is mapped lazily and may not
        // be resident the first time bfBase is confirmed. 20 ticks ≈ 10s at the default
        // 500 ms interval, which comfortably covers the load.
        if (_bmdTextPool != 0 || _bmdScanDoneV2) return;
        if (++_bmdScanAttempts >= 20) _bmdScanDoneV2 = true;

        nuint center = _confirmedBfBase != 0 ? _confirmedBfBase : _bfBufferBase;
        if (center == 0) return;

        // Preferred path: locate the message script by its MSG1 magic. Content
        // heuristics alone cannot tell shop text from conversation — they scored the
        // shoe-shop and Velvet Room files above the scene's own dialogue.
        if (TryFindMessageScript(center)) return;

        // ±8 MB. The previous ±512KB window only reached compressed texture data;
        // the BMD for a conversation is mapped from the same archive as its BF but
        // not necessarily adjacent to it.
        const nuint window   = 0x800000;
        const int   pageSize = 0x1000;
        nuint start = center > window ? center - window : 0x10000;
        nuint end   = center + window;

        nuint bfPage = center & ~(nuint)0xFFF; // page holding the BF script — skip it

        // Rank every page and take the best, rather than the first page over a
        // threshold. First-hit locked onto DXT texture data whose 0x20 bytes read as
        // spaces; the real dialogue page scores far higher under IsEnglishSentence.
        var ranked = new System.Collections.Generic.List<(int Score, nuint Addr)>();

        for (nuint addr = start; addr < end; addr += (nuint)pageSize)
        {
            if (addr == bfPage) continue;
            if (!MemoryGuard.IsReadable(addr, pageSize)) continue;

            int score = CountEnglishSentences((byte*)addr, pageSize);
            if (score >= 3) ranked.Add((score, addr));
        }

        if (ranked.Count == 0)
        {
            if (_bmdScanDoneV2)
                _modLog!.Info($"[BMD2] No English text page in 0x{start:X}–0x{end:X} after {_bmdScanAttempts} attempts");
            return;
        }

        ranked.Sort((a, b) => b.Score.CompareTo(a.Score));

        // Log the top few so a wrong pick is visible in the log instead of silent.
        for (int i = 0; i < Math.Min(5, ranked.Count); i++)
        {
            (int sc, nuint pa) = ranked[i];
            _modLog!.Info($"[BMD2] cand#{i} 0x{pa:X} score={sc}: \"{PreviewSentences(pa, pageSize, 90)}\"");
        }

        nuint best = ranked[0].Addr;
        _bmdTextPool = best;
        _bmdPoolLen  = pageSize;
        _poolSlots   = CapturePoolSlots(best, pageSize);
        _modLog!.Info($"[BMD2] TextPool SELECTED 0x{best:X} score={ranked[0].Score} slots={_poolSlots.Length}");

        // If the LLM already answered while we were still hunting for the pool,
        // apply that text now rather than waiting for the next message.
        string? pending = _lastLlmText;
        if (pending != null) WritePoolStrings(pending);
    }

    /// <summary>
    /// Locates the Atlus MessageScript that belongs to the running BF by scanning for
    /// the "MSG1" magic. In P5R the message script is normally embedded in the .bf
    /// flowscript, so the first valid header at or after bfBase is this scene's dialogue.
    ///
    /// MessageScript binary header (32 bytes):
    ///   +0x00 fileType(1) format(1) userId(2)
    ///   +0x04 fileSize(4)
    ///   +0x08 magic "MSG1"
    ///   +0x0C extSize(4)
    ///   +0x10 relocationTable(4)   +0x14 relocationTableSize(4)
    ///   +0x18 messageCount(4)
    ///   +0x1C isRelocated(2) version(2)
    /// The magic sits at +0x08, so a hit implies a header start 8 bytes earlier.
    /// </summary>
    private unsafe bool TryFindMessageScript(nuint center)
    {
        const nuint back = 0x100000; // 1 MB before bfBase
        const nuint fwd  = 0x400000; // 4 MB after — embedded MSG1 trails the BF code
        nuint start = center > back ? center - back : 0x10000;
        nuint end   = center + fwd;

        var hits = new System.Collections.Generic.List<(nuint Base, uint Size, uint Count, int Score)>();

        // Walk committed regions rather than probing every page: VirtualQuery skips
        // whole unmapped spans in one call, so 5 MB costs a handful of syscalls.
        nuint addr = start;
        while (addr < end && hits.Count < 32)
        {
            var (ok, regionBase, regionSize, state, protect) = MemoryGuard.QueryRegion(addr);
            if (!ok) break;
            nuint regionEnd = regionBase + regionSize;
            if (regionEnd <= addr) break;

            const uint MEM_COMMIT = 0x1000, PAGE_NOACCESS = 0x01, PAGE_GUARD = 0x100;
            bool usable = state == MEM_COMMIT
                          && (protect & PAGE_NOACCESS) == 0
                          && (protect & PAGE_GUARD) == 0;

            if (usable)
            {
                nuint scanFrom = addr > regionBase ? addr : regionBase;
                nuint scanTo   = regionEnd < end ? regionEnd : end;

                for (nuint q = scanFrom; q + 4 < scanTo; q++)
                {
                    byte* m = (byte*)q;
                    if (m[0] != (byte)'M' || m[1] != (byte)'S' ||
                        m[2] != (byte)'G' || m[3] != (byte)'1') continue;
                    if (q < 8) continue;

                    nuint hdr = q - 8;
                    if (!MemoryGuard.IsReadable(hdr, 0x20)) continue;

                    uint fileSize = *(uint*)(hdr + 4);
                    uint msgCount = *(uint*)(hdr + 0x18);
                    if (fileSize < 0x40 || fileSize > 0x400000) continue;
                    if (msgCount == 0 || msgCount > 20000)      continue;

                    int usableLen = ReadableLen(hdr, (int)Math.Min(fileSize, 0x40000u));
                    if (usableLen < 0x100) continue;

                    int score = CountEnglishSentences((byte*)hdr, usableLen);
                    hits.Add((hdr, fileSize, msgCount, score));

                    // Only list files that could actually hold the live index — the full
                    // 21-file listing is noise now that these are known to be the global
                    // item/skill tables rather than scene dialogue.
                    if (_cfg.StructDiffEnabled || _currentMsgId == 0 || msgCount > _currentMsgId)
                        _modLog!.Info(
                            $"[MSG1] 0x{hdr:X} size={fileSize} msgs={msgCount} score={score} " +
                            $"{(hdr >= center ? "after" : "before")}bf: \"{PreviewSentences(hdr, usableLen, 80)}\"");
                }
            }
            addr = regionEnd;
        }

        if (hits.Count == 0) return false;

        // Selection is driven by messageCount, not proximity to bfBase. The running
        // script indexes message 0x348 (840), so any file declaring fewer entries than
        // that cannot be the one being read — in the observed run exactly one of 21
        // files qualified. Among qualifying files take the tightest fit: the smallest
        // count that still covers the index, since a scene script sized just past the
        // messages it uses is far likelier than a huge global table.
        (nuint Base, uint Size, uint Count, int Score) pick = default;
        bool found = false;

        if (_currentMsgId != 0)
        {
            foreach (var h in hits)
            {
                if (h.Count <= _currentMsgId || h.Score <= 0) continue;
                if (!found || h.Count < pick.Count) { pick = h; found = true; }
            }
        }

        // No msgId yet, or nothing covers it: fall back to the most English-dense file.
        if (!found)
            foreach (var h in hits)
                if (h.Score > 0 && (!found || h.Score > pick.Score)) { pick = h; found = true; }

        if (!found) return false;

        _bmdPoolLen  = ReadableLen(pick.Base, (int)Math.Min(pick.Size, 0x40000u));
        _bmdTextPool = pick.Base;
        _poolSlots   = CapturePoolSlots(pick.Base, _bmdPoolLen);
        _modLog!.Info(
            $"[MSG1] SELECTED 0x{pick.Base:X} size={pick.Size} msgs={pick.Count} " +
            $"slots={_poolSlots.Length} (msgId=0x{_currentMsgId:X})");

        DumpMessageTableEntry(pick.Base, pick.Count, _bmdPoolLen);

        string? pending = _lastLlmText;
        if (pending != null) WritePoolStrings(pending);
        return true;
    }

    /// <summary>
    /// Dumps the message-table entry for the current msgId so the addressing convention
    /// can be confirmed from a real run rather than assumed. The table follows the 32-byte
    /// header as (kind:int, offset:int) pairs. What is not yet established is what the
    /// offset is relative to — the file start or the end of the table — and whether the
    /// loader has already relocated it to an absolute pointer (the header carries an
    /// isRelocated flag at +0x1C). Both candidate bases are dumped for comparison.
    /// </summary>
    private unsafe void DumpMessageTableEntry(nuint fileBase, uint msgCount, int usableLen)
    {
        if (_currentMsgId == 0 || _currentMsgId >= msgCount) return;

        nuint entry = fileBase + 0x20 + (nuint)(_currentMsgId * 8);
        if (!MemoryGuard.IsReadable(entry, 8)) return;

        uint kind   = *(uint*)entry;
        uint offset = *(uint*)(entry + 4);
        _modLog!.Info($"[MSG1] entry[{_currentMsgId}] kind={kind} offset=0x{offset:X}");

        nuint tableEnd = fileBase + 0x20 + (nuint)(msgCount * 8);
        DumpAt("fileBase+off", fileBase + offset, usableLen, fileBase);
        DumpAt("tableEnd+off", tableEnd + offset, usableLen, fileBase);

        void DumpAt(string label, nuint at, int limit, nuint origin)
        {
            if (at < origin || at - origin >= (nuint)limit) return;
            if (!MemoryGuard.IsReadable(at, 64)) return;
            byte* q = (byte*)at;
            var sb = new System.Text.StringBuilder($"[MSG1] {label} 0x{at:X}: ");
            for (int i = 0; i < 48; i++) sb.Append($"{q[i]:X2} ");
            sb.Append(" \"");
            for (int i = 0; i < 48; i++) sb.Append(IsPrintable(q[i]) ? (char)q[i] : '·');
            sb.Append('"');
            _modLog!.Info(sb.ToString());
        }
    }

    /// <summary>
    /// Largest length up to <paramref name="want"/> that is fully readable from
    /// <paramref name="addr"/>, resolved in page steps. A mapped file's tail pages may
    /// not be resident, so committing to the header's declared size would fail the read.
    /// </summary>
    private static int ReadableLen(nuint addr, int want)
    {
        const int page = 0x1000;
        if (MemoryGuard.IsReadable(addr, want)) return want;
        int len = 0;
        while (len + page <= want && MemoryGuard.IsReadable(addr, len + page)) len += page;
        return len;
    }

    /// <summary>
    /// Same ratio test as <see cref="IsEnglishSentence"/> but over a decoded string, so
    /// UTF-16 runs can reuse it without duplicating the pointer walk.
    /// </summary>
    private static bool IsEnglishString(string s)
    {
        if (s.Length < 10 || s.Length > 400) return false;
        int letters = 0, lower = 0, vowels = 0, digits = 0, spaces = 0, other = 0;
        foreach (char ch in s)
        {
            if (ch >= 'a' && ch <= 'z')      { letters++; lower++; if (IsVowel((byte)ch)) vowels++; }
            else if (ch >= 'A' && ch <= 'Z') { letters++; if (IsVowel((byte)(ch | 0x20))) vowels++; }
            else if (ch == ' ')              spaces++;
            else if (ch >= '0' && ch <= '9') digits++;
            else if (ch == '.' || ch == ',' || ch == '!' || ch == '?' || ch == '\'' ||
                     ch == '"' || ch == '-' || ch == ':' || ch == ';') { }
            else other++;
        }
        if (spaces < 1)                       return false;
        if (letters * 100 < s.Length * 55)    return false;
        if (lower   * 100 < letters * 40)     return false;
        if (vowels  * 100 < letters * 25)     return false;
        if (vowels  * 100 > letters * 60)     return false;
        if (digits  * 100 > s.Length * 10)    return false;
        if (other   * 100 > s.Length * 5)     return false;
        return true;
    }

    /// <summary>
    /// Finds UTF-16LE English runs — printable ASCII in even bytes, 0x00 in odd bytes.
    /// The session's message object holds "52 00 4F 00 52 00" (UTF-16 "ROR"), so the
    /// live dialogue is very likely wide-char. Every scanner before this was single-byte
    /// and would step over UTF-16 text without ever seeing a candidate: the interleaved
    /// nulls break each run after one character.
    /// </summary>
    private static unsafe System.Collections.Generic.List<(nuint Addr, string Text)>
        FindUtf16English(nuint region, int byteLen, int maxHits)
    {
        var hits = new System.Collections.Generic.List<(nuint, string)>();
        byte* p = (byte*)region;
        int i = 0;

        while (i + 1 < byteLen && hits.Count < maxHits)
        {
            if (!(IsPrintable(p[i]) && p[i + 1] == 0)) { i++; continue; }

            int begin = i;
            var sb = new System.Text.StringBuilder(64);
            while (i + 1 < byteLen && IsPrintable(p[i]) && p[i + 1] == 0 && sb.Length < 400)
            {
                sb.Append((char)p[i]);
                i += 2;
            }
            string s = sb.ToString();
            if (IsEnglishString(s)) hits.Add((region + (nuint)begin, s));
        }
        return hits;
    }

    private static unsafe string AsciiPreview(nuint addr, int maxChars)
    {
        if (!MemoryGuard.IsReadable(addr, maxChars)) return "";
        byte* p = (byte*)addr;
        int begin = 0;
        while (begin < maxChars && !IsPrintable(p[begin])) begin++;
        int end = begin;
        while (end < maxChars && IsPrintable(p[end])) end++;
        if (end - begin < 8) return "";
        var sb = new System.Text.StringBuilder(end - begin);
        for (int i = begin; i < end; i++) sb.Append((char)p[i]);
        return IsEnglishString(sb.ToString()) ? sb.ToString() : "";
    }

    /// <summary>
    /// Follows the message object at session+0xC8 — populated exactly while a message is
    /// on screen — dumping every heap pointer it holds and testing each target for both
    /// ASCII and UTF-16 text, then sweeping the heap around the object itself for UTF-16
    /// dialogue. This replaces guessing at pool addresses: the object is the game's own
    /// handle on the live message.
    /// </summary>
    private unsafe void ProbeMessageObject(nuint session)
    {
        if (!MemoryGuard.IsReadable(session + 0xC8, 8)) return;
        nuint obj = *(nuint*)(session + 0xC8);
        if (obj < HeapLow || obj > UserAddrMax) return;
        if (!MemoryGuard.IsReadable(obj, 0x100)) return;

        byte* o = (byte*)obj;
        for (int off = 0; off + 8 <= 0x100; off += 8)
        {
            nuint val = *(nuint*)(o + off);
            if (val < HeapLow || val > UserAddrMax) continue;
            if (!MemoryGuard.IsReadable(val, 256)) continue;

            string ascii = AsciiPreview(val, 96);
            var wide = FindUtf16English(val, 256, 2);
            if (ascii.Length == 0 && wide.Count == 0) continue;

            _modLog!.Info($"[OBJ] +0x{off:X2}→0x{val:X} ascii=\"{ascii}\" " +
                          $"utf16=\"{(wide.Count > 0 ? wide[0].Text : "")}\"");
        }

        // Sweep ±64 KB around the object for UTF-16 English, page by page so an
        // unmapped page in the middle does not abort the whole sweep.
        const int win = 0x10000, page = 0x1000;
        nuint from = obj > (nuint)win ? obj - win : obj;
        int reported = 0;
        for (nuint a = from; a < obj + win && reported < 8; a += page)
        {
            if (!MemoryGuard.IsReadable(a, page)) continue;
            foreach (var (addr, text) in FindUtf16English(a, page, 4))
            {
                _modLog!.Info($"[UTF16] 0x{addr:X}: \"{text}\"");
                if (++reported >= 8) break;
            }
        }
        if (reported == 0)
            _modLog!.Info($"[UTF16] none within ±64KB of msgObj 0x{obj:X}");
    }

    private static bool IsVowel(byte c) =>
        c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u';

    /// <summary>
    /// True only for a byte run that reads as an English sentence. The earlier
    /// "≥10 printable, ≥2 spaces" test passed compressed texture data — binary is full
    /// of 0x20 bytes. These ratios encode what actually separates prose from binary:
    /// mostly letters, mostly lowercase, a plausible vowel share, few digits, almost
    /// no symbol junk.
    /// </summary>
    private static unsafe bool IsEnglishSentence(byte* p, int len)
    {
        if (len < 12 || len > 400) return false;

        int letters = 0, lower = 0, vowels = 0, digits = 0, spaces = 0, other = 0;
        for (int i = 0; i < len; i++)
        {
            byte c = p[i];
            if (c >= 'a' && c <= 'z')      { letters++; lower++; if (IsVowel(c)) vowels++; }
            else if (c >= 'A' && c <= 'Z') { letters++; if (IsVowel((byte)(c | 0x20))) vowels++; }
            else if (c == ' ')             spaces++;
            else if (c >= '0' && c <= '9') digits++;
            else if (c == '.' || c == ',' || c == '!' || c == '?' || c == '\'' ||
                     c == '"' || c == '-' || c == ':' || c == ';') { /* sentence punctuation */ }
            else other++;
        }

        if (spaces < 2)                     return false;
        if (letters * 100 < len * 55)       return false; // ≥55% letters
        if (lower   * 100 < letters * 40)   return false; // ≥40% of letters lowercase
        if (vowels  * 100 < letters * 25)   return false; // English runs ~38% vowels
        if (vowels  * 100 > letters * 60)   return false;
        if (digits  * 100 > len * 10)       return false; // ≤10% digits
        if (other   * 100 > len * 5)        return false; // ≤5% symbol junk
        return true;
    }

    private static bool IsPrintable(byte c) => c >= 0x20 && c <= 0x7E;

    /// <summary>
    /// Counts maximal runs of printable ASCII that read as English. Runs are delimited by
    /// ANY non-printable byte, not just NUL. BMD text carries embedded function codes —
    /// voice cues, speaker-name substitution, line breaks — as control and high bytes, so
    /// walking only to the next NUL yielded one enormous run per file, which failed the
    /// len>400 check and scored every message file zero.
    /// </summary>
    private static unsafe int CountEnglishSentences(byte* buf, int maxBytes)
    {
        int count = 0, pos = 0;
        while (pos < maxBytes)
        {
            while (pos < maxBytes && !IsPrintable(buf[pos])) pos++;
            int begin = pos;
            while (pos < maxBytes && IsPrintable(buf[pos])) pos++;
            if (pos > begin && IsEnglishSentence(buf + begin, pos - begin)) count++;
        }
        return count;
    }

    private static unsafe string PreviewSentences(nuint page, int maxBytes, int maxChars)
    {
        var sb = new System.Text.StringBuilder(maxChars + 8);
        byte* buf = (byte*)page;
        int pos = 0;
        while (pos < maxBytes && sb.Length < maxChars)
        {
            while (pos < maxBytes && !IsPrintable(buf[pos])) pos++;
            int begin = pos;
            while (pos < maxBytes && IsPrintable(buf[pos])) pos++;
            if (pos > begin && IsEnglishSentence(buf + begin, pos - begin))
            {
                if (sb.Length > 0) sb.Append(" | ");
                for (int i = begin; i < pos && sb.Length < maxChars; i++) sb.Append((char)buf[i]);
            }
        }
        return sb.ToString();
    }

    // Records (offset, original length) for every English entry in the pool.
    // Captured once, before any write, so every later write measures against the
    // ORIGINAL entry lengths. Without this, each pass would re-measure the shortened
    // string it wrote last time and the usable space would ratchet down to nothing.
    private static unsafe (int Off, int Len)[] CapturePoolSlots(nuint poolBase, int scanLen)
    {
        var slots = new System.Collections.Generic.List<(int, int)>();
        byte* p = (byte*)poolBase;
        int pos = 0;

        // 512, not 64: a scene's message script holds every line of the conversation.
        // Runs are delimited by any non-printable byte — see CountEnglishSentences.
        while (pos < scanLen && slots.Count < 512)
        {
            while (pos < scanLen && !IsPrintable(p[pos])) pos++;
            int begin = pos;
            while (pos < scanLen && IsPrintable(p[pos])) pos++;
            if (pos > begin && IsEnglishSentence(p + begin, pos - begin))
                slots.Add((begin, pos - begin));
        }
        return slots.ToArray();
    }

    // Overwrites every captured dialogue slot in the BMD text pool with <paramref name="text"/>.
    // Writes at most (origLen - 1) bytes per slot and re-terminates in place, so entry
    // boundaries — and therefore the BMD's internal offset table — stay valid; the renderer
    // still finds each entry exactly where it expects, just with our characters in it.
    // Mapped-file pages are PAGE_READONLY, so the page is first upgraded to PAGE_WRITECOPY:
    // the OS hands back a private copy and the .bmd on disk is never modified.
    private unsafe int WritePoolStrings(string text)
    {
        if (_bmdTextPool == 0 || _poolSlots is null || _poolSlots.Length == 0) return 0;

        const uint PAGE_WRITECOPY = 0x08;
        nuint pageBase = _bmdTextPool;

        // VirtualProtect rounds the range out to page boundaries, so an unaligned base
        // is fine — but the length must cover the whole pool, which for an MSG1 file
        // spans many pages.
        if (!MemoryGuard.IsWritable(_bmdTextPool, 16) &&
            !MemoryGuard.VirtualProtect(pageBase, (nuint)_bmdPoolLen, PAGE_WRITECOPY, out _))
        {
            _modLog!.Warn($"[BMD2] VirtualProtect WRITECOPY failed for page 0x{pageBase:X}");
            return 0;
        }

        byte[] enc = System.Text.Encoding.ASCII.GetBytes(text);
        byte*  p   = (byte*)_bmdTextPool;
        int written = 0;

        foreach ((int off, int len) in _poolSlots)
        {
            int wl = Math.Min(enc.Length, len);
            if (wl <= 0) continue;
            fixed (byte* src = enc)
                System.Buffer.MemoryCopy(src, p + off, len, wl);
            // Pad with spaces rather than writing a NUL. The run is delimited by the
            // surrounding control bytes (function codes, line breaks) which the message
            // parser relies on — injecting a terminator inside the run would truncate
            // the entry as far as the parser is concerned and could desync the page.
            for (int i = wl; i < len; i++) p[off + i] = (byte)' ';
            written++;
        }

        _modLog!.Info(
            $"[BMD2] Pool write: {written}/{_poolSlots.Length} slots at 0x{_bmdTextPool:X} " +
            $"← \"{text[..Math.Min(text.Length, 60)]}\"");
        return written;
    }

    private static unsafe int CountNullTermStrings(byte* buf, int maxBytes, int minPrintable)
    {
        int count = 0, pos = 0;
        while (pos < maxBytes)
        {
            int start = pos, printable = 0;
            while (pos < maxBytes && buf[pos] != 0)
            {
                if (buf[pos] >= 0x20 && buf[pos] <= 0x7E) printable++;
                pos++;
            }
            int len = pos - start;
            if (len > 0 && printable >= minPrintable) count++;
            if (pos >= maxBytes) break;
            pos++; // skip '\0'
        }
        return count;
    }

    /// <summary>
    /// Fires on every BF PC change. Reads 32 bytes at bfBase+pc (session+0x18 base,
    /// session+0x20 offset). Extracts msg_id by taking the last LE uint16 in
    /// [0x0200, 0x07FF] — sub-line indices and opcodes are smaller, unrelated code
    /// values are larger. Requires 3 consecutive windows with the same value before
    /// confirming, eliminating one-off code values as false positives.
    /// When a new msg_id is confirmed, dispatches an LLM request and writes the
    /// response to session+0x9B0 (social-link description slot, confirmed writable).
    /// </summary>
    private unsafe void ProbeBfLine(nuint session)
    {
        if (!Memory.MemoryGuard.IsReadable(session + 0x18, 12)) return;
        uint pc = *(uint*)((byte*)session + 0x20);
        if (pc == _lastBfPc) return;
        _lastBfPc = pc;

        nuint bfBase = *(nuint*)((byte*)session + 0x18);
        if (bfBase == 0) return;

        nuint instrAddr = bfBase + pc;
        const int readLen = 32;
        if (!Memory.MemoryGuard.IsReadable(instrAddr, readLen)) return;

        byte* b = (byte*)instrAddr;

        // Verbose raw-bytes dump — gated so it doesn't spam the log by default.
        if (_cfg.StructDiffEnabled)
        {
            var hex = new System.Text.StringBuilder(96);
            for (int i = 0; i < readLen; i++) hex.Append($"{b[i]:X2} ");
            _modLog!.Info($"[BFInstr] pc=0x{pc:X}: {hex}");
        }

        // Scan for msg_id: take the LAST LE uint16 in [0x0200, 0x07FF].
        // Sub-line indices (3,6,9 → 0x0003-0x0009) and opcodes (0x09, 0x0A → <0x0200)
        // are below the band; large code constants are above it.
        ushort best = 0;
        for (int i = 0; i + 1 < readLen; i++)
        {
            ushort v = (ushort)(b[i] | (b[i + 1] << 8));
            if (v >= 0x0200 && v <= 0x07FF) best = v;
        }

        // Require 3 consecutive windows with the same value before confirming.
        if (best == 0)
        {
            _msgIdStreak = 0;
        }
        else if (best == _msgIdCandidate)
        {
            _msgIdStreak++;
            if (_msgIdStreak == 3 && best != _currentMsgId)
            {
                _currentMsgId    = best;
                _capturedSession = session;
                if (_confirmedBfBase == 0) _confirmedBfBase = bfBase;

                // session+0xD0 → descriptor struct → descriptor+0x18 → actual text bytes.
                // TryReadTextAddr follows all three hops; returns 0 if any link is not yet set.
                nuint capturedTextAddr = TryReadTextAddr(session);

                if (capturedTextAddr != 0)
                {
                    // Dump all three chain levels to verify layout.
                    unsafe
                    {
                        nuint descriptor = *(nuint*)((byte*)session + 0xD0);
                        nuint textObj    = (descriptor != 0 && MemoryGuard.IsReadable(descriptor + 0x18, 8))
                                           ? *(nuint*)(descriptor + 0x18) : 0;

                        // Level 2: textObj bytes (the string-wrapper struct)
                        if (textObj != 0 && MemoryGuard.IsReadable(textObj, 32))
                        {
                            byte* ob = (byte*)textObj;
                            var objSb = new System.Text.StringBuilder($"[MSG] TextObj(0x{textObj:X}): ");
                            for (int di = 0; di < 32; di++)
                            {
                                if (di > 0 && di % 16 == 0) objSb.Append(" | ");
                                objSb.Append($"{ob[di]:X2} ");
                            }
                            _modLog!.Info(objSb.ToString());
                        }

                        // Level 3: actual character bytes (capturedTextAddr = *(textObj))
                        if (MemoryGuard.IsReadable(capturedTextAddr, 64))
                        {
                            byte* tb = (byte*)capturedTextAddr;
                            var txtSb = new System.Text.StringBuilder($"[MSG] CharBytes(0x{capturedTextAddr:X}): ");
                            for (int di = 0; di < 64; di++)
                            {
                                if (di > 0 && di % 16 == 0) txtSb.Append(" | ");
                                txtSb.Append($"{tb[di]:X2} ");
                            }
                            _modLog!.Info(txtSb.ToString());
                        }
                    }
                }
                else
                {
                    // textAddr=0: dump readable/unreadable status for offsets around the
                    // known boundary (+0xB8..+0xE0) to pinpoint the struct end,
                    // plus all external heap pointers in [0x00..0xC8) for fallback probing.
                    unsafe
                    {
                        // Boundary scan: individual IsReadable per slot
                        var bndSb = new System.Text.StringBuilder("[MSG] BndScan: ");
                        for (int di = 0xB8; di <= 0xE0; di += 8)
                        {
                            nuint addr = session + (nuint)di;
                            if (MemoryGuard.IsReadable(addr, 8))
                            {
                                nuint val = *(nuint*)(byte*)addr;
                                bndSb.Append($"+0x{di:X2}=0x{val:X} ");
                            }
                            else
                            {
                                bndSb.Append($"+0x{di:X2}=NR ");
                            }
                        }
                        _modLog!.Info(bndSb.ToString());

                        // Dump raw bytes at the session+0xD0 value when it's a low-memory
                        // (mapped-file) address — shows whether it's BMD text or BF script.
                        if (MemoryGuard.IsReadable(session + 0xD0, 8))
                        {
                            nuint d0val = *(nuint*)((byte*)session + 0xD0);
                            if (d0val >= 0x1000 && d0val < HeapLow && MemoryGuard.IsReadable(d0val, 48))
                            {
                                byte* tb = (byte*)d0val;
                                var dSb = new System.Text.StringBuilder($"[MSG] D0Direct(0x{d0val:X}): ");
                                for (int di = 0; di < 48; di++) dSb.Append($"{tb[di]:X2} ");
                                _modLog!.Info(dSb.ToString());
                            }
                        }

                        // Heap pointer scan over [0x00..0x100) per slot — catches ptrs
                        // beyond VirtualQuery region boundary (e.g. session+0xE0).
                        var ptrSb = new System.Text.StringBuilder("[MSG] SessHeapPtrs: ");
                        for (int di = 0; di < 0x100; di += 8)
                        {
                            nuint slotAddr = session + (nuint)di;
                            if (!MemoryGuard.IsReadable(slotAddr, 8)) continue;
                            nuint val = *(nuint*)(byte*)slotAddr;
                            if (val >= HeapLow && val <= UserAddrMax &&
                                !(val >= session && val < session + 0x1000))
                                ptrSb.Append($"+0x{di:X2}→0x{val:X} ");
                        }
                        _modLog!.Info(ptrSb.ToString());

                        // session+0xC8 is the live message object — follow it and sweep
                        // for UTF-16, which every earlier scanner was blind to.
                        ProbeMessageObject(session);

                        // Dump first 48 bytes of each external heap target — shows structure
                        // layout. Gated: this emits a line per pointer and the session-chain
                        // approach it was built for is no longer the primary strategy.
                        for (int di = 0; _cfg.StructDiffEnabled && di < 0x100; di += 8)
                        {
                            nuint slotAddr = session + (nuint)di;
                            if (!MemoryGuard.IsReadable(slotAddr, 8)) continue;
                            nuint val = *(nuint*)(byte*)slotAddr;
                            if (val < HeapLow || val > UserAddrMax) continue;
                            if (val >= session && val < session + 0x1000) continue;
                            if (!MemoryGuard.IsReadable(val, 48)) continue;
                            byte* ep = (byte*)val;
                            var epSb = new System.Text.StringBuilder($"[MSG] ExtObj+0x{di:X2}(0x{val:X}): ");
                            for (int ei = 0; ei < 48; ei++) epSb.Append($"{ep[ei]:X2} ");
                            _modLog!.Info(epSb.ToString());
                        }
                    }
                }
                _currentMsgTextAddr = capturedTextAddr;

                _modLog!.Info($"[MSG] pc=0x{pc:X} msgId=0x{best:X} ({best}) bfBase=0x{bfBase:X} textAddr=0x{_currentMsgTextAddr:X}");

                // Fire-and-forget LLM call; write result to session+0x9B0 on response.
                // Cannot await inside unsafe — call the async method and discard the Task.
                SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
                if (snap is not null)
                {
                    nuint  capturedSess = session;
                    ushort capturedId   = best;
                    _ = DispatchMsgLlmAsync(snap, capturedId, capturedSess);
                }
            }
        }
        else
        {
            _msgIdCandidate = best;
            _msgIdStreak    = 1;
        }
    }

    // ── BMD scanner ───────────────────────────────────────────────────────

    // Scans the entire lower-4GB mapped-file region for a committed, readable
    // region containing ≥5 English dialogue sentences (the BMD string table).
    // BMD is heap-allocated; found via session struct pointer map:
    //   session+0xC8 = BMD base address (heap, above 4 GB, not caught by lower-4GB scan)
    //   session+0xD0 = pointer to current message text inside BMD
    //
    // Confirmed: session+0xD0 - session+0xC8 = byte offset of current msgId's text.
    // Header magic is [20 00 0A 00], NOT [0D 00] — that's why all previous scans missed it.
    //
    // This method locks _bmdBase from session+0xC8, dumps the header, and
    // reverse-scans the first 8 KB for the current-message offset so we can
    // identify the offset table location and entry size.
    private unsafe void TryScanForBmd()
    {
        nuint session = _capturedSession;
        if (session == 0) return;

        if (!MemoryGuard.IsReadable(session + 0xC8, 16))
        {
            _modLog!.Warn($"[BMD] session+0xC8 not readable");
            return;
        }

        nuint bmdBase = *(nuint*)((byte*)session + 0xC8);
        nuint msgPtr  = *(nuint*)((byte*)session + 0xD0);

        if (!MemoryGuard.IsReadable(bmdBase, 64))
        {
            _modLog!.Warn($"[BMD] bmdBase=0x{bmdBase:X} (session+0xC8) not readable");
            return;
        }

        _bmdBase = bmdBase;
        _modLog!.Info($"[BMD] base=0x{bmdBase:X} msgPtr=0x{msgPtr:X} msgId={_currentMsgId}");

        // Dump first 64 bytes of the BMD header
        byte* hdr = (byte*)bmdBase;
        var hexSb = new System.Text.StringBuilder("[BMD] Header: ");
        for (int i = 0; i < 64; i++)
        {
            if (i > 0 && i % 16 == 0) hexSb.Append(" | ");
            hexSb.Append($"{hdr[i]:X2} ");
        }
        _modLog!.Info(hexSb.ToString());

        // Compute confirmed offset for the current msgId
        if (msgPtr <= bmdBase) return;
        nuint currentOff = msgPtr - bmdBase;
        _modLog!.Info($"[BMD] msgId={_currentMsgId} confirmed offset=0x{currentOff:X}");

        // Reverse scan first 8 KB for the offset value — tells us the table base and entry size
        uint   t32 = (uint)currentOff;
        ushort t16 = (ushort)currentOff;
        for (int pos = 0; pos < 8192 - 1; pos++)
        {
            if (!MemoryGuard.IsReadable(bmdBase + (nuint)pos, 4)) break;
            byte* sp = (byte*)(bmdBase + (nuint)pos);
            ushort v16 = (ushort)(sp[0] | sp[1] << 8);
            if (v16 == t16)
                _modLog!.Info($"[BMD] offset as uint16 @ bmd+0x{pos:X}");
            if (pos + 3 < 8192)
            {
                uint v32 = (uint)(sp[0] | sp[1]<<8 | sp[2]<<16 | sp[3]<<24);
                if (v32 == t32)
                    _modLog!.Info($"[BMD] offset as uint32 @ bmd+0x{pos:X}");
            }
        }
    }

    private static unsafe int CountPrintableRuns(nuint regionBase, int scanBytes)
    {
        byte* p = (byte*)regionBase;
        int runs = 0;
        int i = 0;

        while (i < scanBytes)
        {
            byte c = p[i];
            if (c < 0x20 || c >= 0x7F) { i++; continue; }

            int start = i;
            int letters = 0, spaces = 0;
            while (i < scanBytes)
            {
                c = p[i];
                if (c < 0x20 || c >= 0x7F) break;
                if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) letters++;
                if (c == ' ') spaces++;
                i++;
            }
            int runLen = i - start;
            if (runLen >= 12 && spaces >= 1 && letters * 10 >= runLen * 6)
                runs++;
        }

        return runs;
    }

    private unsafe void TryLogBmdStrings()
    {
        const int MaxScan  = 16384;
        const int MaxPrint = 30;

        byte* p = (byte*)_bmdBase;
        int count = 0;
        int i = 0;

        while (i < MaxScan && count < MaxPrint)
        {
            if (p[i] == 0) { i++; continue; }

            int start = i;
            while (i < MaxScan && p[i] != 0) i++;
            int len = i - start;
            if (len < 4) continue;

            var sb = new System.Text.StringBuilder(Math.Min(len, 120) + 4);
            int printEnd = Math.Min(i, start + 120);
            for (int j = start; j < printEnd; j++)
            {
                byte c = p[j];
                sb.Append(c >= 0x20 && c < 0x7F ? (char)c : '.');
            }
            if (len > 120) sb.Append('…');

            _modLog!.Info($"[BMD][{count:D3}] +0x{start:X4} len={len}: \"{sb}\"");
            count++;
        }

        _modLog!.Info($"[BMD] Logged {count} strings from 0x{_bmdBase:X}.");

        // Reverse-scan: find WHERE in the BMD the known string offsets (0x0120, 0x013E, 0x0158)
        // are stored, to identify the offset table location and entry size.
        nuint bmdBase = _bmdBase;
        uint[] knownOffsets = { 0x0120u, 0x013Eu, 0x0158u };
        for (int pos = 4; pos < 1024; pos++)
        {
            if (!Memory.MemoryGuard.IsReadable(bmdBase + (nuint)pos, 4)) break;
            byte* sp = (byte*)(bmdBase + (nuint)pos);

            uint v32 = (uint)(sp[0] | (sp[1] << 8) | (sp[2] << 16) | (sp[3] << 24));
            foreach (uint ko in knownOffsets)
                if (v32 == ko)
                    _modLog!.Info($"[BMD] OffsetScan: known=0x{ko:X} found as uint32 at bmd+0x{pos:X}");

            if (pos < 1023)
            {
                ushort v16 = (ushort)(sp[0] | (sp[1] << 8));
                foreach (uint ko in knownOffsets)
                    if (v16 == ko)
                        _modLog!.Info($"[BMD] OffsetScan: known=0x{ko:X} found as uint16 at bmd+0x{pos:X}");
            }
        }
    }

    private async System.Threading.Tasks.Task DispatchMsgLlmAsync(
        SocialLinkSnapshot snap, ushort msgId, nuint session)
    {
        using var cts = new System.Threading.CancellationTokenSource(
            TimeSpan.FromSeconds(_cfg.TimeoutSeconds));
        try
        {
            string ctx = Memory.ContextBuilder.Build(snap) + $" [msg_0x{msgId:X}]";
            var req = new Server.GenerateRequest
            {
                ConfidantId   = snap.ConfidantId,
                Rank          = snap.RankLevel,
                Context       = ctx,
                CharacterName = Memory.ConfidantNames.Resolve(snap.ConfidantId),
            };

            string text = await _llmClient!.GenerateAsync(req, cts.Token);
            if (string.IsNullOrWhiteSpace(text)) return;

            // Cache immediately so OnGameMemcpy can use it on the next text copy.
            _lastLlmText = text;
            _modLog!.Info($"[LLM] msgId=0x{msgId:X}: \"{text[..Math.Min(text.Length, 100)]}\"");
            bool wrote      = TryWriteToBmd(msgId, text);
            int  poolWrites = WritePoolStrings(text);
            _modLog!.Info($"[LLM] msgId=0x{msgId:X} — ptrWrite={(wrote ? "OK" : "skip")} poolWrites={poolWrites}");
        }
        catch (Server.InferenceInFlightException)  { /* server busy, ignore */ }
        catch (OperationCanceledException)
        {
            _modLog!.Warn("[LLM] Timeout — keeping original dialogue.");
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            _modLog!.Warn($"[LLM] Error: {ex.Message}");
        }
    }

    private static unsafe bool TryWriteToSessionDesc(nuint session, string text)
    {
        nuint addr = session + 0x9B0;
        if (!Memory.MemoryGuard.IsReadable(addr, 4)) return false;

        byte* cur = (byte*)addr;
        // Measure existing string length — must not exceed it to avoid overwriting adjacent data.
        int origLen = 0;
        while (origLen < 200 && cur[origLen] != 0) origLen++;
        if (origLen < 4) return false;  // empty/binary slot

        if (!Memory.MemoryGuard.IsWritable(addr, origLen + 1)) return false;

        byte[] encoded = System.Text.Encoding.UTF8.GetBytes(text);
        int writeLen   = Math.Min(encoded.Length, origLen);
        fixed (byte* src = encoded)
            System.Buffer.MemoryCopy(src, cur, origLen, writeLen);
        cur[writeLen] = 0;
        return true;
    }

    // Write LLM text directly to the address captured from session+0xD0 at msgId
    // confirmation time. The game pauses BF execution while a message is displayed,
    // so that address remains valid for the duration of the player's reading window.
    // The heap allocation is PAGE_READWRITE — no VirtualProtect needed.
    private unsafe bool TryWriteToBmd(ushort msgId, string text)
    {
        nuint textAddr = _currentMsgTextAddr;

        // Final-chance lazy recovery: if still 0, re-read the chain now.
        // This catches the race where the LLM responds in the same tick that
        // the poll loop's retry would have fired.
        if (textAddr == 0 && _capturedSession != 0)
        {
            textAddr = TryReadTextAddr(_capturedSession);
            if (textAddr != 0)
            {
                _currentMsgTextAddr = textAddr;
                _modLog!.Info($"[BMD] TextAddr lazy-recovered in write path: 0x{textAddr:X}");
            }
        }

        if (textAddr == 0)
        {
            _modLog!.Warn("[BMD] Write: no textAddr captured yet");
            return false;
        }

        const int MaxWrite = 200;
        if (!Memory.MemoryGuard.IsWritable(textAddr, MaxWrite))
        {
            // Mapped-file pages are PAGE_READONLY — upgrade to PAGE_WRITECOPY so the
            // OS gives us a private writable copy without touching the file on disk.
            if (textAddr < HeapLow)
            {
                const uint PAGE_WRITECOPY = 0x08;
                if (!Memory.MemoryGuard.VirtualProtect(textAddr, (nuint)MaxWrite, PAGE_WRITECOPY, out _))
                {
                    _modLog!.Warn($"[BMD] VirtualProtect WRITECOPY failed for 0x{textAddr:X}");
                    return false;
                }
                _modLog!.Info($"[BMD] VirtualProtect WRITECOPY applied to mapped page 0x{textAddr:X}");
            }
            else
            {
                _modLog!.Warn($"[BMD] Write: 0x{textAddr:X} not writable (sz={MaxWrite})");
                return false;
            }
        }

        // Encode as ASCII — Latin characters are valid single-byte Shift-JIS values,
        // so the game's text renderer will display them without corruption.
        byte[] encoded = System.Text.Encoding.ASCII.GetBytes(text);
        int writeLen   = Math.Min(encoded.Length, MaxWrite - 1);

        byte* dst = (byte*)textAddr;
        fixed (byte* src = encoded)
            System.Buffer.MemoryCopy(src, dst, MaxWrite, writeLen);
        dst[writeLen] = 0;

        _modLog!.Info($"[BMD] Wrote {writeLen}b to textAddr=0x{textAddr:X} (msgId=0x{msgId:X})");
        return true;
    }

    private void StartPollLoop()
    {
        _cts      = new CancellationTokenSource();
        _timer    = new PeriodicTimer(TimeSpan.FromMilliseconds(_cfg.PollIntervalMs));
        _pollTask = Task.Run(() => PollLoopAsync(_cts.Token));
    }

    private async Task PollLoopAsync(CancellationToken ct)
    {
        nuint lastSession = 0;

        while (await _timer!.WaitForNextTickAsync(ct))
        {
            if (!_reader!.TryResolve(out nuint session))
            {
                if (lastSession != 0)
                {
                    _diffScanner.Reset();
                    _bridge!.ResetSession();
                    _lastBfPc        = 0;
                    _bfBufferBase    = 0;
                    _bfBufferOff     = 0;
                    _currentMsgId    = 0;
                    _msgIdCandidate  = 0;
                    _msgIdStreak     = 0;
                    _capturedSession = 0;
                    _confirmedBfBase = 0;
                    _bmdBase             = 0;
                    _bmdScanDone         = false;
                    _currentMsgTextAddr  = 0;
                    _bmdTextPool         = 0;
                    _bmdScanDoneV2       = false;
                    _bmdScanAttempts     = 0;
                    _poolSlots           = null;
                    _bmdPoolLen          = 0x1000;
                    _lastLlmText         = null;
                    lock (_largeCopyLock) _largeCopyDsts.Clear();
                    _modLog!.Info("[P5RGenSocialLinks] Hang-out ended — session cleared.");
                }
                lastSession = 0;
                continue;
            }

            if (session != lastSession)
            {
                lastSession = session;
                _diffScanner.Reset();

                SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
                if (snap is null) continue;

                _modLog!.Info(
                    $"[P5RGenSocialLinks] Hang-out: Confidant={snap.ConfidantId} Rank={snap.RankLevel} Scene={snap.SceneNumber} (0x{session:X})");

                // Fallback dispatch if hook isn't active.
                if (!_hookActive)
                    _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap), lineIndex: 0);

                continue;
            }

            // BF buffer discovery: live scan of session struct every tick.
            TryFindBfBuffer(session);

            // Always maintain the diff snapshot — TryFindBfBufferFromSnapshot reads it
            // to catch the BF script pointer even after it has been cleared from live
            // memory (it exists for only one poll interval at session+0x60).
            string? diff = _diffScanner.Diff(session);
            if (_cfg.StructDiffEnabled && diff is not null)
                _modLog!.Info($"[P5RGenSocialLinks] {diff}");

            // Snapshot-based discovery: catches transient ptrs that TryFindBfBuffer
            // missed because the game set+cleared them between our two live reads.
            TryFindBfBufferFromSnapshot(session);

            // BF line probe: read current dialogue text from bfBase+pc on every tick.
            ProbeBfLine(session);

            // Lazy textAddr re-probe: session+0xD0 may not be populated at BF-PC detection
            // time (cold first-message path). Retry every tick until the C++ dialogue
            // system catches up; the LLM latency (3-30s) gives us plenty of cycles.
            if (_currentMsgId != 0 && _currentMsgTextAddr == 0 && _capturedSession != 0)
            {
                nuint recovered = TryReadTextAddr(_capturedSession);
                if (recovered != 0)
                {
                    _currentMsgTextAddr = recovered;
                    _modLog!.Info($"[MSG] TextAddr recovered=0x{recovered:X} (msgId=0x{_currentMsgId:X})");
                }
            }

            // One-shot bfBase-vicinity scan: find the BMD dialogue string pool.
            if (_confirmedBfBase != 0 && !_bmdScanDoneV2)
                TryScanBmdVicinity();

            // One-shot BMD scan per session — fires once after first msgId confirmation.
            if (_confirmedBfBase != 0 && _bmdBase == 0 && !_bmdScanDone)
            {
                _bmdScanDone = true;
                TryScanForBmd();
            }
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    public void Suspend()
    {
        _cts.Cancel();
        _conversationHook?.Disable();
        _memcpyHook?.Disable();
        _bfDispatchHook?.Disable();
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Suspended.");
    }

    public void Resume()
    {
        _conversationHook?.Enable();
        _memcpyHook?.Enable();
        _bfDispatchHook?.Enable();
        StartPollLoop();
        _logger?.WriteLine("[P5RGenSocialLinks] Resumed.");
    }

    public void Unload()
    {
        _cts.Cancel();
        _pollTask?.Wait(TimeSpan.FromSeconds(2));
        _timer?.Dispose();
        _llmClient?.Dispose();
        _conversationHook?.Disable();
        _memcpyHook?.Disable();
        _bfDispatchHook?.Disable();
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Unloaded.");
    }

    public bool CanUnload()  => true;
    public bool CanSuspend() => true;
    public Action Disposing  => () => { };
}
