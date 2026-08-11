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

        if (dst < HeapLow || src < HeapLow) return;

        // Record all large heap-to-heap copy destinations. The BF scene script is
        // loaded in one bulk copy (several KB) before session detection. We store
        // the dst here and probe it for BF content when the session is first seen.
        if (count >= 500 && count <= 500_000)
        {
            lock (_largeCopyLock)
            {
                if (_largeCopyDsts.Count < 150)
                    _largeCopyDsts.Add(dst);
            }
        }

        // Small-copy vowel filter: IEEE 754 float data (>?@ABCfwD…) has zero
        // lowercase vowels. English dialogue always has ≥3.
        if (count < 10 || count > 150) return;
        if (!Memory.MemoryGuard.IsReadable(dst, (int)count)) return;

        byte* d = (byte*)dst;
        int vowels = 0;
        for (nuint i = 0; i < count; i++)
        {
            byte b = d[i];
            if (b == 'a' || b == 'e' || b == 'i' || b == 'o' || b == 'u') vowels++;
        }
        if (vowels < 3) return;

        var text = new System.Text.StringBuilder(160);
        for (nuint i = 0; i < count && text.Length < 160; i++)
            text.Append(d[i] >= 0x20 && d[i] <= 0x7E ? (char)d[i] : '·');

        _modLog!.Info(
            $"[MemcpyText] src=0x{src:X} dst=0x{dst:X} n={count} vowels={vowels}: \"{text}\"");
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
                _modLog!.Info($"[MSG] pc=0x{pc:X} msgId=0x{best:X} ({best}) bfBase=0x{bfBase:X}");

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
    // Uses an absolute range because bfBase shifts by 200+ MB between runs,
    // making a relative window unreliable.
    // Fires once per session (_bmdScanDone gate in caller).
    private unsafe void TryScanForBmd()
    {
        nuint bfBase = _confirmedBfBase;
        if (bfBase == 0) return;

        const uint  MEM_COMMIT    = 0x1000;
        const uint  PAGE_NOACCESS = 0x01;
        const uint  PAGE_GUARD    = 0x100;
        // All P5R mapped-file assets observed in [0x60000000, 0xFFFF0000].
        // Scan the full lower-4GB so the window never misses due to bfBase drift.
        nuint scanStart = (nuint)0x1000UL;
        nuint scanEnd   = (nuint)0xFFFF0000UL;

        _modLog!.Info($"[BMD] Scan start: bfBase=0x{bfBase:X} (full lower-4GB scan)");

        int regionsChecked = 0;
        nuint probe = scanStart;
        while (probe < scanEnd)
        {
            var (ok, regBase, regSize, state, protect) = MemoryGuard.QueryRegion(probe);
            if (!ok || regSize == 0) break;

            nuint next = regBase + regSize;

            if (state == MEM_COMMIT &&
                (protect & PAGE_NOACCESS) == 0 &&
                (protect & PAGE_GUARD) == 0 &&
                regSize >= 1024 &&
                regSize <= 64u * 1024u * 1024u)
            {
                bool isBfPage = bfBase >= regBase && bfBase < regBase + regSize;
                if (!isBfPage)
                {
                    // Probe 128 KB: BMD header + offset table can consume the first 8–16 KB.
                    int probeLen = (int)Math.Min(regSize, 131072u);
                    if (MemoryGuard.IsReadable(regBase, probeLen))
                    {
                        int sentences = CountDialogueSentences(regBase, probeLen);
                        regionsChecked++;

                        // Log any region with even one qualifying sentence (diagnostic).
                        if (sentences >= 1)
                        {
                            byte* p4 = (byte*)regBase;
                            _modLog!.Info(
                                $"[BMD] Region 0x{regBase:X} sz=0x{regSize:X} prot=0x{protect:X}" +
                                $" s={sentences} [{p4[0]:X2} {p4[1]:X2} {p4[2]:X2} {p4[3]:X2}]");
                        }

                        if (sentences >= 5)
                        {
                            _bmdBase = regBase;
                            _modLog!.Info($"[BMD] Candidate locked: 0x{regBase:X} sentences={sentences}");
                            TryLogBmdStrings();
                            return;
                        }
                    }
                }
            }

            if (next <= regBase) break; // overflow guard
            probe = next;
        }

        _modLog!.Info($"[BMD] Scan done — {regionsChecked} text regions, none hit ≥10 sentences.");
    }

    private static unsafe int CountDialogueSentences(nuint regionBase, int scanBytes)
    {
        byte* p = (byte*)regionBase;
        int sentences = 0;
        int i = 0;

        while (i < scanBytes)
        {
            if (p[i] == 0) { i++; continue; }

            int start = i;
            while (i < scanBytes && p[i] != 0) i++;
            int len = i - start;
            // Short strings (bone names, labels) are always < 25 chars.
            // Bone names also have very short average word length ("b l b …" → avg 1-2).
            if (len < 25) continue;

            int spaces = 0, vowels = 0, ascii = 0;
            for (int j = start; j < i; j++)
            {
                byte c = p[j];
                if (c == ' ') spaces++;
                byte lc = (byte)(c | 0x20);
                if (lc == 'a' || lc == 'e' || lc == 'i' || lc == 'o' || lc == 'u') vowels++;
                if (c >= 0x20 && c < 0x7F) ascii++;
            }

            // avgWordLen = len / (wordCount): filters "b l b seifuk02" (avg ≈ 2)
            int avgWordLen = len / (spaces + 1);
            if (spaces >= 2 && vowels >= 4 && ascii >= len * 9 / 10 && avgWordLen >= 4)
                sentences++;
        }

        return sentences;
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

            _modLog!.Info($"[LLM] msgId=0x{msgId:X}: \"{text[..Math.Min(text.Length, 100)]}\"");
            bool wrote = TryWriteToSessionDesc(session, text);
            _modLog!.Info(wrote
                ? $"[LLM] Wrote {text.Length} chars to session+0x9B0."
                : "[LLM] Write-back SKIPPED (session gone or slot too small).");
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
                    _bmdBase         = 0;
                    _bmdScanDone     = false;
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
