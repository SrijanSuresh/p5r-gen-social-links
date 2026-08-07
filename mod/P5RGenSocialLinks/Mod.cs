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

    // Dialogue heap sits above 256 GB; CLR/runtime copies are all below 4 GB.
    private static readonly nuint HeapLow = unchecked((nuint)0x4000000000UL);

    // Cached BF script buffer address — found by TryFindBfBuffer() on any tick,
    // then used by ProbeBfLine() for the rest of the session.
    private nuint  _bfBufferBase;
    private int    _bfBufferOff;   // session struct offset where we found it (for logging)

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

        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        _logger.WriteLine($"[P5RGenSocialLinks] Base: 0x{moduleBase:X}");

        _reader = new SocialLinkReader(moduleBase, _cfg.VerboseChain,
            msg => _logger!.WriteLine(msg));
        _bridge = new DialogueBridge(_llmClient!, new LoggerAdapter(_logger!), _cfg);

        loader.GetController<IReloadedHooks>()?.TryGetTarget(out _hooks);

        TryActivateHook();
        SetupMemcpyHook();
        StartPollLoop();

        _logger.WriteLine($"[P5RGenSocialLinks] Started — hook:{(_hookActive ? "ON" : "OFF")} poll:ON");
    }

    private bool _hookActive;

    private void SetupMemcpyHook()
    {
        if (_hooks is null)
        {
            _logger!.WriteLine("[P5RGenSocialLinks] Memcpy inner hook skipped — IReloadedHooks null.");
            return;
        }
        nuint moduleBase = (nuint)Process.GetCurrentProcess().MainModule!.BaseAddress;
        // Hook the REP MOVSB inner function directly (not the outer dispatcher).
        // The dialogue system reaches FUN_1405a8570 via a function pointer, bypassing
        // FUN_1405a8590, so hooking the inner function catches all paths.
        nuint addr = moduleBase + 0x5A8570;
        _memcpyHook = _hooks.CreateHook<MemcpyInnerDelegate>(OnGameMemcpy, (long)addr).Activate();
        _logger!.WriteLine($"[P5RGenSocialLinks] Memcpy inner hook ACTIVE at 0x{addr:X}");
    }

    private unsafe void OnGameMemcpy(nuint dst, nuint src, nuint count)
    {
        // Diagnostic logging disabled — memcpy hook served its purpose (confirmed the
        // BF script is loaded once before session detection, not per-line).
        // [BFLine] probe via ProbeBfLine() is the active diagnostic path now.
        _memcpyHook!.OriginalFunction(dst, src, count);
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
    private ushort _lastBfPc;       // change-detection: fires only when PC moves

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
    private unsafe void TryFindBfBuffer(nuint session)
    {
        if (_bfBufferBase != 0) return;   // already cached
        const int sessionScan = 1024;   // extended: BF ptr may be past +0x200
        const int probeScan   = 512;
        const int minRun      = 20;

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
    /// Logs one line per dialogue advance using the cached BF buffer + BF PC.
    /// </summary>
    private unsafe void ProbeBfLine(nuint session)
    {
        if (_bfBufferBase == 0) return;   // buffer not found yet

        if (!Memory.MemoryGuard.IsReadable(session + 0x20, 2)) return;
        ushort pc = *(ushort*)((byte*)session + 0x20);
        if (pc == _lastBfPc) return;
        _lastBfPc = pc;

        nuint lineAddr = _bfBufferBase + pc;
        if (!Memory.MemoryGuard.IsReadable(lineAddr, 64)) return;
        byte* b = (byte*)lineAddr;

        var hex  = new System.Text.StringBuilder(24);
        for (int i = 0; i < 8; i++) hex.Append($"{b[i]:X2} ");

        var text = new System.Text.StringBuilder(128);
        for (int i = 0; i < 64; i++)
            if (b[i] >= 0x20 && b[i] <= 0x7E) text.Append((char)b[i]);

        _modLog!.Info(
            $"[BFLine] pc=0x{pc:X4} @0x{lineAddr:X} [0x{_bfBufferOff:X3}+pc] [{hex}]: \"{text}\"");
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
                    _lastBfPc     = 0;
                    _bfBufferBase    = 0;
                    _bfBufferOff     = 0;
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
        }
    }

    // ── Lifecycle ─────────────────────────────────────────────────────────

    public void Suspend()
    {
        _cts.Cancel();
        _conversationHook?.Disable();
        _memcpyHook?.Disable();
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Suspended.");
    }

    public void Resume()
    {
        _conversationHook?.Enable();
        _memcpyHook?.Enable();
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
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Unloaded.");
    }

    public bool CanUnload()  => true;
    public bool CanSuspend() => true;
    public Action Disposing  => () => { };
}
