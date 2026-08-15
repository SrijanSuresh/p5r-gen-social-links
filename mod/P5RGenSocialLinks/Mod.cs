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

    // Address of the interpreter's MOVZX, resolved by signature at load. Zero means the
    // scan did not find it and the pool heuristic remains the only path to the text.
    private nuint                        _msgByteFetch;
    private MsgInterpreterWatch?         _msgWatch;

    // Last record pointer reported to the log. The hook fires once per character, so the
    // interesting event is the value changing, not the value existing.
    private nuint                        _lastWatchedRecord;

    // Scratch for previewing a watched record. Separate from _scanBuf so a preview can
    // never disturb a pool scan mid-walk, even though both run on the poll thread today.
    private readonly byte[]              _recordBuf = new byte[128];
    private readonly byte[]              _mirrorBuf = new byte[128];

    // Guards the pool write path. Flushing now happens from two places — the poll tick
    // and the thread-pool continuation that finishes a generation — and both walk the
    // plan and write into game memory. Arming takes it too, because replacing _heapPools
    // under a flush would have it writing through a base it is halfway done with.
    private readonly object              _writeLock = new();

    // Signed distance from a record in an armed pool to the same record in the copy the
    // speech bubble reads, discovered by watching both be read within a millisecond of
    // each other with identical text.
    //
    // The game keeps every scene's dialogue in two places at a constant offset, and which
    // one the bubble reads is not something ranking can determine — measured at
    // 0x3C60B8F0 one run and 0x2E67F270 the next. Twin arming by content match found it
    // sometimes and missed it in the run that produced this field: one region armed, and
    // every write landed in the text log while the bubble kept the script.
    private long                         _twinDelta;

    // True between a hang-out starting and ending. Arming reads the interpreter's own
    // pointer, and the interpreter serves every string in the game, so this is what
    // keeps a menu label from being mistaken for the scene's dialogue pool.
    private volatile bool                _sessionActive;

    // Consecutive dialogue reads landing outside the armed pool. Two means the scene
    // moved to a different pool; one is a menu or a name plate passing through.
    private int                          _outsideReads;
    private (nuint Addr, string Text)    _lastWatched;

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

    // The MSG1 files reachable from bfBase are all global item/skill tables, so the pool
    // write is now a one-shot probe of the write path rather than a per-message action:
    // if skill descriptions in-game show the mock text, write→render is proven.
    private bool _poolWriteDone;

    // True when the pool is the heap dialogue allocation rather than a mapped MSG1 file.
    // The heap pool is the correct target, so it is rewritten on every message; the
    // one-shot guard applies only to the MSG1 item tables.
    private bool _poolIsHeap;

    // Every high-scoring heap region, not just the winner. Ranking by content picked a
    // shader block first and, once fixed, still would not have found the live scene: the
    // confirmed address 0x41DD7F6389 fell outside every region the scan surfaced. Writing
    // to all strong candidates removes the need to guess correctly on the first try.
    private readonly System.Collections.Generic.List<(nuint Base, int Len, (int Off, int Len)[] Slots)>
        _heapPools = new();

    // Slots grouped into records, one list per armed pool, and the index of the record
    // each pool is expected to render next.
    //
    // Writing every record with one generated line is what makes the text log repeat the
    // same sentence at three different widths: those are three records, all overwritten.
    // Measured timing says we do not have to. The MSG dispatch fires ~3s before the
    // renderer reads the record and the write completes ~1.4s before it, so there is room
    // to write exactly the record that is about to be drawn.
    private readonly System.Collections.Generic.List<System.Collections.Generic.List<(int Start, int Count)>>
        _poolRecords = new();
    private readonly System.Collections.Generic.List<int> _poolNextRecord = new();

    // The scene's records as a plan: capacity and original line per record, plus where
    // each one is in its life cycle. Built once from region 0 at arm time, because every
    // armed region is a copy of the same script with identical record widths.
    //
    // This is what makes generation independent of the player. All of it is knowable
    // before a single line is displayed; waiting for a dispatch event to learn a record's
    // size or its scripted text was only ever necessary while the buffer was a guess.
    private readonly System.Collections.Generic.List<RecordPlan> _plan = new();

    // UTF-16 hunt state. _utf16CpyLogged caps the memcpy log; the sweep cursor lets a
    // multi-GB heap scan resume across ticks instead of stalling one.
    private int   _utf16CpyLogged;
    private nuint _heapSweepCursor;
    private int   _utf16SweepHits;
    private bool  _heapSweepDone;

    // Every game object pointer observed so far lands in 0x42xxxxxxxx, while the linear
    // sweep starts at HeapLow (0x4000000000) and spends whole sessions crawling through
    // DirectInput and font data around 0x4188xxxxxx without ever reaching game memory.
    // Confirmed dialogue addresses were 0x41DD7F6389 and 0x42102CAAA9, so the sweep must
    // start below 0x42 — the system-string filter handles the DirectInput and font data
    // that sits around 0x4188xxxxxx.
    private static readonly nuint GameHeapStart = unchecked((nuint)0x4100000000UL);

    // Upper bound for the scan. Without one the walk continued into 0x7FF... DLL space and
    // scored the CLR's own resource strings ("Cannot create an abstract class."), which
    // wasted the region budget and armed writes against runtime memory that cannot hold
    // dialogue and must not be modified.
    private static readonly nuint GameHeapEnd = unchecked((nuint)0x4400000000UL);

    // Pointers harvested from the per-message D0 array — swept before the linear scan,
    // since they point at objects the game is using for the message on screen right now.
    private readonly System.Collections.Generic.List<nuint> _sweepSeeds = new();
    private readonly object _seedLock = new();

    // Most recent LLM response. Written on the async LLM thread; read volatilely on the
    // game thread inside OnGameMemcpy so no lock is needed (reference read/write is
    // atomic on x64, and we only need eventual visibility, not strict ordering).
    private volatile string? _lastLlmText;

    /// <summary>
    /// The scene's original scripted lines, read out of the pool before we overwrite it.
    ///
    /// This is the context the model was missing: it knew who and where, but never what
    /// the conversation was actually about, so Ryuji answered the room rather than the
    /// scene. The lines we destroy are exactly the lines needed to respond to.
    ///
    /// Capture is once-only, at arm time. After the first write every slot holds our own
    /// generated text, so a later read would feed the model its own words back as if they
    /// were the script.
    /// </summary>
    private readonly System.Collections.Generic.List<string> _sceneScript = new();

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
            TryResolveMsgInterpreter();
            SetupMemcpyHook();
            // BfDispatch hook crashes regardless of handler — abandoned, hunting text ptr via CE instead
            StartPollLoop();

            _modLog.Always($"[P5RGenSocialLinks] Started — hook:{(_hookActive ? "ON" : "OFF")} poll:ON");
            _modLog.Always($"[P5RGenSocialLinks] Log mirrored to {ModLogger.LogPath}");
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

        // UTF-16 dialogue detection. The earlier ASCII probe concluded dialogue never
        // flows through this function, but it could not have seen wide-char text at all —
        // interleaved nulls end every ASCII run after one character. This retries that
        // question with the right encoding.
        //
        // The prefilter reads 8 bytes and rejects nearly every copy the game makes, so
        // the expensive scan runs only on plausible wide strings. That matters: this is
        // one of the hottest functions in the process.
        if (_utf16CpyLogged >= 40 || count < 16 || count > 800) return;
        if (!Memory.MemoryGuard.IsReadable(dst, (int)count)) return;

        byte* d = (byte*)dst;
        if (d[1] != 0 || d[3] != 0 || d[5] != 0 || d[7] != 0)      return;
        if (!IsPrintable(d[0]) || !IsPrintable(d[2]) ||
            !IsPrintable(d[4]) || !IsPrintable(d[6]))              return;

        var wide = FindUtf16English(dst, (int)count, 1);
        if (wide.Count == 0) return;

        _utf16CpyLogged++;
        _modLog!.Info($"[UTF16cpy] src=0x{src:X} dst=0x{dst:X} n={count}: \"{wide[0].Text}\"");
    }

    /// <summary>
    /// Locate the message interpreter's byte-fetch instruction and record its address.
    ///
    /// This is the first half of replacing the heap heuristic. Everything the pool code
    /// does today - scan tens of megabytes, score regions for English, arm the top one
    /// plus its content-identical twin - exists only because we could not ask the game
    /// which bytes it was about to render. This instruction can be asked: the struct in
    /// RBX holds the record pointer and the cursor, so hooking it turns a guess into a
    /// read (learning.md Ch. 65-66).
    ///
    /// Resolution is separated from hooking on purpose. A signature that resolves proves
    /// the pattern survived this build; a hook that misbehaves is a different failure,
    /// and diagnosing them together in a live game means restarting P5R for each guess.
    /// </summary>
    private void TryResolveMsgInterpreter()
    {
        try
        {
            using var scanner = new FunctionScanner();
            nuint? addr = scanner.TryFindFirst(Signatures.MsgByteFetch);
            if (addr is null)
            {
                _modLog!.Always(
                    "[P5RGenSocialLinks] MsgByteFetch sig NOT FOUND — p5r.exe build differs " +
                    "from the one the pattern was taken from. Pool heuristic stays in charge.");
                return;
            }

            _msgByteFetch = addr.Value + Signatures.MsgByteFetchToMovzx;
            nuint rva     = addr.Value - scanner.ModuleBase;

            // Expected P5R.exe+17A3D1F. Logging the offset rather than the absolute
            // address is what makes this comparable to a disassembler across runs.
            _modLog!.Always(
                $"[P5RGenSocialLinks] MsgByteFetch sig OK: P5R.exe+{rva:X} " +
                $"(abs 0x{addr.Value:X}, movzx at 0x{_msgByteFetch:X})");

            if (!_cfg.MsgHookEnabled)
            {
                _modLog!.Always("[P5RGenSocialLinks] Msg watch disabled by config.");
                return;
            }
            if (_hooks is null)
            {
                _modLog!.Always("[P5RGenSocialLinks] Msg watch skipped — IReloadedHooks null.");
                return;
            }

            try
            {
                _msgWatch = new MsgInterpreterWatch(_hooks, _msgByteFetch);
                _msgWatch.StartSampling();
                _modLog!.Always("[P5RGenSocialLinks] Msg watch ACTIVE (sampling at 5ms).");
            }
            catch (Exception ex)
            {
                // Deliberately broad, and only around this one call. Assembling the stub
                // runs through a third-party assembler whose failure type is not part of
                // the interface we reference, and every path after this point still works
                // without the watch — so an unknown assembler error must degrade to the
                // pool heuristic rather than abort mod startup.
                _modLog!.Always($"[P5RGenSocialLinks] Msg watch FAILED to install: {ex.Message}");
                _msgWatch = null;
            }
        }
        catch (InvalidOperationException ex)
        {
            _modLog!.Always($"[P5RGenSocialLinks] MsgByteFetch scan FAILED: {ex.Message}");
        }
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

            bool dispatched = _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap),
                                                       lineIndex: fireCount,
                                                       maxChars: NextRecordCapacity());
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

    /// <summary>
    /// ASCII counterpart to <see cref="FindUtf16English"/>. Cheat Engine located the live
    /// dialogue at 0x41DD7F6389 and 0x42102CAAA9 as single-byte text — in the heap, not
    /// the mapped BMD region. Both halves of that were already implemented and never
    /// combined: the ASCII scanners only ever ran over the bfBase vicinity, and the heap
    /// sweep only ever looked for wide chars.
    /// </summary>
    private static unsafe System.Collections.Generic.List<(nuint Addr, string Text)>
        FindAsciiEnglish(nuint region, int byteLen, int maxHits)
    {
        var hits = new System.Collections.Generic.List<(nuint, string)>();
        byte* p = (byte*)region;
        int i = 0;

        while (i < byteLen && hits.Count < maxHits)
        {
            while (i < byteLen && !IsPrintable(p[i])) i++;
            int begin = i;
            while (i < byteLen && IsPrintable(p[i])) i++;

            int len = i - begin;
            if (len < 10 || len > 400) continue;

            var sb = new System.Text.StringBuilder(len);
            for (int k = begin; k < i; k++) sb.Append((char)p[k]);
            string s = sb.ToString();
            if (IsEnglishString(s)) hits.Add((region + (nuint)begin, s));
        }
        return hits;
    }

    // Reusable scan buffer — the heap sweep reads hundreds of megabytes per pass and
    // must not allocate a fresh array per chunk.
    private const int ScanChunk = 0x40000; // 256 KB
    private readonly byte[] _scanBuf = new byte[ScanChunk];

    /// <summary>
    /// Allocation-free approximation of the letter/lowercase/space ratios in
    /// <see cref="IsEnglishString"/>. Deliberately looser — it only has to reject the
    /// obvious non-prose cheaply so the real test runs on a small remainder.
    /// </summary>
    private static bool QuickEnglishBytes(byte[] buf, int begin, int len)
    {
        int letters = 0, lower = 0, spaces = 0;
        int end = begin + len;
        for (int i = begin; i < end; i++)
        {
            byte c = buf[i];
            if      (c >= 'a' && c <= 'z') { letters++; lower++; }
            else if (c >= 'A' && c <= 'Z') letters++;
            else if (c == ' ')             spaces++;
        }
        return spaces >= 1
               && letters * 100 >= len * 55
               && lower   * 100 >= letters * 40;
    }

    /// <summary>
    /// Buffer-based counterpart to <see cref="FindAsciiEnglish"/>. Operates on a copy made
    /// by <see cref="MemoryGuard.TryRead"/> so a region freed mid-scan cannot fault.
    /// </summary>
    private static System.Collections.Generic.List<(nuint Addr, string Text)>
        FindAsciiEnglishBuf(byte[] buf, int len, nuint baseAddr, int maxHits)
    {
        var hits = new System.Collections.Generic.List<(nuint, string)>();
        int i = 0;

        while (i < len && hits.Count < maxHits)
        {
            while (i < len && !IsPrintable(buf[i])) i++;
            int begin = i;
            while (i < len && IsPrintable(buf[i])) i++;

            int slen = i - begin;
            if (slen < 10 || slen > 400) continue;

            // Byte-level pre-check before allocating. A full scan covers ~1.5 GB, and
            // materialising a string for every printable run produced millions of
            // allocations per pass — enough GC pressure to visibly stutter the game.
            if (!QuickEnglishBytes(buf, begin, slen)) continue;

            string s = System.Text.Encoding.ASCII.GetString(buf, begin, slen);
            if (IsEnglishString(s)) hits.Add((baseAddr + (nuint)begin, s));
        }
        return hits;
    }

    /// <summary>
    /// Captures (offset, length) slots across a region using buffered reads. Strings that
    /// straddle a chunk boundary are dropped rather than stitched — at 256 KB chunks that
    /// is a negligible fraction, and it keeps every read bounds-checked.
    /// </summary>
    /// <summary>
    /// Renders the captured script as a context fragment, or empty if nothing was caught.
    /// </summary>
    /// <remarks>
    /// Framed as what the scene is about rather than as lines to imitate. Handing the
    /// model verbatim dialogue invites it to parrot a line back, which reads worse than
    /// generic filler — the goal is a reply that belongs in this conversation, not a
    /// paraphrase of one already in it.
    /// </remarks>
    private string ScriptContext()
    {
        if (_sceneScript.Count == 0) return "";
        return " The scene's original dialogue, for subject matter only — do not repeat " +
               "or paraphrase these lines: \"" +
               string.Join("\" \"", _sceneScript) + "\"";
    }

    /// <summary>
    /// Reads the scripted lines out of a pool region, in address order.
    /// </summary>
    /// <remarks>
    /// Must run before the first write to this region. The pool is the only place the
    /// scene's script exists in memory, and overwriting it is destructive — there is no
    /// second chance to read it, which is why this is called at arm time rather than
    /// lazily when the context is first needed.
    ///
    /// Lines are capped in count and total length: the context field is limited to 1024
    /// characters server-side, and a 36-slot scene at ~40 characters each would exceed it
    /// on its own and crowd out the rolling session history.
    /// </remarks>
    private string[] CaptureSceneScript(nuint poolBase, int scanLen, int maxLines, int maxChars)
    {
        var lines = new System.Collections.Generic.List<string>();
        var seen  = new System.Collections.Generic.HashSet<string>();
        int used  = 0;

        for (int chunkOff = 0; chunkOff < scanLen && lines.Count < maxLines; chunkOff += ScanChunk)
        {
            int want = Math.Min(ScanChunk, scanLen - chunkOff);
            if (!MemoryGuard.TryRead(poolBase + (nuint)chunkOff, _scanBuf, want)) continue;

            foreach (var (_, text) in FindAsciiEnglishBuf(_scanBuf, want, poolBase + (nuint)chunkOff, 20000))
            {
                string line = text.Trim();
                if (line.Length < 8) continue;

                // Consecutive slots repeat the same line often enough that the raw list
                // wastes most of the budget on duplicates.
                if (!seen.Add(line)) continue;

                // Refuse anything we generated. Arming can recur mid-session, and feeding
                // the model its own output back as "the script" would compound drift.
                if (_lastLlmText != null && line.Contains(_lastLlmText)) continue;

                lines.Add(line);
                used += line.Length + 1;
                if (lines.Count >= maxLines || used >= maxChars) return lines.ToArray();
            }
        }
        return lines.ToArray();
    }

    private (int Off, int Len)[] CapturePoolSlotsSafe(nuint poolBase, int scanLen)
    {
        var slots = new System.Collections.Generic.List<(int, int)>();

        for (int chunkOff = 0; chunkOff < scanLen && slots.Count < 20000; chunkOff += ScanChunk)
        {
            int want = Math.Min(ScanChunk, scanLen - chunkOff);
            if (!MemoryGuard.TryRead(poolBase + (nuint)chunkOff, _scanBuf, want)) continue;

            foreach (var (addr, text) in FindAsciiEnglishBuf(_scanBuf, want, poolBase + (nuint)chunkOff, 20000))
            {
                slots.Add(((int)(addr - poolBase), text.Length));
                if (slots.Count >= 20000) break;
            }
        }
        return slots.ToArray();
    }

    /// <summary>
    /// Print absolute addresses of the anchor's first slots, for setting a hardware
    /// read watchpoint on them in a debugger.
    ///
    /// The ranking tells us which region looks most like dialogue; it cannot tell us
    /// which slot the renderer actually reads, which is why every slot gets written and
    /// both rows of the bubble show the same line. That question is answerable only by
    /// observing the consumer (learning.md Ch. 65), and observing it starts with an
    /// address to watch.
    ///
    /// Hunting that address by string-scanning the process is possible but slow and
    /// ambiguous — the same line exists in several copies. The mod already holds the
    /// exact offsets, so printing them turns a search into a paste.
    ///
    /// The list has to cover the whole region, not just the head. A watchpoint catches
    /// only reads that happen after it is armed, and the renderer reads a slot once, on
    /// the transition into that line — so the useful target is always a line that has
    /// not been displayed yet. Logging the first 8 slots meant the player had to hand-read
    /// memory as soon as the scene got past them.
    /// </summary>
    private void LogWatchpointTargets(nuint poolBase, (int Off, int Len)[] slots, int maxSlots)
    {
        int n = Math.Min(maxSlots, slots.Length);
        _modLog!.Info($"[SLOTS] anchor 0x{poolBase:X} — first {n} of {slots.Length} slots:");
        for (int i = 0; i < n; i++)
        {
            (int off, int len) = slots[i];
            nuint addr = poolBase + (nuint)off;

            // Read the text back rather than reusing the capture-time string: this runs
            // before any write, so it is the scripted line, and it must match what is on
            // screen for the address to be worth watching.
            string text = AsciiPreview(addr, Math.Min(len, 64));
            _modLog!.Info($"[SLOTS]   0x{addr:X} len={len} \"{text}\"");
        }
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

    /// <summary>
    /// Resumable UTF-16 sweep of the game heap, independent of the session struct.
    /// session+0xC8 held a live message object one run and the flag value 0x80004001 the
    /// next, so anything gated on it runs only intermittently; this is anchored to the
    /// heap itself instead.
    ///
    /// A budget of bytes per tick keeps a multi-GB heap from stalling the poll loop —
    /// the cursor persists so each tick continues where the last stopped.
    /// </summary>
    private unsafe void SweepHeapForUtf16()
    {
        if (_utf16SweepHits >= 40 || _heapSweepDone) return;

        const long budget    = 24L * 1024 * 1024; // per tick
        const int  chunk     = 0x100000;          // 1 MB per read
        const uint MEM_COMMIT = 0x1000, PAGE_NOACCESS = 0x01, PAGE_GUARD = 0x100;

        // Seeded sweeps first — ±2 MB around pointers the game is using for the message
        // currently on screen is far better odds than anywhere the linear scan has reached.
        nuint[] seeds;
        lock (_seedLock) { seeds = _sweepSeeds.ToArray(); _sweepSeeds.Clear(); }

        foreach (nuint seed in seeds)
        {
            if (_utf16SweepHits >= 40) return;
            const int span = 0x200000;
            nuint sFrom = seed > (nuint)span ? seed - span : seed;
            for (nuint a = sFrom; a < seed + span; a += chunk)
            {
                if (!MemoryGuard.IsReadable(a, chunk)) continue;
                foreach (var (addr2, t) in FindAsciiEnglish(a, chunk, 64))
                {
                    if (IsSystemString(t)) continue;
                    _modLog!.Info($"[SEEDASCII] 0x{addr2:X}: \"{t}\"");
                    if (++_utf16SweepHits >= 40) return;
                }
                foreach (var (addr2, t) in FindUtf16English(a, chunk, 16))
                {
                    if (IsSystemString(t)) continue;
                    _modLog!.Info($"[SEEDUTF16] 0x{addr2:X}: \"{t}\"");
                    if (++_utf16SweepHits >= 40) return;
                }
            }
        }

        long  scanned = 0;
        nuint addr    = _heapSweepCursor != 0 ? _heapSweepCursor : GameHeapStart;

        while (scanned < budget)
        {
            var (ok, regionBase, regionSize, state, protect) = MemoryGuard.QueryRegion(addr);
            if (!ok || regionSize == 0) { _heapSweepDone = true; break; }

            nuint regionEnd = regionBase + regionSize;
            if (regionEnd <= addr || regionEnd > UserAddrMax) { _heapSweepDone = true; break; }

            bool usable = state == MEM_COMMIT
                          && (protect & PAGE_NOACCESS) == 0
                          && (protect & PAGE_GUARD) == 0;

            if (usable)
            {
                nuint from = addr > regionBase ? addr : regionBase;
                while (from < regionEnd && scanned < budget)
                {
                    int len = (int)Math.Min((ulong)(regionEnd - from), (ulong)chunk);
                    if (len <= 0 || !MemoryGuard.IsReadable(from, len)) break;

                    // 64 candidates per chunk, then filter: the sweep previously spent its
                    // entire hit budget on Windows runtime strings in the low heap and
                    // stopped before reaching any game data.
                    foreach (var (a, t) in FindAsciiEnglish(from, len, 64))
                    {
                        if (IsSystemString(t)) continue;
                        _modLog!.Info($"[HEAPASCII] 0x{a:X}: \"{t}\"");
                        if (++_utf16SweepHits >= 40) { _heapSweepCursor = regionEnd; return; }
                    }
                    foreach (var (a, t) in FindUtf16English(from, len, 16))
                    {
                        if (IsSystemString(t)) continue;
                        _modLog!.Info($"[HEAPUTF16] 0x{a:X}: \"{t}\"");
                        if (++_utf16SweepHits >= 40) { _heapSweepCursor = regionEnd; return; }
                    }
                    from    += (nuint)len;
                    scanned += len;
                }
            }
            addr = regionEnd;
        }

        _heapSweepCursor = addr;
    }

    /// <summary>
    /// Rejects Windows runtime strings. The heap sweep spent its whole budget reporting
    /// DirectInput names, font copyright blocks and impersonation flags — all genuine
    /// UTF-16, none of it game text.
    /// </summary>
    private static bool IsSystemString(string s) =>
        s.Length == 0 ||
        s.Contains("Microsoft") || s.Contains("Windows") || s.Contains("Corporation") ||
        s.Contains("Copyright")  || s.Contains("http")    || s.Contains(".dll") ||
        s.Contains("OpenType")   || s.Contains("License") || s.Contains("reserved") ||
        s.Contains("Impersonation") || s.Contains("DirectInput") || s.Contains("Version") ||
        // Shader and engine identifiers. The top-ranked region was full of "float3
        // position" — valid English by every ratio test, and not remotely dialogue.
        s.Contains("float")  || s.Contains("_")      || s.Contains("()") ||
        s.Contains("vec")    || s.Contains("Matrix") || s.Contains("Buffer") ||
        s.Contains("Shader") || s.Contains("Texture");

    /// <summary>
    /// Scores a string as conversational rather than merely English. Item names, skill
    /// labels and shader identifiers all pass the ratio test; what separates dialogue is
    /// sentence punctuation and second-person address.
    ///
    /// Weighted rather than boolean because P5R splits lines on embedded function codes,
    /// so many genuine fragments ("? You got any big") end mid-sentence and would fail
    /// any single hard requirement.
    /// </summary>
    private static int DialogueScore(string s)
    {
        if (!IsEnglishString(s) || IsSystemString(s)) return 0;

        int score = 0;
        char last = s[^1];
        if (last == '.' || last == '!' || last == '?') score += 3;
        else if (last == ',' || last == '"')           score += 1;

        if (s.Contains(" you") || s.Contains("You ") || s.Contains(" I ") ||
            s.Contains("I'm")  || s.Contains(" me")   || s.Contains(" we ") ||
            s.Contains("Ryuji"))                       score += 3;

        if (s.Contains('\'')) score += 1;   // contractions
        if (s.Contains(' '))  score += 1;

        return score;
    }

    /// <summary>
    /// Walks session+0xD0 as a pointer array. The per-message dumps show it holding
    /// 0x42... heap pointers whose values change on every single message, which makes it
    /// the per-message context block — and unlike session+0xC8 it has been populated on
    /// every run observed. It sits below HeapLow, which is precisely why the pointer
    /// filters everywhere else discarded it.
    ///
    /// Each slot is tested for text, then one level deeper, since a character buffer is
    /// usually reached through a wrapper object rather than referenced directly.
    /// </summary>
    private unsafe void ProbeD0Array(nuint session)
    {
        if (!MemoryGuard.IsReadable(session + 0xD0, 8)) return;
        nuint arr = *(nuint*)(session + 0xD0);
        if (arr < 0x10000 || arr > UserAddrMax) return;
        if (!MemoryGuard.IsReadable(arr, 0x200)) return;

        int logged = 0;

        for (int i = 0; i < 0x200 && logged < 12; i += 8)
        {
            nuint p1 = *(nuint*)(arr + (nuint)i);
            if (p1 < 0x10000 || p1 > UserAddrMax)      continue;
            if (!MemoryGuard.IsReadable(p1, 256))      continue;

            // Seed the sweep even when this slot holds no text itself: it points into the
            // region the game is actively using for this message.
            if (p1 >= HeapLow)
                lock (_seedLock) { if (_sweepSeeds.Count < 32) _sweepSeeds.Add(p1); }

            if (ReportText($"+0x{i:X2}", p1)) { logged++; continue; }

            for (int j = 0; j < 0x40 && logged < 12; j += 8)
            {
                nuint p2 = *(nuint*)(p1 + (nuint)j);
                if (p2 < 0x10000 || p2 > UserAddrMax)  continue;
                if (!MemoryGuard.IsReadable(p2, 256))  continue;
                if (ReportText($"+0x{i:X2}+0x{j:X2}", p2)) logged++;
            }
        }

        if (logged == 0) _modLog!.Info($"[D0] array 0x{arr:X}: no text in 2 levels");

        bool ReportText(string label, nuint at)
        {
            string ascii = AsciiPreview(at, 96);
            var    wide  = FindUtf16English(at, 256, 1);
            string utf16 = wide.Count > 0 ? wide[0].Text : "";
            if (IsSystemString(ascii) && IsSystemString(utf16)) return false;
            _modLog!.Info($"[D0] {label} 0x{at:X} ascii=\"{ascii}\" utf16=\"{utf16}\"");
            return true;
        }
    }

    /// <summary>
    /// Locates the scene's dialogue pool on the heap and captures its slots.
    ///
    /// Cheat Engine established the layout: consecutive lines land in the same regions at
    /// increasing offsets — "Protein Lovers gym!" at 0x41DD7F6389, the next line 0x56A
    /// later at 0x41DD7F68F3, with a second region advancing by exactly the same delta.
    /// The scene's lines therefore sit sequentially in one allocation rather than being
    /// reallocated per line, which makes the containing region a stable write target for
    /// the whole conversation.
    ///
    /// Regions are ranked by how many non-system English sentences they hold and the top
    /// candidates are logged, so a wrong pick is visible rather than silent — the same
    /// mistake the mapped-file scan made when it locked onto the shoe shop.
    /// </summary>
    private unsafe void TryFindHeapDialoguePool()
    {
        const uint MEM_COMMIT = 0x1000, PAGE_NOACCESS = 0x01, PAGE_GUARD = 0x100;
        // 8 MB per region, not 256 KB. Sampling only a region's start is what hid the live
        // dialogue: the confirmed address 0x41DD7F6389 sits ~12 MB past the base of the
        // nearest region the scan reported, so it was never read.
        const int  maxScan     = 0x800000;
        const int  maxRegions  = 4096;
        // 2 GB, not 512 MB. At 8 MB per region the old budget stopped after 64 regions —
        // far short of the game's heap — so the scan could exhaust itself before ever
        // reaching the allocation holding the live scene.
        const long totalBudget = 2048L * 1024 * 1024;

        var ranked = new System.Collections.Generic.List<(int Score, nuint Base, int Len, string Sample)>();
        nuint addr = GameHeapStart;
        int   seen = 0;
        long  totalScanned = 0;

        while (addr < GameHeapEnd && seen < maxRegions && totalScanned < totalBudget)
        {
            var (ok, regionBase, regionSize, state, protect) = MemoryGuard.QueryRegion(addr);
            if (!ok || regionSize == 0) break;
            nuint regionEnd = regionBase + regionSize;
            if (regionEnd <= addr) break;
            seen++;

            bool usable = state == MEM_COMMIT
                          && (protect & PAGE_NOACCESS) == 0
                          && (protect & PAGE_GUARD) == 0;

            if (usable && regionSize >= 0x1000 && regionSize <= 0x4000000)
            {
                int len = (int)Math.Min((ulong)regionSize, (ulong)maxScan);

                // Read through TryRead in chunks rather than dereferencing the region
                // directly: the game frees heap on its own threads and a raw walk over
                // megabytes will eventually fault on memory that vanished mid-scan.
                int    score  = 0;
                int    matches = 0;
                string sample = "";
                int    read   = 0;

                for (int chunkOff = 0; chunkOff < len; chunkOff += ScanChunk)
                {
                    int want = Math.Min(ScanChunk, len - chunkOff);
                    if (!MemoryGuard.TryRead(regionBase + (nuint)chunkOff, _scanBuf, want)) break;
                    read += want;

                    foreach (var (a, t) in FindAsciiEnglishBuf(_scanBuf, want,
                                                              regionBase + (nuint)chunkOff, 8192))
                    {
                        int s = DialogueScore(t);
                        if (s == 0) continue;
                        if (sample.Length == 0) sample = t;
                        score += s;
                        matches++;
                    }
                }

                // Rank by average score per matched line, not the sum. A summed score is
                // really a measure of size: a 2MB item-description table outscores a
                // 400KB conversation pool on volume alone, which is how the region that
                // actually renders ended up ranked #14. Per-line, casual speech (ends in
                // punctuation, second-person, contractions) separates cleanly from
                // "Restores 20 HP to one ally."
                if (matches >= 30)
                {
                    int avg = score * 100 / matches;
                    ranked.Add((avg, regionBase, len, sample));
                }
                totalScanned += read;
            }
            addr = regionEnd;
        }

        // Coverage matters as much as the ranking: if the budget ran out, the right region
        // may simply never have been read, which looks identical to a bad ranking.
        _modLog!.Info($"[POOL] scanned {seen} regions, {totalScanned / (1024 * 1024)} MB, " +
                      $"{ranked.Count} scored" +
                      $"{(totalScanned >= totalBudget ? " (BUDGET EXHAUSTED — coverage incomplete)" : "")}");

        if (ranked.Count == 0) return;
        ranked.Sort((a, b) => b.Score.CompareTo(a.Score));

        // Average-per-line ranking reliably puts the live scene at #0 ("Here we are...
        // Protein Lovers gym!", avg 4.72) above item tables at 3.18 and shader source at
        // 1.00. Rank alone is not enough to pick the write set, though, for two reasons
        // learned the hard way:
        //
        //  - Taking the top 3 by rank once armed "ternTableOffset: -1", a data table that
        //    averaged well without being dialogue, and writing it crashed the game. Rank
        //    is a similarity score, not a type check.
        //  - Taking only #0 rendered nothing. P5R holds the scene's dialogue in TWO
        //    buffers: one the text log reads, one the renderer reads. Writing #0 filled
        //    the backlog with generated lines while the speech bubble kept the original.
        //    The twin was sitting at alt #9 (avg 2.35) with an identical sample — far
        //    below any sane rank cutoff, and identical in content.
        //
        // So: anchor on #0 by rank, then include its content-identical siblings. Matching
        // on text rather than score finds the second copy wherever it ranks, and cannot
        // pull in an unrelated structure, because a data table never shares a sample with
        // the scene's dialogue.
        _heapPools.Clear();
        _poolRecords.Clear();
        _poolNextRecord.Clear();
        _plan.Clear();
        string anchorSample = ranked[0].Sample;

        var selected = new System.Collections.Generic.List<int> { 0 };
        for (int i = 1; i < ranked.Count && selected.Count < 8; i++)
        {
            if (ranked[i].Sample == anchorSample) selected.Add(i);
        }

        // MaxWriteRegions still caps rank-based additions, for diagnosing a scene that
        // ranks poorly. Siblings are exempt: they are the same buffer content, so they
        // carry the safety of the anchor rather than the risk of an unranked guess.
        int extraByRank = Math.Max(_cfg.MaxWriteRegions, 1) - 1;
        for (int i = 1; i < ranked.Count && extraByRank > 0; i++)
        {
            if (selected.Contains(i)) continue;
            selected.Add(i);
            extraByRank--;
        }
        selected.Sort();

        // List the runners-up. If the ranking ever shifts and the scene stops landing at
        // the top, or a third copy appears, these are the evidence needed to see it.
        for (int i = 0; i < Math.Min(12, ranked.Count); i++)
        {
            if (selected.Contains(i)) continue;
            _modLog!.Info($"[POOL] alt #{i} 0x{ranked[i].Base:X} avg={ranked[i].Score / 100.0:F2}: " +
                          $"\"{ranked[i].Sample}\"");
        }

        // Snapshot the script from the anchor before anything is armed for writing.
        // Ordering is load-bearing: the write is destructive and the pool is the only
        // copy, so this is the last moment the scene's real dialogue exists to be read.
        if (_sceneScript.Count == 0)
        {
            var script = CaptureSceneScript(ranked[0].Base, ranked[0].Len,
                                            maxLines: 12, maxChars: 480);
            if (script.Length > 0)
            {
                _sceneScript.AddRange(script);
                _modLog!.Info($"[SCRIPT] captured {script.Length} original lines: " +
                              $"\"{script[0][..Math.Min(script[0].Length, 60)]}\"");
            }
        }

        foreach (int i in selected)
        {
            var c = ranked[i];
            var slots = CapturePoolSlotsSafe(c.Base, c.Len);
            if (slots.Length == 0) continue;
            _heapPools.Add((c.Base, c.Len, slots));
            _poolRecords.Add(GroupSlotsIntoRecords(c.Base, slots));
            // -1 means nothing observed yet, so the next write targets record 0.
            _poolNextRecord.Add(0);
            string why = i == 0 ? "anchor" : (c.Sample == anchorSample ? "twin" : "rank");
            _modLog!.Info(
                $"[POOL] ARM #{i} ({why}) 0x{c.Base:X} len={c.Len} avg={c.Score / 100.0:F2} " +
                $"slots={slots.Length}: \"{c.Sample}\"");
            if (i == 0) LogWatchpointTargets(c.Base, slots, maxSlots: 40);
        }

        if (_heapPools.Count == 0) return;
        _poolIsHeap = true;

        // Before the pending write below, for the reason in Ch. 64: this is the
        // last moment each record still holds its scripted line.
        BuildRecordPlan();
        _modLog!.Info($"[POOL] {_heapPools.Count} regions armed for write");

        string? pending = _lastLlmText;
        if (pending != null) WriteNextRecordReactive(pending);
    }

    /// <summary>
    /// Writes <paramref name="text"/> into every armed heap region. Same in-place rules as
    /// the single-pool writer: never exceed a slot's original run length, and pad with
    /// spaces rather than a NUL so the surrounding function codes stay intact.
    /// </summary>
    /// <summary>
    /// Returns how many bytes of <paramref name="enc"/> to write into a slot of
    /// <paramref name="slotLen"/> bytes, breaking on a word boundary rather than
    /// mid-word.
    /// </summary>
    /// <remarks>
    /// The renderer reads the slot in place and the write cannot exceed the original
    /// line's length, so a long generation is clipped rather than wrapped. A raw
    /// Math.Min cut produced "…you're seriously the onl" on screen — text that reads
    /// as a bug rather than as a shortened line.
    ///
    /// Backing up to the last space costs a word but always ends cleanly. The 60%
    /// floor guards the degenerate case: if the only space sits near the start, a
    /// word-boundary cut would throw away most of the slot, and a hard cut carries
    /// more of the sentence.
    /// </remarks>
    /// <summary>
    /// Group slots that are one byte apart into a single record.
    ///
    /// The ASCII scanner reports printable runs, and it splits on any non-printable byte
    /// — including 0x0A. A two-line speech bubble therefore arrives as two "slots" that
    /// were never two strings: they are one message with a line break in the middle,
    /// stored contiguously. Measured live:
    ///
    ///   0x4250B2FBC7 len=33  "The equipment's kinda crappy, but"
    ///   0x4250B2FBE9 len=25  "they got tons of variety."
    ///
    /// 0xFBC7 + 33 = 0xFBE8, and the next run starts at 0xFBE9. Exactly one byte between
    /// them, and that byte is the newline. Where a genuinely new record starts the gap is
    /// tens of bytes of header instead.
    ///
    /// This is the fix for the oldest cosmetic bug in the mod: writing each fragment
    /// independently put the whole generated line in row one and the same line again in
    /// row two. Rows are not independent, so they cannot be written independently.
    /// </summary>
    private System.Collections.Generic.List<(int Start, int Count)>
        GroupSlotsIntoRecords(nuint poolBase, (int Off, int Len)[] slots)
    {
        var records = new System.Collections.Generic.List<(int, int)>();
        var gap     = new byte[MaxJoinGap];
        int i = 0;
        while (i < slots.Length)
        {
            int start = i++;
            while (i < slots.Length && SameRecord(poolBase, slots[i - 1], slots[i], gap)) i++;
            records.Add((start, i - start));
        }
        return records;
    }

    private const int MaxJoinGap = 32;

    /// <summary>
    /// Build the record layout, recovering short text the slot scanner skipped.
    ///
    /// The scanner requires a run to read as an English sentence before it counts as a
    /// slot, which is right when deciding whether a region is a script and wrong once
    /// inside one. "you gotta" is nine characters and falls below that floor, so it was
    /// never a slot — and so it survived the write and appeared on screen as
    /// "Let's pump it up then, Joker. you gotta— Wait, that ain't it!".
    ///
    /// Inside a record the question is no longer "is this dialogue" — the record has
    /// already been identified — but "is this text the bubble will draw". A three-
    /// character printable run qualifies. Bytes outside the runs are left alone: the
    /// Shift-JIS dash after "you gotta" is a glyph the game renders, not text we own.
    /// </summary>
    private ((int Off, int Len)[] Slots, System.Collections.Generic.List<(int Start, int Count)> Records)
        BuildRecordLayout(nuint poolBase, (int Off, int Len)[] raw)
    {
        var slots   = new System.Collections.Generic.List<(int, int)>(raw.Length + 8);
        var records = new System.Collections.Generic.List<(int, int)>();
        var gapBuf  = new byte[MaxJoinGap];

        int i = 0;
        while (i < raw.Length)
        {
            int start = slots.Count;
            slots.Add(raw[i]);
            i++;

            while (i < raw.Length && SameRecord(poolBase, raw[i - 1], raw[i], gapBuf))
            {
                AddGapRuns(poolBase, raw[i - 1], raw[i], gapBuf, slots);
                slots.Add(raw[i]);
                i++;
            }
            records.Add((start, slots.Count - start));
        }
        return (slots.ToArray(), records);
    }

    /// Add printable runs of three or more characters found between two fragments.
    private void AddGapRuns(nuint poolBase, (int Off, int Len) prev, (int Off, int Len) next,
                            byte[] buf, System.Collections.Generic.List<(int, int)> slots)
    {
        int gapStart = prev.Off + prev.Len;
        int gap      = next.Off - gapStart;
        if (gap <= 1 || gap > MaxJoinGap) return;
        if (!MemoryGuard.TryRead(poolBase + (nuint)gapStart, buf, gap)) return;

        int run = 0;
        for (int b = 0; b <= gap; b++)
        {
            bool printable = b < gap && IsPrintable(buf[b]);
            if (printable) { run++; continue; }
            if (run >= 3) slots.Add((gapStart + b - run, run));
            run = 0;
        }
    }


    /// <summary>
    /// Whether two adjacent text runs belong to the same speech bubble.
    ///
    /// Decided by what is in the gap, not by how wide it is. Measured on a live scene:
    ///
    ///   27B  0A F2 23 00 00 F1 21 F2 05 FF FF F1 41 F7 61 09 ...   message boundary
    ///   12B  0A 79 6F 75 20 67 6F 74 74 61 83 D2                   same bubble
    ///
    /// The second is a newline, the word "you gotta", and a two-byte Shift-JIS dash — no
    /// control codes at all. The first is a block of them: F2 23, F1 21, F2 05 FF FF,
    /// F1 41, F7.
    ///
    /// The previous rule joined runs exactly one byte apart, so it split that bubble in
    /// two, wrote the generated line into the first half, and left "you gotta— Wait, that
    /// ain't it!" showing underneath. Widening it to twelve would have worked on this
    /// scene and been luck: 12 against 27 is a property of these two samples, while the
    /// presence of an F1/F2 control block is a property of the format.
    /// </summary>
    private bool SameRecord(nuint poolBase, (int Off, int Len) prev, (int Off, int Len) next,
                            byte[] buf)
    {
        int gapStart = prev.Off + prev.Len;
        int gap      = next.Off - gapStart;

        if (gap < 1) return false;
        if (gap == 1) return true;              // the bare newline between two rows
        if (gap > MaxJoinGap) return false;     // far enough apart to be padding

        if (!MemoryGuard.TryRead(poolBase + (nuint)gapStart, buf, gap)) return false;
        foreach (byte b in new System.ReadOnlySpan<byte>(buf, 0, gap))
            if (b is 0xF1 or 0xF2 or 0xF7) return false;   // BMD function code: new message
        return true;
    }

    /// <summary>
    /// Distribute <paramref name="text"/> across fragments of the given widths, breaking
    /// on word boundaries.
    ///
    /// Each fragment is a fixed-length slot the renderer reads in place, so this is not
    /// free-form wrapping: fragment k may hold at most widths[k] bytes, and any fragment
    /// the text does not reach must come back empty so the caller can blank it. A bubble
    /// whose second row still held the original dialogue would be worse than one that is
    /// simply short.
    /// </summary>
    private static string[] WrapAcrossFragments(string text, int[] widths)
    {
        var lines = new string[widths.Length];
        string[] words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        int w = 0;

        for (int f = 0; f < widths.Length; f++)
        {
            var line = new System.Text.StringBuilder();
            while (w < words.Length)
            {
                int need = line.Length == 0 ? words[w].Length : line.Length + 1 + words[w].Length;
                if (need > widths[f]) break;
                if (line.Length > 0) line.Append(' ');
                line.Append(words[w]);
                w++;
            }

            // A single word wider than the fragment would never fit, so the loop above
            // would leave this fragment empty and every later one empty too — the whole
            // line lost to one long word. Hard-cut it instead.
            if (line.Length == 0 && w < words.Length)
            {
                line.Append(words[w][..Math.Min(words[w].Length, widths[f])]);
                w++;
            }

            lines[f] = line.ToString();
        }
        return lines;
    }

    /// <summary>
    /// The reactive write: put <paramref name="text"/> in whatever record comes next and
    /// step past it.
    ///
    /// Kept for <c>pregen_lookahead = 0</c>, and used by nothing else. When
    /// pre-generation is on, both paths writing would have them competing for the same
    /// records with different text — so the reactive path stands down rather than being
    /// deleted, because it is the fallback if pre-generated lines ever read as
    /// disconnected from the scene.
    /// </summary>
    private int WriteNextRecordReactive(string text)
    {
        if (_cfg.PregenLookahead > 0) return 0;
        if (_poolNextRecord.Count == 0) return 0;

        int target  = _poolNextRecord[0];
        int written = WriteRecord(target, text);
        if (written > 0)
            for (int r = 0; r < _poolNextRecord.Count; r++) _poolNextRecord[r] = target + 1;
        return written;
    }

    /// <summary>
    /// Write one record, by index, into every armed region.
    ///
    /// The index is the same in all regions because they are copies of one script with
    /// identical record widths. Letting each region track its own cursor produced exactly
    /// the drift you would expect — region 0 writing record 8 while region 1 wrote record
    /// 7 — which puts the same sentence on two different lines of the same scene.
    ///
    /// Write per record, not per slot: the unit the player sees is the bubble, and a
    /// bubble spans every fragment up to the next header gap. And one record, not all of
    /// them — that breadth is what filled the text log with a single sentence repeated at
    /// every width, and it was also the entire blast radius.
    /// </summary>
    private unsafe int WriteRecord(int index, string text)
    {
        if (!_cfg.PoolWriteEnabled)
        {
            _modLog!.Info($"[POOL] write disabled by config ← \"{text[..Math.Min(text.Length, 50)]}\"");
            return 0;
        }
        if (_heapPools.Count == 0) return 0;

        int totalSlots = 0, regions = 0;

        for (int r = 0; r < _heapPools.Count; r++)
        {
            var (poolBase, poolLen, slots) = _heapPools[r];

            // Validate the whole range, not just the first bytes. The region was captured
            // on an earlier tick and the game may have freed it since; writing through a
            // stale base would fault fatally the same way the scan did.
            if (!MemoryGuard.IsWritable(poolBase, poolLen)) continue;

            byte* p = (byte*)poolBase;
            int wrote = 0;

            var records = _poolRecords[r];
            int target  = index;
            if (target < 0 || target >= records.Count) continue;

            {
                (int start, int count) = records[target];

                var widths = new int[count];
                for (int k = 0; k < count; k++) widths[k] = slots[start + k].Len;

                string[] lines = WrapAcrossFragments(text, widths);

                int mirrored = 0;
                for (int k = 0; k < count; k++)
                {
                    (int off, int len) = slots[start + k];
                    byte[] enc = System.Text.Encoding.ASCII.GetBytes(lines[k]);
                    int    wl  = Math.Min(enc.Length, len);

                    // Snapshot before overwriting. The twin is identified by still holding
                    // these exact bytes, so this has to be captured while they exist —
                    // the same ordering constraint as the scene-script capture in Ch. 64.
                    byte[] original = new byte[len];
                    bool   haveOriginal = len <= _mirrorBuf.Length &&
                                          MemoryGuard.TryRead(poolBase + (nuint)off, original, len);

                    if (wl > 0)
                        fixed (byte* src = enc)
                            System.Buffer.MemoryCopy(src, p + off, len, wl);

                    // Blank the tail. Anything left unwritten is the original dialogue,
                    // and a row of scripted text under a row of generated text reads far
                    // worse than a short line. The newline between fragments is outside
                    // [off, off+len) and is never touched.
                    for (int i = wl; i < len; i++) p[off + i] = (byte)' ';

                    if (haveOriginal && MirrorToTwin(poolBase + (nuint)off, original, len, p + off))
                        mirrored++;
                }

                if (mirrored > 0)
                    _modLog!.Info($"[TWIN] mirrored {mirrored}/{count} rows at delta " +
                                  $"0x{Math.Abs(_twinDelta):X}");

                // The cursor no longer moves here. With a plan, which record to write is
                // chosen by index rather than taken from the cursor, so writing must not
                // also mean "the player advanced" — that conflation is what made three
                // generations land on the record the player was reading. The cursor now
                // means one thing only: how far the interpreter has actually got.
                _modLog!.Info($"[POOL] region {r} record {target}/{records.Count} " +
                              $"@0x{poolBase + (nuint)slots[start].Off:X} rows={count}");
                wrote++;
            }

            if (wrote > 0) { totalSlots += wrote; regions++; }
        }

        _modLog!.Info($"[POOL] wrote {totalSlots} records across {regions} regions ← " +
                      $"\"{text[..Math.Min(text.Length, 50)]}\"");
        return totalSlots;
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

        // 20000, not 512. Slots are captured from the region base outward, so a low cap
        // covered only the first sliver of a multi-megabyte region — text deeper in was
        // scored but never made writable, which defeated the whole point of scanning it.
        while (pos < scanLen && slots.Count < 20000)
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

        // One shot per session. These are the item/skill tables, not scene dialogue, so
        // rewriting them on every message corrupts more descriptions for no new
        // information — a single write still answers whether the renderer picks it up.
        if (!_poolIsHeap)
        {
            if (_poolWriteDone) return 0;
            _poolWriteDone = true;
        }

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
                    // Gated: these fire on every message and the pool write path no longer
                    // depends on any of them.
                    unsafe
                    {
                        if (!_cfg.StructDiffEnabled) goto skipDiag;
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

                        // session+0xD0 holds a per-message pointer array; it has been
                        // populated on every run, unlike +0xC8.
                        ProbeD0Array(session);

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

                        skipDiag: ;
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

    /// <summary>
    /// The reactive generation, fired by a message dispatch. Inert while pre-generation
    /// is on.
    ///
    /// Leaving both running does not merely waste inference — the server answers one
    /// request at a time and 429s the rest, so a dispatch arriving mid-queue would take
    /// the slot the queue was about to use and neither path would keep up. Worse, this
    /// path fires for every msgId including ones that are not spoken lines: msgId 0x348
    /// recurred through a whole hang-out and consumed a record each time, which is how a
    /// 22-record scene ran out at 22/22 with lines still to come.
    /// </summary>
    private async System.Threading.Tasks.Task DispatchMsgLlmAsync(
        SocialLinkSnapshot snap, ushort msgId, nuint session)
    {
        if (_cfg.PregenLookahead > 0) return;

        using var cts = new System.Threading.CancellationTokenSource(
            TimeSpan.FromSeconds(_cfg.TimeoutSeconds));
        try
        {
            // Pydantic rejects context over 1024 chars with a 422, which would surface as
            // a silent generation failure rather than an obvious error. The script is
            // already budgeted well under this; the clamp guards against the pieces
            // growing independently and quietly crossing the limit.
            string ctx = Memory.ContextBuilder.Build(snap) + ScriptContext() + $" [msg_0x{msgId:X}]";
            if (ctx.Length > 1000) ctx = ctx[..1000];
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
            int  heapWrites = WriteNextRecordReactive(text);
            _modLog!.Info($"[LLM] msgId=0x{msgId:X} — ptrWrite={(wrote ? "OK" : "skip")} " +
                          $"poolWrites={poolWrites} heapWrites={heapWrites}");
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

    /// <summary>
    /// Log the record the interpreter is reading, whenever it changes.
    ///
    /// Sampling on the poll tick rather than logging from the stub is not a shortcut: the
    /// hook fires once per character of every message in the game, so anything managed on
    /// that path would run tens of thousands of times a second. The stub writes two words
    /// to a fixed buffer and nothing else; this reads the buffer at 500 ms and reports
    /// only transitions.
    ///
    /// The consequence is that short-lived records can be missed entirely. That is fine
    /// for verification — what needs proving is that the pointer is real, in the heap, and
    /// changes line to line — and it is the wrong mechanism for actually driving writes,
    /// which will key off the record changing rather than a timer.
    /// </summary>
    private void ReportWatchedRecord()
    {
        if (_msgWatch is null) return;

        // Opening the backlog re-reads every past line at once, which is a legitimate
        // burst rather than a bug — but logging all of it drowns the console and puts
        // dozens of writes on a tick the game is waiting behind. Every entry is still
        // processed for the cursor and the twin delta; only the log is capped.
        const int MaxLogged = 6;
        int logged = 0, suppressed = 0;

        // Outside a hang-out none of this is useful: no pool to arm, no plan to advance,
        // no twin worth learning. The interpreter is busiest exactly then — the title
        // screen, save loading and menus push far more text than a conversation does — and
        // previewing every record through ReadProcessMemory and logging it was enough to
        // be felt as stutter during boot. Draining without inspecting keeps the sampler's
        // queue from filling while costing nothing.
        if (!_sessionActive)
        {
            _msgWatch.DrainSeen();
            return;
        }

        foreach ((nuint record, int cursor) in _msgWatch.DrainSeen())
        {
            if (record == _lastWatchedRecord) continue;
            _lastWatchedRecord = record;

            // Preview from the record base, not from record+cursor. The two are captured
            // at slightly different points and are not a coherent pair: the cursor has
            // usually run on to wherever that message ended, which is trailing control
            // bytes rather than text. The base is the stable half.
            string text = ReadRecordPreview(record);

            // Whether the record falls inside a region the pool heuristic armed is the
            // whole question this hook exists to answer. "in-pool" means the two agree; a
            // scene line reported as "elsewhere" means the ranking missed the live buffer.
            string where = AdvancePoolCursor(record) ? "in-pool  " : "elsewhere";

            // Arm from the read itself. Gated on an active hang-out because the
            // interpreter serves every string in the game — menus, item names, the
            // newspaper — and arming on the first record seen anywhere would point the
            // writer at a UI buffer.
            //
            // Requiring a dialogue-length preview is the second gate: a scene line is a
            // sentence, and a twelve-character floor rejects labels without pretending to
            // be a classifier.
            if (_sessionActive && text.Length >= 12)
            {
                TryArmFromRecord(record);
            }

            LearnTwinDelta(record, text);

            if (logged++ < MaxLogged)
                _modLog!.Info($"[WATCH] {where} record=0x{record:X} cursor=0x{cursor:X} \"{text}\"");
            else
                suppressed++;
        }

        if (suppressed > 0)
            _modLog!.Info($"[WATCH] +{suppressed} more records this tick (backlog scroll)");
    }

    /// <summary>
    /// Write every record whose text has arrived but has not reached game memory yet.
    ///
    /// Split from generation because the two fail differently and at different times: a
    /// 503 loses the text, a freed region loses the write. Keeping them separate means a
    /// write that fails on one tick is simply retried on the next, with the generated
    /// line still in hand.
    ///
    /// Records already read by the interpreter are skipped. That is the Ch. 71 bug
    /// restated as a rule — overwriting what the player is looking at is what made the
    /// sentence change under them.
    /// </summary>
    private void FlushReadyRecords()
    {
        lock (_writeLock)
        foreach (RecordPlan record in _plan.ToArray())
        {
            if (record.State != RecordState.Ready || record.Generated is null) continue;
            if (!record.IsWritable) continue;

            // Behind the player is not a write, it is vandalism of the backlog. Observed
            // as "#0 written, -32 ahead of the player": the request was issued while the
            // cursor was at 0, the player advanced 32 records during the round trip, and
            // the answer arrived to overwrite a line spoken half a minute earlier.
            //
            // The freeze in AdvancePoolCursor catches this only for records the
            // interpreter reported; a scene skipped fast enough leaves gaps it never saw.
            if (record.Index < CurrentRecord())
            {
                record.State = RecordState.Rendered;   // out of the way for good
                _modLog!.Info($"[PREGEN] #{record.Index} discarded — player is at " +
                              $"{CurrentRecord()}");
                continue;
            }

            if (WriteRecord(record.Index, record.Generated) > 0)
            {
                record.State      = RecordState.Written;
                record.WasWritten = true;
                _modLog!.Info($"[PREGEN] #{record.Index} written, " +
                              $"{record.Index - CurrentRecord()} ahead of the player");
            }
        }
    }

    /// <summary>
    /// Report how much of a scene was actually replaced, and which records were not.
    ///
    /// Coverage is the number that says whether this works, and it had to be counted by
    /// hand from the log until now — 16 of 22, with the gaps at 3-6 and 14-15. Both gaps
    /// were bursts where the player advanced faster than inference could keep up, which is
    /// a fact about pace rather than a fault, and it is invisible without this line.
    /// </summary>
    private void LogSceneCoverage()
    {
        if (_plan.Count == 0) return;

        int written = 0, pending = 0, tooSmall = 0;
        var gaps = new System.Text.StringBuilder();
        foreach (RecordPlan record in _plan)
        {
            bool replaced = record.WasWritten;
            if (replaced) { written++; continue; }
            if (record.Original.Length < 8) continue;   // never a candidate

            // Records we deliberately never queue are not failures, and counting them as
            // misses understated the result: the first report read 17/22 when five of the
            // gaps were short interjections the queue was told to leave alone.
            if (record.Capacity < 24) { tooSmall++; continue; }

            pending++;
            if (gaps.Length < 60) gaps.Append(gaps.Length > 0 ? ", " : "").Append('#').Append(record.Index);
        }

        int candidates = written + pending;
        int percent    = candidates == 0 ? 0 : written * 100 / candidates;
        _modLog!.Info($"[SCENE] replaced {written}/{candidates} records ({percent}%)" +
                      (tooSmall > 0 ? $", {tooSmall} too short to try" : "") +
                      (gaps.Length > 0 ? $" — missed {gaps}" : ""));
    }

    /// Index the player is currently at, as far as the interpreter has told us.
    private int CurrentRecord() =>
        _poolNextRecord.Count > 0 ? Math.Max(0, _poolNextRecord[0] - 1) : 0;

    /// <summary>
    /// Context for one specific record: the scene, the line being replaced, and what has
    /// already been said.
    ///
    /// The line being replaced is the part the reactive path never had. It was sending
    /// "hang-out with Ryuji at Protein Lovers gym" for every line of the scene, so the
    /// model wrote ambience and repeated itself — two consecutive generations came back
    /// as "You comin' back here every week like me now?" and "You comin' here more often
    /// now?", which are the same sentence.
    /// </summary>
    private string BuildRecordContext(SocialLinkSnapshot snap, RecordPlan record)
    {
        var ctx = new System.Text.StringBuilder(ContextBuilder.Build(snap));

        ctx.Append(" The line you are replacing is: \"").Append(record.Original).Append('"');
        ctx.Append(" Say something with the same purpose, in your own words.");

        // What the player has actually heard, most recent last. Four lines rather than
        // two: with two, consecutive generations came back as "Yeah, it was a sweet gym"
        // followed by "You think you can just crash our gym session for free, huh?" —
        // each fine alone, and plainly not the same conversation.
        //
        // Generated text where it exists, the script where it does not. A record the
        // queue missed still played on screen, so leaving it out would describe a
        // conversation with a hole in it.
        // Drop the oldest lines until it fits, rather than truncating the finished string.
        // The server rejects anything over 1024 characters, and a tail cut would remove
        // "Continue from the last one." while keeping the least relevant history —
        // exactly the wrong end to lose.
        const int Budget = 1000;
        int from = Math.Max(0, record.Index - 4);

        while (true)
        {
            var history = new System.Text.StringBuilder();
            if (from < record.Index)
            {
                // Attribution matters more than it looks. These are the speaker's own
                // earlier lines, and calling them "the conversation so far" made the model
                // read them as the other person's: replacing Ryuji explaining the pricing,
                // it answered him instead — "Fair enough, bro, that's a small price."
                history.Append(" You have just said, oldest first:");
                for (int i = from; i < record.Index; i++)
                {
                    string said = _plan[i].Generated ?? _plan[i].Original;
                    if (said.Length > 0) history.Append(" \"").Append(said).Append('"');
                }
                history.Append(" Carry on from your own last line — you are still talking, " +
                               "not replying to someone.");
            }

            if (ctx.Length + history.Length <= Budget || from >= record.Index)
            {
                string full = ctx.Append(history).ToString();
                return full.Length > Budget ? full[..Budget] : full;
            }

            from++;   // give up the oldest line and try again
        }
    }

    /// <summary>
    /// Ask the server for one record's replacement line, off the poll thread.
    ///
    /// The record is passed rather than looked up on completion: by the time the answer
    /// arrives the player may have moved on, the plan may have been rebuilt, or the
    /// session may have ended, and resolving an index against a list that has since
    /// changed is how a generated line lands on the wrong bubble.
    /// </summary>
    private void RequestForRecord(SocialLinkSnapshot snap, RecordPlan record)
    {
        _ = Task.Run(async () =>
        {
            using var cts = new CancellationTokenSource(TimeSpan.FromSeconds(_cfg.TimeoutSeconds));
            try
            {
                var req = new Server.GenerateRequest
                {
                    ConfidantId   = snap.ConfidantId,
                    Rank          = snap.RankLevel,
                    Context       = BuildRecordContext(snap, record),
                    CharacterName = Memory.ConfidantNames.Resolve(snap.ConfidantId),
                    MaxChars      = record.Capacity,
                };

                string text = await _llmClient!.GenerateAsync(req, cts.Token);
                if (string.IsNullOrWhiteSpace(text))
                {
                    // Empty means the server had nothing that fit — usually a short record
                    // no complete sentence will go into. Retrying that produces the same
                    // answer while holding a queue slot the whole scene, so it gets a
                    // couple of chances and then keeps its scripted line.
                    GiveUpOrRetry(record, "no line fit the record");
                    return;
                }

                record.Generated = text;
                record.State     = RecordState.Ready;
                _modLog!.Info($"[PREGEN] #{record.Index} ready ({text.Length}/{record.Capacity}): \"{text}\"");

                // Write it now rather than on the next poll tick. Waiting added up to a
                // full 500ms between a line existing and it reaching memory, on top of
                // ~1.5s of inference — and that delay lands exactly where it hurts, at the
                // front of a scene where the queue has not banked any buffer yet.
                FlushReadyRecords();
            }
            catch (Server.InferenceInFlightException)
            {
                // The server was busy with another record. Not an error — put it back and
                // the next tick picks it up, which is what makes the queue self-pacing.
                record.State = RecordState.Pending;
            }
            catch (OperationCanceledException)
            {
                GiveUpOrRetry(record, "timed out");
            }
            catch (Exception ex) when (ex is not OutOfMemoryException)
            {
                GiveUpOrRetry(record, ex.Message);
            }
        });
    }

    /// <summary>
    /// Put a failed record back in the queue, or stop trying.
    ///
    /// A busy server and a record no sentence will fit fail the same way here, and only
    /// one of them is worth retrying. Two attempts covers the transient case; beyond that
    /// the record keeps its scripted line, which is a good line rather than a missing one.
    /// </summary>
    private void GiveUpOrRetry(RecordPlan record, string why)
    {
        const int MaxAttempts = 2;

        if (++record.Attempts >= MaxAttempts)
        {
            record.State = RecordState.Rendered;   // out of the queue for good
            _modLog!.Info($"[PREGEN] #{record.Index} keeping the script after " +
                          $"{record.Attempts} attempts ({why})");
            return;
        }
        record.State = RecordState.Pending;
    }

    /// <summary>
    /// Keep the next few records generated, without waiting for the player to reach them.
    ///
    /// Runs on the poll tick. Everything it needs was known at arm time, so this asks a
    /// question the reactive path could not: not "what should Ryuji say now" but "what
    /// should he say at record 14", answerable long before record 14 is on screen.
    ///
    /// One request is issued per tick at most. The server processes one at a time and
    /// answers 429 to anything overlapping, so firing the whole window at once would
    /// convert a queue into a pile of rejections.
    /// </summary>
    private void PumpPregen(SocialLinkSnapshot snap)
    {
        // Below this, a complete sentence in character does not fit and the server
        // correctly returns nothing. Measured: records of 15-21 chars produced
        // "You can grab a" and "Guess we good to" before fragments were rejected.
        const int MinGeneratableChars = 24;

        int lookahead = _cfg.PregenLookahead;
        if (lookahead <= 0 || _plan.Count == 0) return;

        int from = _poolNextRecord.Count > 0 ? Math.Max(0, _poolNextRecord[0]) : 0;
        int to   = Math.Min(_plan.Count, from + lookahead);

        for (int i = from; i < to; i++)
        {
            RecordPlan record = _plan[i];
            if (record.State != RecordState.Pending) continue;

            // Records whose scripted line is too short to be a spoken line are skipped
            // rather than generated for. A three-character record is a fragment of UI,
            // and replacing it wastes a slow inference on something nobody reads as
            // dialogue.
            if (record.Original.Length < 8 || record.Capacity < MinGeneratableChars)
            {
                record.State = RecordState.Rendered;   // permanently out of the way
                continue;
            }

            record.State = RecordState.InFlight;
            RequestForRecord(snap, record);
            return;   // one per tick
        }
    }

    /// <summary>
    /// Start the queue immediately after arming, without waiting for the next poll tick.
    ///
    /// The front of a scene is where coverage is always lost — #3 through #6 in the last
    /// three sessions — because the player is reading at full speed while the queue has
    /// banked nothing. Half a second of idling before the first request is spent at the
    /// exact moment it can least be afforded.
    /// </summary>
    private void KickPregen()
    {
        if (_cfg.PregenLookahead <= 0 || _plan.Count == 0) return;
        if (!_reader!.TryResolve(out nuint session)) return;

        SocialLinkSnapshot? snap = SocialLinkReader.TryReadFromPtr(session);
        if (snap is not null) PumpPregen(snap);
    }

    /// <summary>
    /// Build the scene plan from region 0: one entry per record, with its capacity and
    /// the scripted line it currently holds.
    ///
    /// Ordering is load-bearing in the same way as Ch. 64's script capture. The original
    /// text only exists until the first write, and it is the best context the model can
    /// be given about this specific moment — not the scene in general, but the line it is
    /// replacing. Reading it later means reading whatever we ourselves wrote.
    /// </summary>
    private void BuildRecordPlan()
    {
        _plan.Clear();
        if (_heapPools.Count == 0 || _poolRecords.Count == 0) return;

        (nuint poolBase, _, (int Off, int Len)[] slots) = _heapPools[0];
        var records = _poolRecords[0];

        for (int i = 0; i < records.Count; i++)
        {
            (int start, int count) = records[i];

            int capacity = 0;
            var line     = new System.Text.StringBuilder();
            for (int k = 0; k < count; k++)
            {
                (int off, int len) = slots[start + k];
                capacity += len;

                // Rows of one bubble join with a space, not a newline: the model is being
                // shown a sentence, and the row split is a rendering detail it has no use
                // for. Wrapping puts the breaks back on the way out.
                if (line.Length > 0) line.Append(' ');
                line.Append(AsciiPreview(poolBase + (nuint)off, len));
            }

            _plan.Add(new RecordPlan
            {
                Index    = i,
                Capacity = capacity + (count - 1),
                Original = line.ToString().Trim(),
            });
        }

        LogSlotSeparators(poolBase, slots);

        _modLog!.Info($"[PLAN] {_plan.Count} records; " +
                      $"capacity {MinCapacity()}-{MaxCapacity()} chars");
        for (int i = 0; i < Math.Min(4, _plan.Count); i++)
            _modLog!.Info($"[PLAN]   {_plan[i]}");
    }

    /// <summary>
    /// Dump the bytes between consecutive text runs, to find out what actually separates
    /// the rows of one bubble from the start of the next message.
    ///
    /// Grouping currently joins runs exactly one byte apart, because the one separator
    /// ever inspected in a hex editor was a bare 0x0A. That is not the whole story: a
    /// bubble rendered as
    ///
    ///     "Let's pump it up then, Joker. you gotta— Wait, that ain't it!"
    ///
    /// had its rows twelve bytes apart, so grouping saw two records, wrote the generated
    /// line into the first, and left the rest of the script showing underneath.
    ///
    /// Widening the threshold by guesswork is how the 211-slot write happened. Twelve is
    /// close to the ~27 that separates genuine messages, and the difference has to be read
    /// out of the bytes rather than assumed from the distance.
    /// </summary>
    private void LogSlotSeparators(nuint poolBase, (int Off, int Len)[] slots)
    {
        var buf = new byte[32];
        for (int i = 1, shown = 0; i < slots.Length && shown < 10; i++)
        {
            int gapStart = slots[i - 1].Off + slots[i - 1].Len;
            int gap      = slots[i].Off - gapStart;
            if (gap <= 1 || gap > 32) continue;   // 1 is the known newline; huge gaps are padding

            if (!MemoryGuard.TryRead(poolBase + (nuint)gapStart, buf, gap)) continue;

            var hex = new System.Text.StringBuilder(gap * 3);
            for (int b = 0; b < gap; b++) hex.Append(buf[b].ToString("X2")).Append(' ');

            _modLog!.Info($"[GAP] {gap,2}B between #{i - 1} and #{i}: {hex}");
            shown++;
        }
    }

    private int MinCapacity()
    {
        int min = int.MaxValue;
        foreach (var p in _plan) min = Math.Min(min, p.Capacity);
        return min == int.MaxValue ? 0 : min;
    }

    private int MaxCapacity()
    {
        int max = 0;
        foreach (var p in _plan) max = Math.Max(max, p.Capacity);
        return max;
    }

    /// <summary>
    /// Characters the next record to be written can hold, or 0 when there is no target.
    ///
    /// This is the sum of the record's fragment widths plus one per join: fragments are
    /// the rows of one bubble and the generated line is wrapped across all of them, so
    /// the budget is the whole bubble rather than its first row.
    ///
    /// Region 0 decides. The armed regions are copies of the same script and their
    /// records have identical widths, so asking any one of them gives the same answer.
    /// </summary>
    private int NextRecordCapacity()
    {
        if (_poolRecords.Count == 0 || _heapPools.Count == 0) return 0;

        var records = _poolRecords[0];
        int target  = _poolNextRecord[0];
        if (target < 0 || target >= records.Count) return 0;

        (int start, int count) = records[target];
        (_, _, (int Off, int Len)[] slots) = _heapPools[0];

        int capacity = 0;
        for (int k = 0; k < count; k++) capacity += slots[start + k].Len;
        return capacity + (count - 1);   // the newline between rows carries a word break
    }

    /// True when the address lies inside a region already armed for writing.
    private bool InArmedPool(nuint addr)
    {
        foreach ((nuint poolBase, int poolLen, _) in _heapPools)
            if (addr >= poolBase && addr < poolBase + (nuint)poolLen) return true;
        return false;
    }

    /// <summary>
    /// Arm the region containing a record the interpreter just read.
    ///
    /// This replaces the heap scan, and the difference is not an optimisation — it is a
    /// different question. The scan asked "which of 4030 regions looks most like
    /// dialogue?", walked 1.4 GB scoring English, and took 33 seconds, during which the
    /// first six lines of the scene played untouched. This asks "what region is that
    /// address in?", which VirtualQuery answers in microseconds, about an address the
    /// game supplied by reading it.
    ///
    /// Validation still happens, because the interpreter serves every string in the game
    /// and the first record seen could be a menu label. But validating one candidate the
    /// game pointed at is a different problem from searching for it.
    /// </summary>
    private void TryArmFromRecord(nuint record)
    {
        const int MinSlots  = 6;
        const int MinRegion = 0x4000;         // 16 KB
        const int MaxRegion = 0x400000 * 8;   // 32 MB; scene pools measured 45 KB - 1.4 MB

        // A scene's script is small. Measured across five sessions the gym pool held 36
        // text runs every time, while the pool the game reads alongside it — menu prompts,
        // "AHang out with him", "ACheck bond with Ryuji" — held 340 to 409.
        //
        // Without this the two take turns arming each other. Reads interleave between them
        // all through a scene, so consecutive outside reads happen constantly, and every
        // flip rebuilt the plan and discarded everything generated so far. Coverage fell to
        // 50%, with records 0-6 written and then wiped.
        const int MaxSceneSlots = 120;

        if (InArmedPool(record) || IsTwinOfArmed(record))
        {
            _outsideReads = 0;
            return;
        }

        // A hang-out is not one pool. The first scene of a session was a conversation on
        // the street and armed 0x41DBE73000; the gym pool appeared fourteen seconds later
        // at 0x4250C8B13C and never got a look, because arming had already happened.
        //
        // Reads landing outside the armed region are how a scene change announces itself.
        // Two in a row, because one stray read is a menu or a name plate, while a scene
        // that has genuinely moved keeps reading from its new pool.
        if (_heapPools.Count > 0 && ++_outsideReads < 2) return;

        (bool ok, nuint regionBase, nuint regionSize, uint state, uint _) =
            MemoryGuard.QueryRegion(record);
        const uint MEM_COMMIT = 0x1000;
        if (!ok || state != MEM_COMMIT) return;
        if (regionSize < MinRegion || regionSize > MaxRegion) return;
        if (!MemoryGuard.IsWritable(regionBase, (int)Math.Min(regionSize, 0x1000))) return;

        var slots = CapturePoolSlotsSafe(regionBase, (int)regionSize);
        if (slots.Length < MinSlots || slots.Length > MaxSceneSlots)
        {
            _modLog!.Info($"[ARM] 0x{regionBase:X} rejected — {slots.Length} text runs, " +
                          $"outside {MinSlots}-{MaxSceneSlots}. Not a scene script.");
            return;
        }

        // Re-arming the region already armed would rebuild the plan and throw away every
        // line generated for it — the same damage as a flip, from a no-op.
        if (_heapPools.Count > 0 && _heapPools[0].Base == regionBase)
        {
            _outsideReads = 0;
            return;
        }

        // Exactly one region is armed, and it is replaced rather than added to. Arming two
        // and writing the same record index into both assumed they were copies. They were
        // not — 207 records against 180, two unrelated pools, so index N named a different
        // line in each.
        //
        // The second copy is still written, by MirrorToTwin, which finds it at a learned
        // offset and verifies byte-for-byte before touching anything. That is the
        // mechanism that actually knows two addresses hold the same line.
        lock (_writeLock)
        {
        _outsideReads = 0;
        _twinDelta    = 0;   // learned for the pool being replaced; meaningless for this one
        _heapPools.Clear();
        _poolRecords.Clear();
        _poolNextRecord.Clear();

        var (layout, records) = BuildRecordLayout(regionBase, slots);
        _heapPools.Add((regionBase, (int)regionSize, layout));
        _poolRecords.Add(records);
        _poolNextRecord.Add(0);
        _poolIsHeap = true;

        _modLog!.Info($"[ARM] 0x{regionBase:X} len={regionSize} slots={slots.Length} " +
                      "— from a live read, no scan");

        BuildRecordPlan();
        }

        // Outside the lock: the request goes to the thread pool and its continuation takes
        // the same lock to flush.
        KickPregen();
    }

    /// <summary>
    /// True when the address is the armed pool's second copy, at the learned offset.
    ///
    /// Without this every twin read looks like a read outside the armed region and would
    /// trigger a re-arm onto the copy — the two pools then take turns evicting each other
    /// for the length of the scene.
    /// </summary>
    private bool IsTwinOfArmed(nuint addr)
    {
        if (_twinDelta == 0) return false;
        return InArmedPool((nuint)((long)addr + _twinDelta))
            || InArmedPool((nuint)((long)addr - _twinDelta));
    }

    /// <summary>
    /// Learn where the second copy of a record lives, from two reads of the same text.
    ///
    /// The interpreter reads both copies of a line within about a millisecond, and the
    /// sampler catches the pair as two consecutive distinct records with identical
    /// preview text. Their difference is the delta, and it has held constant across every
    /// pair within a run.
    ///
    /// This replaces content-matched twin arming, which searched a ranked list for a
    /// region with the same sample and missed entirely in one run — leaving a single
    /// region armed, every write landing in the text log, and the bubble showing the
    /// original script. Ranking cannot see a relationship the game never expresses as
    /// similarity of score; the hook observes it directly.
    /// </summary>
    private void LearnTwinDelta(nuint record, string text)
    {
        // Short or empty previews are not evidence: control-code runs and blank records
        // collide constantly, and a delta learned from one would aim writes at nothing.
        if (text.Length >= 12 && text == _lastWatched.Text && record != _lastWatched.Addr)
        {
            long delta = (long)record - (long)_lastWatched.Addr;
            if (delta != _twinDelta)
            {
                _twinDelta = delta;
                _modLog!.Info($"[TWIN] delta={(delta < 0 ? "-" : "")}0x{Math.Abs(delta):X} " +
                              $"from 0x{_lastWatched.Addr:X} / 0x{record:X} \"{text}\"");
            }
        }
        _lastWatched = (record, text);
    }

    /// <summary>
    /// Copy a just-written span into the twin, but only where the twin still holds what
    /// the target held before the write.
    ///
    /// The delta comes from observation and observation can be wrong, so this never
    /// writes on the strength of arithmetic alone. The caller passes the original bytes;
    /// if the mirror does not match them byte for byte, the address is not the same line
    /// and nothing is written. That check is what keeps a bad delta from turning into the
    /// 211-slot data-table write of Ch. 63 by another route.
    /// </summary>
    private unsafe bool MirrorToTwin(nuint addr, byte[] original, int len, byte* written)
    {
        if (_twinDelta == 0 || len <= 0 || len > _mirrorBuf.Length) return false;

        // Both directions, because the sign of the delta is an accident of which copy the
        // interpreter happened to read first. LearnTwinDelta computes (this - previous),
        // and the game alternates, so the log shows the same offset as +0x33D83600 and
        // -0x33D83600 within one scene. Trying one direction meant the mirror silently
        // refused about half the time — it fired zero times in a whole hang-out, and the
        // text log showed the original script throughout.
        //
        // Trying both is safe precisely because the guard below does the deciding: an
        // address only gets written if it still holds the bytes the target held.
        return TryMirrorAt((nuint)((long)addr - _twinDelta), original, len, written)
            || TryMirrorAt((nuint)((long)addr + _twinDelta), original, len, written);
    }

    private unsafe bool TryMirrorAt(nuint mirror, byte[] original, int len, byte* written)
    {
        if (!MemoryGuard.TryRead(mirror, _mirrorBuf, len)) return false;

        for (int i = 0; i < len; i++)
            if (_mirrorBuf[i] != original[i]) return false;

        if (!MemoryGuard.IsWritable(mirror, len)) return false;

        byte* dst = (byte*)mirror;
        for (int i = 0; i < len; i++) dst[i] = written[i];
        return true;
    }

    /// <summary>
    /// Point a pool at the record after the one just rendered. Returns whether
    /// <paramref name="addr"/> landed in an armed pool at all.
    ///
    /// The hook reports the record's base, which sits a header ahead of the text — the
    /// one measured live had its first character at +0x28. So the match is not
    /// containment but "the first record whose text begins at or shortly after this
    /// address", within a window wide enough to clear any header seen so far.
    ///
    /// Advancing on render rather than on write is deliberate. A write that never gets
    /// displayed (the player skipped, the scene branched) must not consume a record, or
    /// the mod would drift one line ahead of the game and stay there for the rest of the
    /// scene.
    /// </summary>
    private bool AdvancePoolCursor(nuint addr)
    {
        const int HeaderWindow = 0x80;
        bool inPool = false;

        for (int r = 0; r < _heapPools.Count; r++)
        {
            (nuint poolBase, int poolLen, (int Off, int Len)[] slots) = _heapPools[r];
            if (addr < poolBase || addr >= poolBase + (nuint)poolLen) continue;
            inPool = true;

            int offset  = (int)(addr - poolBase);
            var records = _poolRecords[r];

            for (int i = 0; i < records.Count; i++)
            {
                int textOff = slots[records[i].Start].Off;
                if (textOff < offset) continue;
                if (textOff - offset > HeaderWindow) break;   // ordered; no closer match ahead

                // Never move backwards. The backlog re-reads earlier records while the
                // player scrolls it, and treating that as progress would rewind the write
                // target onto lines already spoken.
                if (i + 1 > _poolNextRecord[r]) _poolNextRecord[r] = i + 1;

                // Freeze it. Once the interpreter has read a record the player has seen
                // it, and rewriting it is the bug they described as the text switching.
                // Backlog re-reads freeze too, which is correct — a line in the log has
                // certainly been shown.
                if (i < _plan.Count && _plan[i].State != RecordState.Rendered)
                {
                    _plan[i].State = RecordState.Rendered;
                    _modLog!.Info($"[PLAN] #{i} rendered — frozen");
                }
                break;
            }
        }
        return inPool;
    }

    /// <summary>
    /// First readable English run at or shortly after <paramref name="record"/>.
    ///
    /// Read through ReadProcessMemory rather than by dereferencing. The pointer arrives
    /// from a stub that captured it inside the game's own loop, and by the time the poll
    /// tick looks at it the message may have been freed — checking IsReadable and then
    /// walking raw bytes races the allocator in exactly the way that killed the scan in
    /// Ch. 61, and an access violation there is uncatchable.
    ///
    /// The scan starts at the base and skips forward because a record begins with a
    /// header: the one measured live had its text at +0x28, and that offset is not
    /// guaranteed to be constant across message kinds.
    /// </summary>
    private string ReadRecordPreview(nuint record)
    {
        const int Window = 128;
        if (!MemoryGuard.TryRead(record, _recordBuf, Window)) return "";

        for (int start = 0; start < Window; start++)
        {
            if (!IsPrintable(_recordBuf[start])) continue;

            int end = start;
            while (end < Window && IsPrintable(_recordBuf[end])) end++;
            if (end - start < 8) { start = end; continue; }

            var sb = new System.Text.StringBuilder(end - start);
            for (int i = start; i < end; i++) sb.Append((char)_recordBuf[i]);
            string candidate = sb.ToString();
            if (IsEnglishString(candidate)) return candidate;
            start = end;
        }
        return "";
    }

    private async Task PollLoopAsync(CancellationToken ct)
    {
        nuint lastSession = 0;

        while (await _timer!.WaitForNextTickAsync(ct))
        {
            ReportWatchedRecord();

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
                    _poolWriteDone       = false;
                    _poolIsHeap          = false;
                    _heapPools.Clear();
                    _poolRecords.Clear();
                    _poolNextRecord.Clear();
                    LogSceneCoverage();
                    _plan.Clear();
                    // Cleared with the session: the next hang-out is a different scene,
                    // and a stale script would describe a conversation that already ended.
                    _sceneScript.Clear();
                    _utf16CpyLogged      = 0;
                    _heapSweepCursor     = 0;
                    _utf16SweepHits      = 0;
                    _heapSweepDone       = false;
                    _lastLlmText         = null;
                    lock (_largeCopyLock) _largeCopyDsts.Clear();
                    _sessionActive = false;
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
                _sessionActive = true;

                // Fallback dispatch if hook isn't active.
                if (!_hookActive)
                    _bridge!.DispatchAsync(snap, ContextBuilder.Build(snap),
                                          lineIndex: 0,
                                          maxChars: NextRecordCapacity());

                continue;
            }

            // Keep the queue ahead of the player, and land anything it has finished.
            //
            // Flush first: text generated on an earlier tick should reach memory before a
            // new request is issued, so a busy server delays the next line rather than the
            // one already paid for.
            SocialLinkSnapshot? live = SocialLinkReader.TryReadFromPtr(session);
            if (live is not null)
            {
                FlushReadyRecords();
                PumpPregen(live);
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

            // Primary target: the heap dialogue pool. The mapped-file scans below reach
            // only the global item/skill tables — the live conversation is on the heap,
            // as ASCII, confirmed at 0x41DD7F6389 / 0x42102CAAA9.
            if (_cfg.HeapScanEnabled && _currentMsgId != 0 && _heapPools.Count == 0)
                TryFindHeapDialoguePool();

            // Diagnostic sweeps, off by default now that the pool location is known.
            if (_cfg.StructDiffEnabled)
            {
                if (_confirmedBfBase != 0 && !_bmdScanDoneV2) TryScanBmdVicinity();
                if (_currentMsgId != 0)                       SweepHeapForUtf16();
            }

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
        _msgWatch?.Dispose();
        _msgWatch = null;
        _diffScanner.Reset();
        _logger?.WriteLine("[P5RGenSocialLinks] Unloaded.");
    }

    public bool CanUnload()  => true;
    public bool CanSuspend() => true;
    public Action Disposing  => () => { };
}
