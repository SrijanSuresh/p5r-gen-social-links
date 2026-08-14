using System;
using System.IO;
using Reloaded.Mod.Interfaces;
using Reloaded.Mod.Interfaces.Internal;

namespace P5RGenSocialLinks;

/// <summary>
/// Thin wrapper around ILoggerV2 that filters messages below the configured level, and
/// mirrors everything to a file.
///
/// Levels in ascending severity: info &lt; warn &lt; off.
///
/// The file sink exists because the console is not readable after the fact. It holds a
/// few hundred lines, it cannot be scrolled while the game has focus, and a crash takes
/// it with the process — so the run that most needs inspecting is the one whose output
/// is gone. Every session so far has been diagnosed from screenshots of a scrolled
/// window, which reliably shows the part that happened to be on screen rather than the
/// part that mattered.
/// </summary>
internal sealed class ModLogger
{
    private readonly ILoggerV2 _inner;
    private readonly int       _level;  // 0=info, 1=warn, 2=off
    private readonly string?   _path;
    private readonly object    _fileLock = new();

    // Held open rather than reopened per line. Opening, appending and closing a file for
    // every message is fine at a 500ms tick and is not fine when the player opens the
    // backlog: that re-reads dozens of records at once, and the resulting burst of log
    // lines turned into a burst of file-open syscalls on a thread the game is waiting on.
    // AutoFlush keeps the crash-safety that per-line opening was there to provide.
    private readonly StreamWriter? _writer;

    /// <summary>
    /// Where the mirror is written. Temp rather than the mod directory: the mod folder
    /// lives under Reloaded's install and is not guaranteed writable by the game process,
    /// and a logger that throws on its first line is worse than no logger.
    /// </summary>
    internal static string LogPath =>
        Path.Combine(Path.GetTempPath(), "p5r-gen-social-links.log");

    internal ModLogger(ILoggerV2 inner, string levelName)
    {
        _inner = inner;
        _level = levelName.ToLowerInvariant() switch
        {
            "warn" => 1,
            "off"  => 2,
            _      => 0,  // "info" or unknown → show everything
        };

        // Truncate per launch. Appending across runs turns "what did this session do"
        // into a search problem, and these logs are read to answer questions about one
        // specific run.
        try
        {
            File.WriteAllText(LogPath, $"=== P5RGenSocialLinks {DateTime.Now:yyyy-MM-dd HH:mm:ss} ===\n");
            _path   = LogPath;
            _writer = new StreamWriter(LogPath, append: true) { AutoFlush = true };
        }
        catch (IOException)
        {
            _path = null;
        }
        catch (UnauthorizedAccessException)
        {
            _path = null;
        }
    }

    internal void Info(string msg)
    {
        if (_level <= 0) { _inner.WriteLine(msg); Mirror(msg); }
    }

    internal void Warn(string msg)
    {
        if (_level <= 1) { _inner.WriteLine(msg); Mirror(msg); }
    }

    // Direct pass-through for critical startup messages that always appear.
    internal void Always(string msg)
    {
        _inner.WriteLine(msg);
        Mirror(msg);
    }

    /// <summary>
    /// Append one line through a writer held open for the process lifetime.
    ///
    /// AutoFlush is on, so the tail still survives a hard crash — which is the whole
    /// reason the earlier version reopened the file per line. Reopening turned out to
    /// cost far more than it bought: a backlog burst produced dozens of log lines in a
    /// few milliseconds, and dozens of open/close syscalls with it.
    /// </summary>
    private void Mirror(string msg)
    {
        if (_writer is null) return;
        try
        {
            lock (_fileLock)
                _writer.WriteLine($"{DateTime.Now:HH:mm:ss.fff}  {msg}");
        }
        catch (IOException)
        {
            // A locked or full disk must never take the game down over a log line.
        }
        catch (UnauthorizedAccessException)
        {
        }
    }
}
