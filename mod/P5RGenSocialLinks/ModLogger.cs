using System;
using Reloaded.Mod.Interfaces;
using Reloaded.Mod.Interfaces.Internal;

namespace P5RGenSocialLinks;

/// <summary>
/// Thin wrapper around ILoggerV2 that filters messages below the configured level.
/// Levels in ascending severity: info < warn < off.
/// </summary>
internal sealed class ModLogger
{
    private readonly ILoggerV2 _inner;
    private readonly int       _level;  // 0=info, 1=warn, 2=off

    internal ModLogger(ILoggerV2 inner, string levelName)
    {
        _inner = inner;
        _level = levelName.ToLowerInvariant() switch
        {
            "warn" => 1,
            "off"  => 2,
            _      => 0,  // "info" or unknown → show everything
        };
    }

    internal void Info(string msg)
    {
        if (_level <= 0) _inner.WriteLine(msg);
    }

    internal void Warn(string msg)
    {
        if (_level <= 1) _inner.WriteLine(msg);
    }

    // Direct pass-through for critical startup messages that always appear.
    internal void Always(string msg) => _inner.WriteLine(msg);
}
