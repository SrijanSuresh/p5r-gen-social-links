using System;
using System.Collections.Generic;

namespace P5RGenSocialLinks.Memory;

/// <summary>
/// Rolling buffer of LLM responses generated in the current hang-out session.
/// Used to build richer per-line context: "we already said X; now generate Y."
/// Also provides deduplication: the same response is never recorded twice.
/// Cleared when the session pointer changes (new hang-out begins).
/// </summary>
internal sealed class SessionHistory
{
    private const int MaxEntries = 8;

    private readonly List<string> _entries    = new(MaxEntries);
    private readonly HashSet<int>  _seenHashes = new();
    private nuint _currentSession;

    /// <returns>False if the response is a duplicate — caller should discard it.</returns>
    internal bool RecordResponse(nuint sessionPtr, string response)
    {
        if (sessionPtr != _currentSession)
        {
            _entries.Clear();
            _seenHashes.Clear();
            _currentSession = sessionPtr;
        }

        int hash = StringComparer.OrdinalIgnoreCase.GetHashCode(response.Trim());
        if (!_seenHashes.Add(hash)) return false;   // duplicate

        if (_entries.Count >= MaxEntries)
            _entries.RemoveAt(0);
        _entries.Add(response);
        return true;
    }

    /// <summary>
    /// Returns a compact "Prior dialogue: [line1] | [line2] | ..." string,
    /// or empty string if no prior lines this session.
    /// </summary>
    internal string BuildPriorContext(nuint sessionPtr)
    {
        if (sessionPtr != _currentSession || _entries.Count == 0)
            return string.Empty;

        return "Prior dialogue: " + string.Join(" | ", _entries);
    }

    internal int Count => _entries.Count;

    internal void Reset()
    {
        _entries.Clear();
        _seenHashes.Clear();
        _currentSession = 0;
    }
}
