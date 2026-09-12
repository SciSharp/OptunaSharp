namespace OptunaSharp.PresentationHost;

/// <summary>Small, bounded local-demo store. Every individual study is serialized.</summary>
public sealed class RunStore
{
    public const int MaximumSessions = 8;
    public static readonly TimeSpan IdleTtl = TimeSpan.FromMinutes(20);
    private readonly object _gate = new();
    private readonly Dictionary<string, Entry> _runs = [];

    public RunSnapshot Create(RunConfig config)
    {
        config.Validate();
        lock (_gate)
        {
            RemoveExpired();
            if (_runs.Count >= MaximumSessions)
                throw new SessionLimitException("Eight runs are already active. Reset/delete a run or wait for its 20-minute idle expiry.");
            var run = new Nsga3Run(config);
            _runs.Add(run.Snapshot.RunId, new Entry(run));
            return run.Snapshot;
        }
    }

    public RunSnapshot? Read(string id) => WithRun(id, r => r.Snapshot);
    public RunSnapshot? Step(string id) => WithRun(id, r => r.Step());

    private RunSnapshot? WithRun(string id, Func<Nsga3Run, RunSnapshot> action)
    {
        Entry entry;
        lock (_gate)
        {
            RemoveExpired();
            if (!_runs.TryGetValue(id, out entry!)) return null;
            // Take the per-run gate while holding store gate so deletion cannot race this lookup.
            Monitor.Enter(entry.Gate);
        }
        try
        {
            entry.LastAccess = DateTimeOffset.UtcNow;
            return action(entry.Run);
        }
        finally { Monitor.Exit(entry.Gate); }
    }

    public bool Delete(string id)
    {
        lock (_gate)
        {
            if (!_runs.TryGetValue(id, out var entry)) return false;
            lock (entry.Gate) return _runs.Remove(id);
        }
    }

    private void RemoveExpired()
    {
        var now = DateTimeOffset.UtcNow;
        foreach (var (id, entry) in _runs.ToArray())
        {
            if (!Monitor.TryEnter(entry.Gate)) continue;
            try
            {
                if (now - entry.LastAccess > IdleTtl) _runs.Remove(id);
            }
            finally { Monitor.Exit(entry.Gate); }
        }
    }

    private sealed class Entry(Nsga3Run run)
    {
        public object Gate { get; } = new();
        public Nsga3Run Run { get; } = run;
        public DateTimeOffset LastAccess { get; set; } = DateTimeOffset.UtcNow;
    }
}

public sealed class SessionLimitException(string message) : Exception(message);
