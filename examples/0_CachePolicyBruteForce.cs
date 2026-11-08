#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// Cache-policy test matrix — BruteForceSampler runs EVERY configuration of a conditional cache design space
// exactly once (no cache → nothing else; LRU/FIFO → a capacity; LFU → a capacity and whether counts age;
// TTL expiry only where it makes sense) against a replayed request trace, and stops by itself when the tree
// is exhausted: the "benchmark the whole matrix" job, without hand-writing nested loops.
//
// source: https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.BruteForceSampler.html
//         (exhaustive search over a define-by-run — possibly conditional — space; the study stops when done)
//         https://en.wikipedia.org/wiki/Cache_replacement_policies
//
// Real objective, cache simulation: a deterministic 200,000-request trace over 50,000 keys with a Zipf(0.9)
// popularity that drifts (a new hot set every 50k requests, as when a product goes viral) and per-key object
// sizes; each configuration replays it and scores the byte hit ratio minus a memory charge for the cache's
// size (0.4 % per thousand entries). The space is conditional, so the number of configurations is not a
// product of the parameter counts — BruteForceSampler discovers the tree as it runs.
//
//   Run:  dotnet run examples/0_CachePolicyBruteForce.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Warning);   // silence the per-trial INFO lines

var trace = Trace.Generate(requests: 200_000, keys: 50_000, seed: 23);
var study = Optuna.CreateStudy(sampler: new BruteForceSampler(seed: 0), direction: StudyDirection.Maximize, studyName: "cache-matrix");
study.Optimize(t => Objective(t, trace), nTrials: 10_000);         // an upper bound; the sampler ends it

Console.WriteLine($"configurations evaluated: {study.Trials.Count} (the whole conditional tree, each once)");
var distinct = study.Trials.Select(t => string.Join(",", t.Params.OrderBy(kv => kv.Key).Select(kv => $"{kv.Key}={kv.Value}"))).Distinct().Count();
Console.WriteLine($"distinct configurations: {distinct}");
Console.WriteLine();
Console.WriteLine("  score   byte-hit   configuration");
foreach (var t in study.Trials.OrderByDescending(t => t.Value).Take(8))
    Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "{0,7:F4} {1,9:P1}   {2}", t.Value, t.UserAttrs["byte_hit_ratio"],
        string.Join(", ", t.Params.Select(kv => $"{kv.Key}={kv.Value}"))));

/// <summary>Replays the trace through one cache configuration.</summary>
/// <param name="t">The trial; the parameters it asks for depend on the earlier answers.</param>
/// <param name="trace">The request trace.</param>
/// <returns>Byte hit ratio minus the memory charge.</returns>
static double Objective(Trial t, Trace trace)
{
    string policy = t.SuggestCategorical("policy", new[] { "none", "lru", "fifo", "lfu" });
    if (policy == "none")
    {
        t.SetUserAttr("byte_hit_ratio", 0.0);
        return 0.0;
    }
    int capacity = (int)t.SuggestCategorical("capacity", new[] { 1_000, 4_000, 16_000 });
    bool aging = policy == "lfu" && t.SuggestCategorical("aging", new[] { false, true });   // only LFU has counts
    long ttl = policy == "fifo" ? 0 : t.SuggestCategorical("ttl_requests", new[] { 0, 20_000 });   // 0 = never expire
    double hit = Cache.Replay(trace, policy, capacity, aging, ttl);
    t.SetUserAttr("byte_hit_ratio", hit);
    return hit - 0.004 * capacity / 1000.0;
}

/// <summary>A request trace: which key each request asks for, and every key's object size.</summary>
/// <param name="Keys">The key of each request, in order.</param>
/// <param name="Sizes">The object size of each key, in KiB.</param>
sealed record Trace(int[] Keys, int[] Sizes)
{
    /// <summary>Generates a drifting Zipf trace; the same seed gives the same trace.</summary>
    /// <param name="requests">Number of requests.</param>
    /// <param name="keys">Size of the key space.</param>
    /// <param name="seed">Trace seed.</param>
    /// <returns>The trace.</returns>
    public static Trace Generate(int requests, int keys, int seed)
    {
        var rng = new Random(seed);   // trace generation only — samplers never use System.Random
        var cdf = new double[keys];
        double sum = 0;
        for (int k = 0; k < keys; k++)
            cdf[k] = sum += 1 / Math.Pow(k + 1, 0.9);
        var perm = Enumerable.Range(0, keys).ToArray();
        var reqs = new int[requests];
        for (int i = 0; i < requests; i++)
        {
            if (i % 50_000 == 0)
                rng.Shuffle(perm);                                   // the popular set changes
            int rank = Array.BinarySearch(cdf, rng.NextDouble() * sum);
            reqs[i] = perm[rank < 0 ? ~rank : rank];
        }
        var sizes = Enumerable.Range(0, keys).Select(_ => 1 + (int)Math.Exp(3 * rng.NextDouble())).ToArray();
        return new Trace(reqs, sizes);
    }
}

/// <summary>Byte-hit-ratio simulation of LRU, FIFO and LFU caches with optional expiry.</summary>
static class Cache
{
    /// <summary>Replays the trace and returns the fraction of requested bytes served from the cache.</summary>
    /// <param name="trace">The trace.</param>
    /// <param name="policy">"lru", "fifo" or "lfu".</param>
    /// <param name="capacity">Maximum number of cached entries.</param>
    /// <param name="aging">LFU only: halve every count every 10,000 requests so stale favourites fade.</param>
    /// <param name="ttl">Entries older than this many requests are expired on access (0 = no expiry).</param>
    /// <returns>The byte hit ratio in [0, 1].</returns>
    public static double Replay(Trace trace, string policy, int capacity, bool aging, long ttl)
    {
        var inserted = new Dictionary<int, long>();
        var lastUse = new Dictionary<int, long>();
        var counts = new Dictionary<int, double>();
        var order = new SortedSet<(double Priority, long Tiebreak, int Key)>();
        var priority = new Dictionary<int, (double, long, int)>();
        long hitBytes = 0, totalBytes = 0;
        for (long i = 0; i < trace.Keys.Length; i++)
        {
            int key = trace.Keys[i];
            int size = trace.Sizes[key];
            totalBytes += size;
            if (aging && i % 10_000 == 0 && i > 0)
                foreach (var k in counts.Keys.ToList())
                    Reprioritize(k, counts[k] / 2);
            if (inserted.TryGetValue(key, out long born) && ttl > 0 && i - born > ttl)
                Evict(key);                                          // expired: treat as a miss
            if (inserted.ContainsKey(key))
            {
                hitBytes += size;
                if (policy == "lru") Reprioritize(key, i);
                if (policy == "lfu") Reprioritize(key, counts[key] + 1);
                continue;
            }
            if (inserted.Count >= capacity)
                Evict(order.Min.Key);                                // the lowest priority goes
            inserted[key] = i;
            counts[key] = 1;
            Reprioritize(key, policy == "lfu" ? 1 : i);
        }
        return (double)hitBytes / totalBytes;

        // a key's priority: last use (LRU), insertion time (FIFO) or its (aged) hit count (LFU).
        void Reprioritize(int key, double value)
        {
            if (priority.TryGetValue(key, out var old))
                order.Remove(old);
            if (policy == "lfu")
                counts[key] = value;
            var entry = (value, lastUse[key] = lastUse.GetValueOrDefault(key) + 1, key);
            order.Add(entry);
            priority[key] = entry;
        }

        void Evict(int key)
        {
            order.Remove(priority[key]);
            priority.Remove(key);
            inserted.Remove(key);
            counts.Remove(key);
        }
    }
}
