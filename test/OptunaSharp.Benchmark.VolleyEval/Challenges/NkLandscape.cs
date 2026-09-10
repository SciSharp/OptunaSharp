using System;
using System.Collections.Generic;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Samplers;

namespace OptunaSharp.Benchmark.VolleyEval.Challenges;

/// <summary>
///     <b>Challenge 2 — the categorical / epistatic radical.</b> Kauffman's NK landscape
///     (Best-Practice §3.3), chosen over the richer W-Model for v0 because it is ~40 lines and has an
///     <b>exactly computable optimum</b>. Because the space is pure-categorical, the covariance-adapting
///     samplers (CmaEs, LmCma) have no relative move and abstain to independent random sampling — the
///     first-class <b>inert</b> result this challenge exists to surface.
///
///     <para>
///         <b>Objective</b> (maximize F, so minimize −F):
///         <c>F(b) = (1/N)·Σ_{i=1}^{N} f_i(b_i, b_{i+1}, …, b_{i+K})</c>, where each <c>f_i</c> is a
///         fixed seeded random table of <c>2^{K+1}</c> values in <c>[0, 1)</c> and the neighbours are the
///         <b>adjacent, cyclic</b> K loci. <c>N = 24</c> binary loci are exposed as
///         <see cref="Trial.SuggestCategorical(string, IEnumerable{CategoricalChoice})" /> over
///         <c>{0, 1}</c> — categorical on purpose.
///     </para>
///
///     <para>
///         <b>Ruggedness knob.</b> <c>K</c> (v0 uses 2): K = 0 is unimodal; larger K is deceptive/rugged.
///     </para>
///
///     <para>
///         <b>Optimum</b> is computed <b>exactly by dynamic programming</b> over the cyclic adjacent
///         neighbourhood: fix the first K bits (<c>2^K</c> choices) and transfer along the ring keeping
///         the last K placed bits as state, closing the cycle with the wrap contributions — so the
///         normalization ceiling is exact, not a proxy. The <b>random baseline</b> is the clean analytic
///         <c>E[F] = 0.5</c> (uniform tables).
///     </para>
/// </summary>
public sealed class NkLandscape : IChallenge
{
    private static readonly CategoricalChoice[] Bits = { 0L, 1L };

    private readonly double[][] _tables; // _tables[i] has length 2^{K+1}
    private readonly int _n;
    private readonly int _k;
    private readonly double _optF; // exact max F (in [0,1]); Optimum objective = −_optF

    /// <param name="n">Number of binary loci (v0: 24).</param>
    /// <param name="k">Epistatic neighbourhood size / ruggedness (v0: 2).</param>
    /// <param name="instanceId">Seeds the per-locus fitness tables.</param>
    /// <param name="budget">Per-run trial budget.</param>
    public NkLandscape(int n = 24, int k = 2, int instanceId = 0, int budget = 200)
    {
        if (k < 1 || k >= n) throw new ArgumentOutOfRangeException(nameof(k), "Require 1 ≤ K < N.");
        _n = n;
        _k = k;
        InstanceId = instanceId;
        Budget = budget;
        _tables = BuildTables(n, k, instanceId);
        _optF = ComputeOptimumF(_tables, k, n);
    }

    /// <inheritdoc />
    public string Name => "nk_landscape";

    /// <inheritdoc />
    public IReadOnlyList<string> Tags => new[] { "categorical", "rugged", "epistatic", "single-obj" };

    /// <inheritdoc />
    public int Dimension => _n;

    /// <inheritdoc />
    public int NumObjectives => 1;

    /// <inheritdoc />
    public IReadOnlyList<StudyDirection> Directions => new[] { StudyDirection.Minimize };

    /// <inheritdoc />
    public int InstanceId { get; }

    /// <summary>Objective-space optimum = −max(F). (We minimize −F.)</summary>
    public double Optimum => -_optF;

    /// <summary>Objective-space random baseline = −E[F] = −0.5.</summary>
    public double RandomBaseline => -0.5;

    /// <inheritdoc />
    public IReadOnlyList<double> ReferencePoint => Array.Empty<double>();

    /// <inheritdoc />
    public int Budget { get; }

    /// <summary>Pure-categorical space — the inert (≡ random fallback) check is meaningful here.</summary>
    public bool DetectInert => true;

    /// <summary>The exact maximum fitness F* ∈ [0,1] (the DP optimum). Exposed for tests.</summary>
    public double OptimumF => _optF;

    /// <inheritdoc />
    public IReadOnlyList<double> Evaluate(Trial trial)
    {
        var b = new int[_n];
        for (int i = 0; i < _n; i++)
        {
            var choice = trial.SuggestCategorical($"b{i}", Bits);
            b[i] = (int)Convert.ToInt64(choice.Value); // choice.Value is a boxed long (0 or 1)
        }
        return new[] { -Fitness(b) }; // minimize −F
    }

    /// <summary>The raw fitness F(b) ∈ [0,1] for a bit-string. Exposed for tests (DP-vs-brute-force parity).</summary>
    public double Fitness(int[] b)
    {
        double f = 0.0;
        for (int i = 0; i < _n; i++)
            f += _tables[i][NeighbourhoodIndex(b, i)];
        return f / _n;
    }

    /// <inheritdoc />
    public double Score(Study study)
    {
        double bestF = -study.BestValue; // best (minimum) −F ⇒ best F
        return (bestF - 0.5) / (_optF - 0.5);
    }

    // Index into f_i's table: the K+1 bits b_i, b_{i+1}, …, b_{i+K} (cyclic), MSB-first.
    private int NeighbourhoodIndex(int[] b, int i)
    {
        int idx = 0;
        for (int k = 0; k <= _k; k++)
            idx = (idx << 1) | b[(i + k) % _n];
        return idx;
    }

    private static double[][] BuildTables(int n, int k, int instanceId)
    {
        var rng = new LazyRandomState(2000 + instanceId).Rng;
        int size = 1 << (k + 1);
        var tables = new double[n][];
        for (int i = 0; i < n; i++)
        {
            tables[i] = new double[size];
            for (int j = 0; j < size; j++)
                tables[i][j] = (double)rng.uniform(0.0, 1.0);
        }
        return tables;
    }

    /// <summary>
    ///     Exact max of F by dynamic programming over the ring. Fix the first K bits (the
    ///     <c>prefix</c>), then transfer position by position keeping the last K placed bits as the DP
    ///     state; each placed bit at <c>j</c> finalizes contribution <c>f_{j-K}</c>. After all N bits are
    ///     placed, add the wrap contributions <c>f_{N-K..N-1}</c> (whose neighbourhoods reach back into
    ///     the fixed prefix). Complexity O(2^K · N · 2^K); trivially cheap for K ≤ 4.
    /// </summary>
    private static double ComputeOptimumF(double[][] tables, int k, int n)
    {
        double best = double.NegativeInfinity;
        int kBits = 1 << k;

        for (int prefix = 0; prefix < kBits; prefix++)
        {
            // DP state = last K chosen bits (as int in [0, 2^K)); value = accumulated fitness so far.
            var cur = new Dictionary<int, double> { [prefix] = 0.0 };

            for (int j = k; j < n; j++)
            {
                var next = new Dictionary<int, double>();
                foreach (var kv in cur)
                {
                    int state = kv.Key;
                    double val = kv.Value;
                    for (int bit = 0; bit < 2; bit++)
                    {
                        // window for f_{j-K} = (state's K bits) followed by bit.
                        int windowIdx = (state << 1) | bit;
                        double cand = val + tables[j - k][windowIdx];
                        int nstate = windowIdx & (kBits - 1);
                        if (!next.TryGetValue(nstate, out double ex) || cand > ex)
                            next[nstate] = cand;
                    }
                }
                cur = next;
            }

            foreach (var kv in cur)
            {
                int state = kv.Key;   // last K bits (positions N-K..N-1)
                double val = kv.Value;
                double wrap = 0.0;
                for (int t = 0; t < k; t++)
                {
                    int i = n - k + t;         // position whose neighbourhood wraps
                    int windowIdx = 0;
                    for (int kk = 0; kk <= k; kk++)
                    {
                        int pos = (i + kk) % n;
                        int bit = pos >= n - k
                            ? (state >> (k - 1 - (pos - (n - k)))) & 1
                            : (prefix >> (k - 1 - pos)) & 1;
                        windowIdx = (windowIdx << 1) | bit;
                    }
                    wrap += tables[i][windowIdx];
                }
                double total = val + wrap;
                if (total > best) best = total;
            }
        }
        return best / n;
    }
}
