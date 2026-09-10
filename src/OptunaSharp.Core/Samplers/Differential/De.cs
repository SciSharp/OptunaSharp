// Differential Evolution (DE) — the classic Storn & Price global optimizer, ask-and-tell core.
//
// Reference:
//   Rainer Storn and Kenneth Price. "Differential Evolution — A Simple and Efficient Heuristic for
//   Global Optimization over Continuous Spaces." Journal of Global Optimization 11:341–359 (1997).
//
// DE has no `cmaes`/Optuna/OptunaHub counterpart, so — like `LmCma` — this is a native, textbook-faithful
// implementation validated by convergence, with no upstream trajectory to bit-match. The RNG is
// NumSharp's MT19937 `RandomState` (the project's "never System.Random" discipline). The optimizer
// works in an arbitrary (unbounded) coordinate space: mutation/crossover never clamp, so a bounded
// caller (both `DifferentialEvolutionSampler` and the JANUS `DePopulationHost`) clamps each candidate
// to its box itself — exactly the concession `LmCma` documents for the per-trial protocol.

using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Samplers.Differential;

/// <summary>The DE mutation strategy (base/1 with binomial crossover).</summary>
internal enum DeStrategy
{
    /// <summary>DE/rand/1/bin — donor = x_r1 + F·(x_r2 − x_r3), three distinct random parents.</summary>
    Rand1Bin,

    /// <summary>DE/best/1/bin — donor = x_best + F·(x_r1 − x_r2), the current best plus a difference.</summary>
    Best1Bin,
}

/// <summary>
///     A Differential Evolution optimizer with a target-indexed ask/tell interface (Storn &amp; Price 1997).
///     Each <see cref="AskCandidate" /> builds the trial vector <c>u_i</c> for one <i>target</i> parent
///     <c>x_i</c> — a mutant (<see cref="DeStrategy" />) binomially crossed with the target — and
///     <see cref="Tell" /> performs DE's greedy one-to-one selection: <c>x_i ← u_i</c> iff
///     <c>f(u_i) ≤ f(x_i)</c>. State (population, values, generation) round-trips through
///     <see cref="ToState" /> / <see cref="FromState" /> so the standalone sampler can persist it in trial
///     system attrs (as <c>LmCma</c> does), while the JANUS host holds one live in memory.
/// </summary>
internal sealed class De
{
    private const double Eps = 1e-12;

    private readonly int _nDim;
    private readonly int _popsize;
    private readonly double _f;             // differential weight F
    private readonly double _cr;            // crossover rate CR
    private readonly DeStrategy _strategy;

    private double[][] _pop;                // NP × d — current population (arbitrary coordinate space)
    private double[] _popVal;              // NP     — objective values (minimization; ±inf allowed)
    private int _t;                        // generation counter

    private NumPyRandom _rng;

    /// <param name="initialPop">The initial population (NP vectors of length d). Cloned defensively.</param>
    /// <param name="initialVal">Objective values of the initial population (minimization); a non-finite
    ///     entry means "unknown" — the first offspring to beat it is always accepted.</param>
    /// <param name="f">Differential weight F (default 0.8).</param>
    /// <param name="cr">Crossover rate CR (default 0.9).</param>
    /// <param name="strategy">Mutation strategy (default DE/rand/1/bin).</param>
    /// <param name="seed">Seed for the MT19937 stream (null ⇒ nondeterministic).</param>
    internal De(double[][] initialPop, double[] initialVal,
        double f = 0.8, double cr = 0.9, DeStrategy strategy = DeStrategy.Rand1Bin, int? seed = null)
    {
        if (initialPop is null || initialPop.Length == 0) throw new ArgumentException("initialPop must be non-empty");
        int popsize = initialPop.Length;
        if (popsize < 4) throw new ArgumentException("DE requires a population of at least 4 (target + 3 donors).");
        if (initialVal is null || initialVal.Length != popsize)
            throw new ArgumentException("initialVal must have one value per population member.");
        int nDim = initialPop[0].Length;
        if (nDim <= 0) throw new ArgumentException("The dimension must be positive.");
        foreach (double[] row in initialPop)
            if (row.Length != nDim) throw new ArgumentException("All population members must share the dimension.");

        _nDim = nDim;
        _popsize = popsize;
        _f = f;
        _cr = cr;
        _strategy = strategy;

        _pop = new double[popsize][];
        for (int i = 0; i < popsize; i++) _pop[i] = (double[])initialPop[i].Clone();
        _popVal = (double[])initialVal.Clone();
        _t = 0;

        _rng = np.random.RandomState();
        if (seed is not null) _rng.seed((uint)seed.Value);
    }

    internal int Dim => _nDim;
    internal int PopulationSize => _popsize;
    internal int Generation => _t;
    internal double F => _f;
    internal double CR => _cr;
    internal DeStrategy Strategy => _strategy;

    /// <summary>Re-seed the MT19937 stream — the per-trial sampler reseeds before each single ask so its
    ///     one candidate is reproducible from the restored population (as <c>LmCma.ReseedRng</c>).</summary>
    internal void ReseedRng(int seed) => _rng.seed((uint)seed);

    /// <summary>The population centroid — the JANUS host's injection center (<see cref="IPopulationHost.Mean" />).</summary>
    internal double[] Mean()
    {
        var m = new double[_nDim];
        for (int i = 0; i < _popsize; i++)
            for (int j = 0; j < _nDim; j++) m[j] += _pop[i][j];
        for (int j = 0; j < _nDim; j++) m[j] /= _popsize;
        return m;
    }

    /// <summary>Per-dimension population standard deviation (population/ddof-0, floored at <see cref="Eps" />)
    ///     — DE's covariance-free analogue of a proposal scale, used by the JANUS host's step-length
    ///     whitening and its reported σ.</summary>
    internal double[] PerDimStd()
    {
        double[] mean = Mean();
        var sd = new double[_nDim];
        for (int j = 0; j < _nDim; j++)
        {
            double s = 0;
            for (int i = 0; i < _popsize; i++) { double diff = _pop[i][j] - mean[j]; s += diff * diff; }
            sd[j] = Math.Max(Math.Sqrt(s / _popsize), Eps);
        }
        return sd;
    }

    /// <summary>Index of the current best (lowest-value) member; non-finite values sort last.</summary>
    internal int BestIndex()
    {
        int best = 0;
        double bestVal = double.PositiveInfinity;
        for (int i = 0; i < _popsize; i++)
        {
            double v = IsFinite(_popVal[i]) ? _popVal[i] : double.PositiveInfinity;
            if (v < bestVal) { bestVal = v; best = i; }
        }
        return best;
    }

    /// <summary>
    ///     Build the trial vector <c>u</c> for target parent <paramref name="target" /> (Storn &amp; Price
    ///     1997, eqs. 1–3): a <see cref="DeStrategy" /> mutant binomially crossed with the target, with the
    ///     <c>j_rand</c> forced coordinate guaranteeing <c>u ≠ x_target</c>. The result is returned in the
    ///     optimizer's coordinate space — the caller clamps it to any box.
    /// </summary>
    internal double[] AskCandidate(int target)
    {
        if (target < 0 || target >= _popsize) throw new ArgumentOutOfRangeException(nameof(target));

        var used = new HashSet<int> { target };
        double[] mutant = new double[_nDim];
        if (_strategy == DeStrategy.Best1Bin)
        {
            int best = BestIndex();
            used.Add(best);
            int r1 = SampleDistinct(used), r2 = SampleDistinct(used);
            for (int i = 0; i < _nDim; i++) mutant[i] = _pop[best][i] + _f * (_pop[r1][i] - _pop[r2][i]);
        }
        else   // DE/rand/1
        {
            int r1 = SampleDistinct(used), r2 = SampleDistinct(used), r3 = SampleDistinct(used);
            for (int i = 0; i < _nDim; i++) mutant[i] = _pop[r1][i] + _f * (_pop[r2][i] - _pop[r3][i]);
        }

        // Binomial crossover: inherit the mutant coordinate where U(0,1) < CR, else the target's; the
        // single j_rand coordinate is always taken from the mutant (guarantees a genuine offspring).
        int jrand = (int)_rng.randint(0, _nDim);
        double[] cross = _rng.uniform(0.0, 1.0, new Shape(_nDim)).ToArray<double>();
        double[] tgt = _pop[target];
        var u = new double[_nDim];
        for (int i = 0; i < _nDim; i++) u[i] = (cross[i] < _cr || i == jrand) ? mutant[i] : tgt[i];
        return u;
    }

    /// <summary>
    ///     Greedy one-to-one selection (Storn &amp; Price 1997, eq. 4): <paramref name="solutions" /> holds
    ///     exactly one evaluated offspring per target, in target order (<c>solutions[i]</c> ↔ parent
    ///     <c>i</c>). Each parent is replaced iff its offspring is finite and no worse (or the parent's own
    ///     value was non-finite, i.e. unknown). Advances the generation counter.
    /// </summary>
    internal void Tell(IReadOnlyList<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException($"Must tell exactly {_popsize} target-ordered solutions.");
        for (int i = 0; i < _popsize; i++)
        {
            double v = solutions[i].Value;
            if (IsFinite(v) && (v <= _popVal[i] || !IsFinite(_popVal[i])))
            {
                _pop[i] = (double[])solutions[i].X.Clone();
                _popVal[i] = v;
            }
        }
        _t += 1;
    }

    /// <summary>
    ///     Order-independent (μ+λ) elitist selection: pool the current population with the evaluated
    ///     <paramref name="offspring" /> and keep the best NP by value. Unlike <see cref="Tell" />'s
    ///     one-to-one greedy pairing, the result does not depend on the offspring order — the selection the
    ///     JANUS host uses, where the driver delivers a coherent generation whose per-slot target order is
    ///     not preserved (the injected Gauss–Newton elite simply competes for a place in the pool). Advances
    ///     the generation counter.
    /// </summary>
    internal void TellElitist(IReadOnlyList<(double[] X, double Value)> offspring)
    {
        var combined = new List<(double[] X, double Val)>(_popsize + offspring.Count);
        for (int i = 0; i < _popsize; i++) combined.Add((_pop[i], _popVal[i]));
        foreach (var (x, v) in offspring)
            if (IsFinite(v)) combined.Add(((double[])x.Clone(), v));
        combined.Sort((a, b) => SortKey(a.Val).CompareTo(SortKey(b.Val)));
        for (int i = 0; i < _popsize; i++)
        {
            _pop[i] = (double[])combined[i].X.Clone();
            _popVal[i] = combined[i].Val;
        }
        _t += 1;
    }

    private static double SortKey(double v) => IsFinite(v) ? v : double.PositiveInfinity;

    /// <summary>A distinct index in [0, NP) not already in <paramref name="exclude" /> (which it mutates).
    ///     NP ≥ 4 guarantees termination for both strategies.</summary>
    private int SampleDistinct(HashSet<int> exclude)
    {
        while (true)
        {
            int r = (int)_rng.randint(0, _popsize);
            if (exclude.Add(r)) return r;
        }
    }

    private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    /// <summary>The population-size default shared by the sampler and the host: <c>max(8, 4 + ⌊3 ln d⌋)</c>
    ///     — the CMA-ES λ formula floored at 8 for donor diversity, kept modest so generations stay short
    ///     enough to iterate at Optuna trial budgets (a DE textbook <c>NP ≈ 10d</c> rarely completes a
    ///     generation in a few-hundred-trial study).</summary>
    internal static int DefaultPopulationSize(int d) => Math.Max(8, 4 + (int)Math.Floor(3.0 * Math.Log(d)));

    // ---- state serialization (own JSON; mirrors the LmCma/SepCma pattern) -------------------------

    internal DeState ToState() => new()
    {
        NDim = _nDim,
        Popsize = _popsize,
        F = _f,
        Cr = _cr,
        Strategy = (int)_strategy,
        Pop = CloneJagged(_pop),
        PopVal = (double[])_popVal.Clone(),
        T = _t,
    };

    internal static De FromState(DeState s)
    {
        var de = new De(s.Pop, s.PopVal, s.F, s.Cr, (DeStrategy)s.Strategy, seed: null);
        de._t = s.T;
        return de;
    }

    private static double[][] CloneJagged(double[][] a)
    {
        var j = new double[a.Length][];
        for (int i = 0; i < a.Length; i++) j[i] = (double[])a[i].Clone();
        return j;
    }
}

/// <summary>Serialized evolving state of a <see cref="De" /> optimizer (own JSON; see <c>LmCmaState</c>).</summary>
internal sealed class DeState
{
    public int NDim { get; set; }
    public int Popsize { get; set; }
    public double F { get; set; }
    public double Cr { get; set; }
    public int Strategy { get; set; }
    public double[][] Pop { get; set; } = Array.Empty<double[]>();
    public double[] PopVal { get; set; } = Array.Empty<double>();
    public int T { get; set; }
}
