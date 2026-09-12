// Particle Swarm Optimization (PSO) — Kennedy & Eberhart's canonical global-best swarm optimizer,
// ask-and-tell core.
//
// References:
//   James Kennedy and Russell Eberhart. "Particle Swarm Optimization." Proc. IEEE Int. Conf. on
//   Neural Networks, 1995.
//   Yuhui Shi and Russell Eberhart. "A Modified Particle Swarm Optimizer." Proc. IEEE Congress on
//   Evolutionary Computation, 1998 — the linearly-decreasing inertia weight (LDIW) used here.
//
// PSO has no `cmaes`/Optuna/OptunaHub counterpart, so — like `De`/`LmCma` — this is a native,
// textbook-faithful implementation validated by convergence, with no upstream trajectory to bit-match.
// The RNG is NumSharp's MT19937 `RandomState` (the project's "never System.Random" discipline). The
// optimizer works in the transformed unit box [0,1]^d: the velocity/position updates never clamp, so a
// bounded caller (`PsoSampler`) clamps each asked candidate to the box itself, and the next `Tell`
// recovers each particle's realized velocity from the actually-evaluated positions (a velocity reset on
// bound contact — a standard PSO bound-handling strategy) — exactly the per-trial concession `De`
// documents.

using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Samplers.ParticleSwarm;

/// <summary>
///     A global-best (gbest) Particle Swarm optimizer with a target-indexed ask/tell interface
///     (Kennedy &amp; Eberhart 1995; Shi &amp; Eberhart 1998 inertia weight). Each particle tracks its
///     personal-best position and the swarm shares one global-best; <see cref="AskCandidate" /> returns
///     the next position of one particle from the velocity-update equation
///     <c>v ← w·v + c1·r1·(pbest − x) + c2·r2·(gbest − x)</c>, <c>x ← x + v</c>, and <see cref="Tell" />
///     folds a full swarm of evaluated positions back in: it recovers each particle's realized velocity
///     from the evaluated (box-clamped) position, updates the personal- and global-bests, and advances
///     the generation. State round-trips through <see cref="ToState" /> / <see cref="FromState" /> so the
///     sampler can persist it in trial system attrs (as <c>De</c> / <c>LmCma</c> do).
/// </summary>
internal sealed class Pso
{
    private const double Eps = 1e-12;

    private readonly int _nDim;
    private readonly int _popsize;
    private readonly double _w0;                 // initial inertia weight (generation 0)
    private readonly double _w1;                 // final inertia weight (after the decay horizon)
    private readonly int _inertiaDecayGenerations; // generations over which w decays linearly w0 → w1
    private readonly double _c1;                 // cognitive (personal-best) acceleration coefficient
    private readonly double _c2;                 // social (global-best) acceleration coefficient
    private readonly double _vMax;               // per-coordinate velocity clamp magnitude (unit-box scale)

    private double[][] _pos;                     // NP × d — current particle positions (unit box space)
    private double[][] _vel;                     // NP × d — current particle velocities
    private double[][] _pbestPos;                // NP × d — personal-best positions
    private double[] _pbestVal;                  // NP     — personal-best objective values (minimization)
    private double[] _gbestPos;                  // d      — global-best position
    private double _gbestVal;                    //          global-best objective value
    private int _t;                              // generation counter

    private NumPyRandom _rng;

    /// <param name="initialPop">The initial swarm positions (NP vectors of length d, in [0,1]^d). Cloned defensively.</param>
    /// <param name="initialVal">Objective values of the initial swarm (minimization); a non-finite entry means "unknown".</param>
    /// <param name="w0">Initial inertia weight (Shi &amp; Eberhart's <c>w_max</c>, default 0.9).</param>
    /// <param name="w1">Final inertia weight (Shi &amp; Eberhart's <c>w_min</c>, default 0.4).</param>
    /// <param name="inertiaDecayGenerations">Generations over which the inertia weight decays linearly from
    ///     <paramref name="w0" /> to <paramref name="w1" /> (then stays at <paramref name="w1" />). Optuna is
    ///     trial-budget-driven and does not expose a maximum generation to the sampler, so this sets the LDIW
    ///     schedule's horizon explicitly.</param>
    /// <param name="c1">Cognitive coefficient (default 1.49445, the Clerc-derived LDIW pairing).</param>
    /// <param name="c2">Social coefficient (default 1.49445).</param>
    /// <param name="vMax">Per-coordinate velocity clamp magnitude in the unit box (default 0.2).</param>
    /// <param name="seed">Seed for the MT19937 stream (null ⇒ nondeterministic).</param>
    internal Pso(double[][] initialPop, double[] initialVal,
        double w0 = 0.9, double w1 = 0.4, int inertiaDecayGenerations = 100,
        double c1 = 1.49445, double c2 = 1.49445, double vMax = 0.2, int? seed = null)
    {
        if (initialPop is null || initialPop.Length == 0) throw new ArgumentException("initialPop must be non-empty");
        int popsize = initialPop.Length;
        if (popsize < 2) throw new ArgumentException("PSO requires a swarm of at least 2 particles.");
        if (initialVal is null || initialVal.Length != popsize)
            throw new ArgumentException("initialVal must have one value per particle.");
        int nDim = initialPop[0].Length;
        if (nDim <= 0) throw new ArgumentException("The dimension must be positive.");
        foreach (double[] row in initialPop)
            if (row.Length != nDim) throw new ArgumentException("All particles must share the dimension.");

        _nDim = nDim;
        _popsize = popsize;
        _w0 = w0;
        _w1 = w1;
        _inertiaDecayGenerations = inertiaDecayGenerations;
        _c1 = c1;
        _c2 = c2;
        _vMax = vMax;

        _pos = CloneJagged(initialPop);
        _vel = new double[popsize][];
        for (int i = 0; i < popsize; i++) _vel[i] = new double[nDim];   // start at rest (zero velocity)
        _pbestPos = CloneJagged(initialPop);
        _pbestVal = (double[])initialVal.Clone();

        // The global best is the best finite personal best; if none is finite yet, seed it as +inf.
        _gbestPos = new double[nDim];
        _gbestVal = double.PositiveInfinity;
        UpdateGlobalBest();
        _t = 0;

        _rng = np.random.RandomState();
        if (seed is not null) _rng.seed((uint)seed.Value);
    }

    internal int Dim => _nDim;
    internal int PopulationSize => _popsize;
    internal int Generation => _t;

    /// <summary>Re-seed the MT19937 stream — the per-trial sampler reseeds before each single ask so its
    ///     one candidate is reproducible from the restored swarm (as <c>De.ReseedRng</c>).</summary>
    internal void ReseedRng(int seed) => _rng.seed((uint)seed);

    /// <summary>The inertia weight at generation <paramref name="t" /> — a linear decrease from
    ///     <see cref="_w0" /> to <see cref="_w1" /> over <see cref="_inertiaDecayGenerations" /> generations
    ///     (Shi &amp; Eberhart 1998), constant at <see cref="_w1" /> thereafter.</summary>
    internal double InertiaWeight(int t)
    {
        if (_inertiaDecayGenerations <= 0) return _w1;
        double frac = Math.Min(1.0, t / (double)_inertiaDecayGenerations);
        return _w0 + (_w1 - _w0) * frac;
    }

    /// <summary>
    ///     Build the next position of particle <paramref name="target" /> from the PSO velocity-update
    ///     equation, drawing the per-coordinate cognitive/social randoms <c>r1, r2</c>. The velocity is
    ///     clamped to <c>±vMax</c> per coordinate; the returned position is in the optimizer's unit-box
    ///     space (the caller clamps it to the box). This is a pure read of the swarm state plus the RNG —
    ///     the persistent velocity/position are advanced by <see cref="Tell" />, so a freshly-restored
    ///     optimizer asks the same candidate after an identical reseed.
    /// </summary>
    internal double[] AskCandidate(int target)
    {
        if (target < 0 || target >= _popsize) throw new ArgumentOutOfRangeException(nameof(target));

        double w = InertiaWeight(_t);
        double[] r1 = _rng.uniform(0.0, 1.0, new Shape(_nDim)).ToArray<double>();
        double[] r2 = _rng.uniform(0.0, 1.0, new Shape(_nDim)).ToArray<double>();

        double[] x = _pos[target];
        double[] v = _vel[target];
        double[] pbest = _pbestPos[target];
        var candidate = new double[_nDim];
        for (int i = 0; i < _nDim; i++)
        {
            double vp = w * v[i] + _c1 * r1[i] * (pbest[i] - x[i]) + _c2 * r2[i] * (_gbestPos[i] - x[i]);
            if (vp > _vMax) vp = _vMax;
            else if (vp < -_vMax) vp = -_vMax;
            candidate[i] = x[i] + vp;
        }
        return candidate;
    }

    /// <summary>
    ///     Fold a full swarm of evaluated positions back in: <paramref name="solutions" /> holds exactly one
    ///     evaluated particle per target, in target order (<c>solutions[i]</c> ↔ particle <c>i</c>). Each
    ///     particle's realized velocity is recovered as <c>x_evaluated − x_previous</c> (which accounts for
    ///     the caller's box clamp), its position is set to the evaluated one, and its personal best is
    ///     updated when the value is finite and no worse (or the personal best was unknown). The global best
    ///     is then refreshed and the generation advanced.
    /// </summary>
    internal void Tell(IReadOnlyList<(double[] X, double Value)> solutions)
    {
        if (solutions.Count != _popsize)
            throw new ArgumentException($"Must tell exactly {_popsize} target-ordered solutions.");
        for (int i = 0; i < _popsize; i++)
        {
            double[] newPos = solutions[i].X;
            var realizedV = new double[_nDim];
            for (int j = 0; j < _nDim; j++) realizedV[j] = newPos[j] - _pos[i][j];
            _vel[i] = realizedV;
            _pos[i] = (double[])newPos.Clone();

            double val = solutions[i].Value;
            if (IsFinite(val) && (val <= _pbestVal[i] || !IsFinite(_pbestVal[i])))
            {
                _pbestPos[i] = (double[])newPos.Clone();
                _pbestVal[i] = val;
            }
        }
        UpdateGlobalBest();
        _t += 1;
    }

    /// <summary>Refresh the global best from the current personal bests (lowest finite value wins).</summary>
    private void UpdateGlobalBest()
    {
        for (int i = 0; i < _popsize; i++)
        {
            if (IsFinite(_pbestVal[i]) && _pbestVal[i] < _gbestVal)
            {
                _gbestVal = _pbestVal[i];
                _gbestPos = (double[])_pbestPos[i].Clone();
            }
        }
    }

    private static bool IsFinite(double v) => !double.IsNaN(v) && !double.IsInfinity(v);

    /// <summary>The default swarm size shared by the sampler: <c>max(10, 4 + ⌊3 ln d⌋)</c> — the CMA-ES λ
    ///     formula floored at 10 for a workable swarm, kept modest so generations stay short enough to
    ///     iterate at Optuna trial budgets (a textbook PSO swarm of 20–40 rarely completes many generations
    ///     in a few-hundred-trial study).</summary>
    internal static int DefaultPopulationSize(int d) => Math.Max(10, 4 + (int)Math.Floor(3.0 * Math.Log(d)));

    private static double[][] CloneJagged(double[][] a)
    {
        var j = new double[a.Length][];
        for (int i = 0; i < a.Length; i++) j[i] = (double[])a[i].Clone();
        return j;
    }

    // ---- state serialization (own JSON; mirrors the De/LmCma pattern) -----------------------------

    internal PsoState ToState() => new()
    {
        NDim = _nDim,
        Popsize = _popsize,
        W0 = _w0,
        W1 = _w1,
        InertiaDecayGenerations = _inertiaDecayGenerations,
        C1 = _c1,
        C2 = _c2,
        VMax = _vMax,
        Pos = CloneJagged(_pos),
        Vel = CloneJagged(_vel),
        PbestPos = CloneJagged(_pbestPos),
        PbestVal = (double[])_pbestVal.Clone(),
        GbestPos = (double[])_gbestPos.Clone(),
        GbestVal = _gbestVal,
        T = _t,
    };

    internal static Pso FromState(PsoState s)
    {
        // Reconstruct through the ctor (validates shapes and seeds pos/pbest), then restore the evolving
        // fields verbatim — the ctor cannot know the velocities, the diverged personal-best positions, or
        // the generation counter, so they are assigned here after construction (as De.FromState does).
        var pso = new Pso(s.Pos, s.PbestVal, s.W0, s.W1, s.InertiaDecayGenerations, s.C1, s.C2, s.VMax, seed: null);
        pso._pos = CloneJagged(s.Pos);
        pso._vel = CloneJagged(s.Vel);
        pso._pbestPos = CloneJagged(s.PbestPos);
        pso._pbestVal = (double[])s.PbestVal.Clone();
        pso._gbestPos = (double[])s.GbestPos.Clone();
        pso._gbestVal = s.GbestVal;
        pso._t = s.T;
        return pso;
    }
}

/// <summary>Serialized evolving state of a <see cref="Pso" /> optimizer (own JSON; see <c>DeState</c>).</summary>
internal sealed class PsoState
{
    public int NDim { get; set; }
    public int Popsize { get; set; }
    public double W0 { get; set; }
    public double W1 { get; set; }
    public int InertiaDecayGenerations { get; set; }
    public double C1 { get; set; }
    public double C2 { get; set; }
    public double VMax { get; set; }
    public double[][] Pos { get; set; } = Array.Empty<double[]>();
    public double[][] Vel { get; set; } = Array.Empty<double[]>();
    public double[][] PbestPos { get; set; } = Array.Empty<double[]>();
    public double[] PbestVal { get; set; } = Array.Empty<double>();
    public double[] GbestPos { get; set; } = Array.Empty<double>();
    public double GbestVal { get; set; }
    public int T { get; set; }
}
