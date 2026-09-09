using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;
using OptunaSharp.Interop.PythonNet;

namespace OptunaSharp.Interop.PyCma;

/// <summary>
///     A sampler that uses <b>pycma</b> — Nikolaus Hansen's <c>cma</c> package
///     (<c>cma.CMAEvolutionStrategy</c> + <c>cma.BoundTransform</c>) — as the CMA-ES backend, a native
///     OptunaSharp <see cref="BaseSampler" /> that drives the <b>real</b> pycma in-process via pythonnet.
///     Faithful port of <c>optuna_integration.PyCmaSampler</c>: the sampler protocol (intersection search
///     space, startup trials, per-parameter transforms to/from cma space, the ask/tell generation replay,
///     direction flipping) is OptunaSharp C# — "Layer 1" — while the CMA-ES numerics (the covariance
///     update and candidate generation inside <c>ask</c>/<c>tell</c>) are genuine pycma calls. So a seeded
///     run reproduces <c>optuna_integration.PyCmaSampler</c> byte-for-byte.
///
///     <para>Unlike Python, this does <b>not</b> require the <c>optuna_integration</c> package (we port
///     it); install only <c>cma</c> into a discoverable CPython (<c>pip install cma</c>).</para>
///
///     <para><b>Distinct from <see cref="CmaEsSampler" />.</b> OptunaSharp's native
///     <c>CmaEsSampler</c> (Core) ports the separate <a href="https://github.com/CyberAgentAILab/cmaes">
///     cmaes</a> package — a different CMA-ES implementation. <see cref="PyCmaSampler" /> wraps the
///     <c>cma</c> (pycma) package, mirroring Optuna's own <c>optuna.integration.PyCmaSampler</c> vs
///     <c>optuna.samplers.CmaEsSampler</c> split. The two are NOT bit-identical to each other; each is
///     byte-exact against its own backend.</para>
///
///     <para>PyCma optimizes float, int <b>and</b> categorical parameters (categoricals via an ordinal
///     representation), skipping only single-value distributions (which the <see cref="Trial" /> samples).
///     This differs from <see cref="CmaEsSampler" />, which routes categoricals to the independent
///     sampler.</para>
///
///     <para><b>Deprecated in v4.9.0</b> (removal scheduled for v6.0.0 upstream) — mirrors
///     <c>@deprecated_class("4.9.0", "6.0.0")</c>. Prefer the native <see cref="CmaEsSampler" />.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new PyCmaSampler(seed: 42));
///     study.Optimize(t =>
///     {
///         double x = t.SuggestFloat("x", -5, 5);
///         double y = t.SuggestFloat("y", -5, 5);
///         return x * x + y * y;
///     }, nTrials: 50);
///     </code>
/// </example>
public sealed class PyCmaSampler : BaseSampler
{
    private const double Eps = 1e-10;   // cma.py's _EPS

    private static readonly Logger Logger = OptunaLogging.GetLogger("optuna_integration.cma");

    private IReadOnlyDictionary<string, object?>? _x0;   // cached after the first relative sample
    private readonly double? _sigma0;
    private readonly IReadOnlyDictionary<string, double>? _cmaStds;
    private readonly IReadOnlyDictionary<string, object?>? _cmaOpts;
    private readonly int _nStartupTrials;
    private readonly BaseSampler _independentSampler;
    private bool _warnIndependentSampling;
    private readonly IntersectionSearchSpace _searchSpace = new();
    private int _seed;   // put into cma_opts["seed"] on every relative sample

    /// <param name="x0">
    ///     Initial parameter values for CMA-ES (external representation). By default the mean of the low
    ///     and high of each distribution (categorical: the middle choice). See <c>cma.CMAEvolutionStrategy</c>.
    /// </param>
    /// <param name="sigma0">
    ///     Initial standard deviation of CMA-ES. By default <c>min_range / 6</c> over the search space
    ///     (categorical: <c>(len(choices) - 1) / 6</c>). See <c>cma.CMAEvolutionStrategy</c>.
    /// </param>
    /// <param name="cmaStds">
    ///     Per-parameter multipliers of <paramref name="sigma0" /> (default 1.0). See
    ///     <c>cma.CMAEvolutionStrategy</c>'s <c>CMA_stds</c>.
    /// </param>
    /// <param name="seed">A random seed for CMA-ES (and the default independent sampler).</param>
    /// <param name="cmaOpts">
    ///     Options passed to the <c>cma.CMAEvolutionStrategy</c> constructor. The <c>BoundaryHandler</c>,
    ///     <c>bounds</c>, <c>CMA_stds</c> and <c>seed</c> entries are ignored (the sampler sets them
    ///     automatically); everything else (e.g. <c>verbose</c>, which defaults to <c>-2</c>) is honored.
    ///     Values must be pythonnet-convertible primitives (int / float / bool / string).
    /// </param>
    /// <param name="nStartupTrials">
    ///     The independent sampler is used instead of CMA-ES until this many trials finish. Defaults to 1.
    /// </param>
    /// <param name="independentSampler">
    ///     Sampler for parameters outside the relative search space (default <see cref="RandomSampler" />
    ///     with the same <paramref name="seed" />).
    /// </param>
    /// <param name="warnIndependentSampling">Emit a warning when a parameter is sampled independently.</param>
    public PyCmaSampler(
        IReadOnlyDictionary<string, object?>? x0 = null,
        double? sigma0 = null,
        IReadOnlyDictionary<string, double>? cmaStds = null,
        int? seed = null,
        IReadOnlyDictionary<string, object?>? cmaOpts = null,
        int nStartupTrials = 1,
        BaseSampler? independentSampler = null,
        bool warnIndependentSampling = true)
    {
        // @deprecated_class("4.9.0", "6.0.0") — mirror Optuna's FutureWarning on instantiation.
        OptunaWarnings.Warn(
            string.Format(Deprecations.Template, "PyCmaSampler", "4.9.0", "6.0.0"),
            WarningCategory.FutureWarning);

        // cma.py's `_imports.check()` — fail fast with a helpful message if pycma is missing.
        PythonRuntime.Run(Python.cma.check);

        // Python: `if seed is None: seed = random.randint(1, 2**32)`, then the SAME resolved seed is
        // shared with the default RandomSampler. An explicit seed is used verbatim (the reproducible
        // path); a null seed resolves to an OS-random int (non-reproducible, like Python).
        int resolvedSeed = seed ?? Random.Shared.Next(1, int.MaxValue);

        _x0 = x0;
        _sigma0 = sigma0;
        _cmaStds = cmaStds;
        _seed = resolvedSeed;
        _cmaOpts = cmaOpts;
        _nStartupTrials = nStartupTrials;
        _independentSampler = independentSampler ?? new RandomSampler(resolvedSeed);
        _warnIndependentSampling = warnIndependentSampling;
    }

    public override void ReseedRng()
    {
        // Python: self._cma_opts["seed"] = random.randint(1, 2**32); self._independent_sampler.reseed_rng()
        _seed = Random.Shared.Next(1, int.MaxValue);
        _independentSampler.ReseedRng();
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // `cma` cannot handle distributions that contain just a single value, so we skip them. Note
            // that the parameter values for such distributions are sampled in `Trial`. (Unlike
            // CmaEsSampler, categoricals are KEPT — PyCma optimizes them as ordinals.)
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        RaiseErrorIfMultiObjective(study);

        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        if (searchSpace.Count == 1)
        {
            Logger.Info(
                "`PyCmaSampler` does not support optimization of 1-D search space. " +
                $"`{_independentSampler.GetType().Name}` is used instead of `PyCmaSampler`.");
            _warnIndependentSampling = false;
            return new OrderedDict<string, object?>();
        }

        var completeTrials = study.GetTrialsInternal(
            deepcopy: false, states: new[] { TrialState.Complete }, useCache: true);
        if (completeTrials.Count < _nStartupTrials)
            return new OrderedDict<string, object?>();

        _x0 ??= InitializeX0(searchSpace);
        double sigma0 = _sigma0 ?? InitializeSigma0(searchSpace);
        // Avoid ZeroDivisionError in cma.CMAEvolutionStrategy.
        sigma0 = Math.Max(sigma0, Eps);

        // `study.trials` — all trials, for the tell/ask compatibility scan.
        List<FrozenTrial> trials = study.GetTrialsInternal(deepcopy: false, states: null, useCache: false);
        StudyDirection direction = study.Direction;

        // The pycma optimizer lives ONLY inside this GIL scope: built, replayed (tell), asked, disposed.
        return PythonRuntime.Run(() =>
        {
            using var optimizer = new Optimizer(searchSpace, _x0!, sigma0, _cmaStds, _cmaOpts, _seed);
            int lastToldTrialNumber = optimizer.Tell(trials, direction);
            return optimizer.Ask(trials, lastToldTrialNumber);
        });
    }

    /// <summary>cma.py's <c>_initialize_x0</c> — the default initial mean (external representation).</summary>
    private static IReadOnlyDictionary<string, object?> InitializeX0(OrderedDict<string, BaseDistribution> searchSpace)
    {
        var x0 = new Dictionary<string, object?>();
        foreach (var (name, distribution) in searchSpace)
        {
            switch (distribution)
            {
                case FloatDistribution fd:
                    x0[name] = fd.Log
                        ? Math.Exp((Math.Log(fd.High) + Math.Log(fd.Low)) / 2.0)   // exp(mean([logHigh, logLow]))
                        : (fd.High + fd.Low) / 2.0;                                 // mean([high, low])
                    break;
                case CategoricalDistribution cd:
                    x0[name] = cd.Choices[(cd.Choices.Count - 1) / 2].Value;
                    break;
                case IntDistribution id:
                    x0[name] = id.Log
                        ? Math.Exp((Math.Log(id.High) + Math.Log(id.Low)) / 2.0)
                        : (object)(long)((id.High + id.Low) / 2.0);                 // int(mean([high, low]))
                    break;
                default:
                    throw new NotImplementedException($"The distribution {distribution} is not implemented.");
            }
        }
        return x0;
    }

    /// <summary>cma.py's <c>_initialize_sigma0</c> — the default step size (min range / 6).</summary>
    private static double InitializeSigma0(OrderedDict<string, BaseDistribution> searchSpace)
    {
        var sigma0s = new List<double>();
        foreach (var (name, distribution) in searchSpace)
        {
            switch (distribution)
            {
                case IntDistribution id:
                    sigma0s.Add(id.Log ? (Math.Log(id.High) - Math.Log(id.Low)) / 6.0 : (id.High - id.Low) / 6.0);
                    break;
                case FloatDistribution fd:
                    sigma0s.Add(fd.Log ? (Math.Log(fd.High) - Math.Log(fd.Low)) / 6.0 : (fd.High - fd.Low) / 6.0);
                    break;
                case CategoricalDistribution cd:
                    sigma0s.Add((cd.Choices.Count - 1) / 6.0);
                    break;
                default:
                    throw new NotImplementedException($"The distribution {distribution} is not implemented.");
            }
        }
        return sigma0s.Min();
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        RaiseErrorIfMultiObjective(study);

        if (_warnIndependentSampling)
        {
            var completeTrials = study.GetTrialsInternal(
                deepcopy: false, states: new[] { TrialState.Complete }, useCache: true);
            if (completeTrials.Count >= _nStartupTrials)
                LogIndependentSampling(trial, paramName);
        }

        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    /// <summary>cma.py's <c>_log_independent_sampling</c> (verbatim message).</summary>
    private void LogIndependentSampling(FrozenTrial trial, string paramName)
    {
        Logger.Warning(
            $"The parameter '{paramName}' in trial#{trial.Number} is sampled independently " +
            $"by using `{_independentSampler.GetType().Name}` instead of `PyCmaSampler` " +
            "(optimization performance may be degraded). " +
            "`PyCmaSampler` does not support dynamic search space or `CategoricalDistribution`. " +
            "You can suppress this warning by setting `warn_independent_sampling` " +
            "to `False` in the constructor of `PyCmaSampler`, " +
            "if this independent sampling is intended behavior.");
    }

    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => _independentSampler.AfterTrial(study, trial, state, values);

    // ------------------------------------------------------------------------------------------------
    //  cma.py's `_Optimizer` — builds the real cma.CMAEvolutionStrategy and drives ask/tell. Every
    //  member touches PyObjects and is called under the GIL (from SampleRelative's PythonRuntime.Run
    //  body). The parameter transforms are pure C# (bit-exact log/exp/round/mean); only the crossing +
    //  the cma calls need the GIL. The optimizer object never escapes the Run scope (Dispose frees it).
    // ------------------------------------------------------------------------------------------------
    private sealed class Optimizer : IDisposable
    {
        private static readonly TrialState[] CompleteOnly = { TrialState.Complete };

        private readonly OrderedDict<string, BaseDistribution> _searchSpace;
        private readonly List<string> _paramNames;
        private readonly PyObject _es;
        private readonly long _popsize;

        internal Optimizer(
            OrderedDict<string, BaseDistribution> searchSpace,
            IReadOnlyDictionary<string, object?> x0,
            double sigma0,
            IReadOnlyDictionary<string, double>? cmaStds,
            IReadOnlyDictionary<string, object?>? cmaOpts,
            int seed)
        {
            _searchSpace = searchSpace;
            _paramNames = searchSpace.Keys.OrderBy(k => k, StringComparer.Ordinal).ToList();
            int dim = _paramNames.Count;

            // Per-dimension bounds in cma space (cma.py's `_Optimizer.__init__`).
            var lows = new double[dim];
            var highs = new double[dim];
            for (int i = 0; i < dim; i++)
            {
                BaseDistribution dist = _searchSpace[_paramNames[i]];
                switch (dist)
                {
                    case CategoricalDistribution cd:
                        // Ordinal representation.
                        lows[i] = -0.5;
                        highs[i] = cd.Choices.Count - 0.5;
                        break;
                    case FloatDistribution fd when fd.Step is not null:
                        double r = fd.High - fd.Low;
                        lows[i] = 0 - 0.5 * fd.Step.Value;
                        highs[i] = r + 0.5 * fd.Step.Value;
                        break;
                    case FloatDistribution fd:
                        lows[i] = ToCmaParams(fd, fd.Low);
                        highs[i] = ToCmaParams(fd, fd.High) - Eps;
                        break;
                    case IntDistribution id when id.Log:
                        lows[i] = ToCmaParams(id, id.Low - 0.5);
                        highs[i] = ToCmaParams(id, id.High + 0.5);
                        break;
                    case IntDistribution id:
                        lows[i] = id.Low - 0.5 * id.Step;
                        highs[i] = id.High + 0.5 * id.Step;
                        break;
                    default:
                        throw new NotImplementedException($"The distribution {dist} is not implemented.");
                }
            }

            var initialCmaParams = new double[dim];
            for (int i = 0; i < dim; i++)
                initialCmaParams[i] = ToCmaParams(_searchSpace[_paramNames[i]], x0[_paramNames[i]]);

            // Build the cma options dict: user opts, then seed (forced) + verbose (setdefault -2), then
            // the sampler-managed BoundaryHandler / bounds / CMA_stds (forced). cma.py sets seed +
            // verbose in PyCmaSampler.__init__ and BoundaryHandler/bounds/CMA_stds in _Optimizer; the
            // net dict is the same.
            using var opts = new PyDict();
            if (cmaOpts is not null)
                foreach (var (k, v) in cmaOpts)
                {
                    if (v is null) { opts[k] = PyObject.None; continue; }
                    using PyObject pv = v.ToPython();
                    opts[k] = pv;
                }
            using (PyObject seedPy = new PyInt(seed))
                opts["seed"] = seedPy;
            if (cmaOpts is null || !cmaOpts.ContainsKey("verbose"))
                using (PyObject verbosePy = new PyInt(-2))
                    opts["verbose"] = verbosePy;
            using (PyObject boundaryHandler = Python.cma.BoundTransform)
                opts["BoundaryHandler"] = boundaryHandler;
            using (PyObject lowsList = PyCmaInterop.ToPyList(lows))
            using (PyObject highsList = PyCmaInterop.ToPyList(highs))
            using (PyObject bounds = new PyList(new[] { lowsList, highsList }))
                opts["bounds"] = bounds;
            if (cmaStds is not null)
            {
                var stds = new double[dim];
                for (int i = 0; i < dim; i++)
                    stds[i] = cmaStds.TryGetValue(_paramNames[i], out double s) ? s : 1.0;
                using PyObject stdsList = PyCmaInterop.ToPyList(stds);
                opts["CMA_stds"] = stdsList;
            }

            using PyObject x0List = PyCmaInterop.ToPyList(initialCmaParams);
            _es = Python.cma.CMAEvolutionStrategy.New(x0List, sigma0, opts);
            _popsize = Python.cma.CMAEvolutionStrategy.popsize_t(_es);
        }

        /// <summary>
        ///     cma.py's <c>_Optimizer.tell</c> — replay each full generation of completed trials into
        ///     pycma. Returns the number of the last trial told (or -1 if none). The <c>ask()</c> before
        ///     each <c>tell()</c> is required by pycma's state machine.
        /// </summary>
        internal int Tell(IReadOnlyList<FrozenTrial> trials, StudyDirection direction)
        {
            List<FrozenTrial> completeTrials = CollectTargetTrials(trials, lastTold: -1, targetStates: CompleteOnly);

            int popsize = (int)_popsize;
            int generation = completeTrials.Count / popsize;
            int dim = _paramNames.Count;
            int lastToldTrialNumber = -1;
            for (int g = 0; g < generation; g++)
            {
                var xs = new double[popsize, dim];
                var ys = new double[popsize];
                for (int k = 0; k < popsize; k++)
                {
                    FrozenTrial t = completeTrials[g * popsize + k];
                    for (int d = 0; d < dim; d++)
                        xs[k, d] = ToCmaParams(_searchSpace[_paramNames[d]], t.Params[_paramNames[d]]);
                    double y = t.Value!.Value;
                    // BoTorch/cma minimize; MAXIMIZE flips the sign.
                    ys[k] = direction == StudyDirection.Maximize ? -1 * y : y;
                    lastToldTrialNumber = t.Number;
                }

                using (PyObject discard = Python.cma.CMAEvolutionStrategy.ask(_es)) { }
                using PyObject xsList = PyCmaInterop.ToPyList(xs);
                using PyObject ysList = PyCmaInterop.ToPyList(ys);
                Python.cma.CMAEvolutionStrategy.tell(_es, xsList, ysList);
            }
            return lastToldTrialNumber;
        }

        /// <summary>
        ///     cma.py's <c>_Optimizer.ask</c> — the next individual's parameters. The index is the count of
        ///     compatible trials after the last told one; under parallel execution it can exceed the
        ///     population, so extra <c>ask()</c>s are burned (matching pycma's behaviour).
        /// </summary>
        internal OrderedDict<string, object?> Ask(IReadOnlyList<FrozenTrial> trials, int lastToldTrialNumber)
        {
            int individualIndex = CollectTargetTrials(trials, lastToldTrialNumber, targetStates: null).Count;
            int popsize = (int)_popsize;
            while (individualIndex >= popsize)
            {
                individualIndex -= popsize;
                using (PyObject discard = Python.cma.CMAEvolutionStrategy.ask(_es)) { }
            }

            double[] cmaParams;
            using (PyObject askList = Python.cma.CMAEvolutionStrategy.ask(_es))
            using (PyObject row = askList.GetItem(individualIndex))
                cmaParams = PyCmaInterop.ToArray(row);

            var ret = new OrderedDict<string, object?>();
            for (int i = 0; i < _paramNames.Count; i++)
                ret[_paramNames[i]] = ToOptunaParams(_searchSpace[_paramNames[i]], cmaParams[i]);
            return ret;
        }

        /// <summary>
        ///     cma.py's <c>_is_compatible</c>: every search-space parameter is present in the trial, its
        ///     distribution is compatible, and its value lies inside the distribution. In sequential runs
        ///     the intersection search space guarantees compatibility; the check matters for distributed
        ///     runs where an incompatible trial may complete concurrently.
        /// </summary>
        private bool IsCompatible(FrozenTrial trial)
        {
            foreach (var (name, distribution) in _searchSpace)
            {
                if (!trial.Params.ContainsKey(name))
                    return false;
                Distribution.CheckDistributionCompatibility(distribution, trial.Distributions[name]);
                double internalRepr = distribution.ToInternalRepr(trial.Params[name]);
                if (!distribution.Contains(internalRepr))
                    return false;
            }
            return true;
        }

        /// <summary>cma.py's <c>_collect_target_trials</c>.</summary>
        private List<FrozenTrial> CollectTargetTrials(
            IReadOnlyList<FrozenTrial> trials, int lastTold, TrialState[]? targetStates)
        {
            var result = new List<FrozenTrial>();
            foreach (FrozenTrial t in trials)
            {
                if (t.Number <= lastTold)
                    continue;
                if (!IsCompatible(t))
                    continue;
                if (targetStates is not null && Array.IndexOf(targetStates, t.State) < 0)
                    continue;
                result.Add(t);
            }
            return result;
        }

        /// <summary>cma.py's <c>_to_cma_params</c> — external parameter value → cma-space coordinate.</summary>
        private static double ToCmaParams(BaseDistribution dist, object? value)
        {
            switch (dist)
            {
                case IntDistribution id when id.Log:
                    return Math.Log(Convert.ToDouble(value));
                case FloatDistribution fd when fd.Log:
                    return Math.Log(Convert.ToDouble(value));
                case FloatDistribution fd when fd.Step is not null:
                    return Convert.ToDouble(value) - fd.Low;
                case CategoricalDistribution cd:
                    return cd.ToInternalRepr(value);   // choices.index(value)
                default:
                    return Convert.ToDouble(value);
            }
        }

        /// <summary>cma.py's <c>_to_optuna_params</c> — cma-space coordinate → external parameter value.</summary>
        private static object? ToOptunaParams(BaseDistribution dist, double cmaParamValue)
        {
            switch (dist)
            {
                case FloatDistribution fd:
                    if (fd.Log)
                        return Math.Exp(cmaParamValue);
                    if (fd.Step is not null)
                    {
                        double v = NpRound(cmaParamValue / fd.Step.Value) * fd.Step.Value + fd.Low;
                        return Math.Min(Math.Max(v, fd.Low), fd.High);
                    }
                    return cmaParamValue;

                case IntDistribution id:
                    if (id.Log)
                    {
                        double v = NpRound(Math.Exp(cmaParamValue));
                        return (long)Math.Min(Math.Max(v, id.Low), id.High);
                    }
                    else
                    {
                        double r = NpRound((cmaParamValue - id.Low) / id.Step);
                        return (long)(r * id.Step + id.Low);
                    }

                case CategoricalDistribution cd:
                    return cd.Choices[(int)NpRound(cmaParamValue)].Value;

                default:
                    return cmaParamValue;
            }
        }

        /// <summary><c>numpy.round</c> — round half to even (banker's rounding), matching numpy's <c>rint</c>.</summary>
        private static double NpRound(double x) => Math.Round(x, MidpointRounding.ToEven);

        public void Dispose() => _es.Dispose();
    }
}
