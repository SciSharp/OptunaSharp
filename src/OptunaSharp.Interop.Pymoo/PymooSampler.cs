using System;
using System.Collections.Generic;
using NumSharp;
using Python.Runtime;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;
using OptunaSharp.SearchSpace;
using OptunaSharp.Interop.PythonNet;

namespace OptunaSharp.Interop.Pymoo;

/// <summary>
///     A multi-objective sampler that drives the <b>real pymoo</b> in-process via pythonnet: one
///     OptunaSharp <see cref="BaseSampler" /> that exposes pymoo's whole multi-objective evolutionary
///     zoo through the <see cref="PymooAlgorithm" /> enum (NSGA-II/III, R-NSGA-II/III, U-NSGA-III,
///     AGE-MOEA/AGE-MOEA-2, RVEA, SMS-EMOA, SPEA2, C-TAEA). The Optuna side — study, storage,
///     ask/tell, search space, direction flipping, seeding — is native OptunaSharp; the operators,
///     reference-direction machinery and hypervolume/decomposition survival are genuine pymoo.
///
///     <para><b>How it works — the reconstruct-from-trials bridge.</b> pymoo exposes a native
///     <c>ask()</c>/<c>tell()</c> interface, so — exactly like <c>OptunaSharp.Interop.PyCma</c>'s
///     <c>PyCmaSampler</c> — the pymoo <c>Algorithm</c> lives <b>only</b> inside a single
///     <see cref="SampleRelative" /> GIL scope: built fresh, seeded, and replayed generation by
///     generation from the study's completed trials (for each full generation, <c>ask()</c> the infill
///     population, stamp its objectives <c>F</c> from the trials' stored values, and <c>tell()</c> it),
///     then <c>ask()</c>ed once more to dole out one individual's parameters for the current trial. The
///     study storage is the single source of truth — nothing pymoo-side persists between trials — which
///     keeps the sampler correct under resumption and reproducible under a fixed seed (pymoo's
///     <c>ask()</c> is deterministic under <c>seed</c>, so rebuilding-and-replaying reproduces the exact
///     population it doled out — verified: a rebuild-each-generation replay equals a single continuous
///     run). This is the same one-row-per-trial doling as OptunaSharp's native <c>NSGAIISampler</c> /
///     <c>AgeMoea2Sampler</c>.</para>
///
///     <para><b>The initial generation.</b> Like every relative sampler under Optuna's define-by-run
///     model (and the native <c>NSGAIISampler</c> / <c>AgeMoea2Sampler</c>), the first
///     <c>population_size</c> trials — generation 0 — are sampled by the <see cref="_independentSampler" />
///     (a <see cref="RandomSampler" /> by default), because the search space is not known until at least
///     one trial has completed. Once generation 0 is complete, pymoo takes over: it is rebuilt with that
///     generation <b>injected as its initial population</b> (pymoo's <c>sampling=</c>), so the tell-replay
///     is coherent (pymoo's population is exactly the trials it is told the objectives of) and every
///     later generation is pymoo's genuine offspring. The independent sampler also covers parameters
///     <b>outside</b> the relative search space (single-value and conditional/dynamic parameters).</para>
///
///     <para><b>Scope of the parity guarantee.</b> Validated <b>behaviorally</b> (every algorithm
///     completes a seeded multi-objective study, drives a spreading front, and reproduces across runs)
///     and by <b>value-parity vs live pymoo</b> (given the same injected initial generation and seed, the
///     bridge's trajectory matches a direct pymoo <c>ask</c>/<c>tell</c> run). It is <b>not</b> bit-exact
///     with a monolithic <c>pymoo.minimize</c> through the sampler: generation 0 uses OptunaSharp's RNG
///     rather than pymoo's own initial sampling, and the Optuna search-space transform differs from a
///     monolithic run. Bit-exactness is the job of the <em>native</em> Layer-1 optimizers (e.g.
///     <c>AgeMoea2Optimizer</c>). Beyond Optuna core (Optuna ships only NSGA-II/III), this is an
///     OptunaSharp extension — the live-runtime complement to the native <c>AgeMoea2Sampler</c>.</para>
///
///     <para>Unlike a Python glue package, this requires no <c>optuna_integration</c>-style dependency;
///     install only <c>pymoo</c> into a discoverable CPython (<c>pip install pymoo</c>).</para>
///
///     <para><b>Search-space transform (the one lossy seam).</b> Parameters are mapped to a continuous
///     unit cube via the bit-exact one-hot <see cref="SearchSpaceTransform" /> (<c>transform01</c>):
///     <c>float</c>/<c>int</c> (with <c>log</c>/<c>step</c>) scale to <c>[0,1]</c>, and categoricals are
///     one-hot encoded (pymoo's continuous operators move each one-hot column, and the argmax on the way
///     back selects the choice). Integer and stepped parameters are rounded on untransform.</para>
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.Minimize },
///                                    sampler: new PymooSampler(PymooAlgorithm.SmsEmoa, populationSize: 40, seed: 0));
///     study.Optimize(t =>
///     {
///         double x = t.SuggestFloat("x", 0, 1);
///         double y = t.SuggestFloat("y", 0, 1);
///         return new[] { x, (1 - x) + y };   // a simple 2-objective trade-off
///     }, nTrials: 400);
///     </code>
/// </example>
public sealed class PymooSampler : BaseSampler
{
    private readonly PymooAlgorithm _algorithm;
    private readonly PymooAlgorithmOptions _options;
    private readonly int _populationSize;
    private readonly BaseSampler _independentSampler;
    private readonly IntersectionSearchSpace _searchSpace = new();
    private int _seed;

    /// <param name="algorithm">Which pymoo multi-objective algorithm to drive. Default
    /// <see cref="PymooAlgorithm.Nsga2" />.</param>
    /// <param name="populationSize">
    ///     Number of individuals per generation, for the algorithms that take a <c>pop_size</c>
    ///     (NSGA-II, R-NSGA-II, AGE-MOEA, AGE-MOEA-2, SMS-EMOA, SPEA2). For the
    ///     reference-direction algorithms (NSGA-III, U-NSGA-III, RVEA, C-TAEA) and R-NSGA-III the
    ///     population size is derived by pymoo from the reference directions / points and this is ignored
    ///     (for the Riesz-energy reference directions it is the default direction count). Must be &gt;= 2.
    /// </param>
    /// <param name="seed">A random seed for pymoo (and the default independent sampler). <c>null</c> ⇒ a
    /// non-reproducible OS-random seed, mirroring pymoo.</param>
    /// <param name="options">Algorithm-specific knobs (reference directions / points / …). Required
    /// content depends on <paramref name="algorithm" /> — see <see cref="PymooAlgorithmOptions" />.</param>
    /// <param name="independentSampler">Sampler for the initial generation and for parameters outside the
    /// relative search space (default <see cref="RandomSampler" /> with the same <paramref name="seed" />).</param>
    public PymooSampler(
        PymooAlgorithm algorithm = PymooAlgorithm.Nsga2,
        int populationSize = 50,
        int? seed = null,
        PymooAlgorithmOptions? options = null,
        BaseSampler? independentSampler = null)
    {
        if (populationSize < 2)
            throw new NumSharp.ValueError("`population_size` must be greater than or equal to 2.");

        // Fail fast with a helpful "pip install pymoo" message if pymoo is missing.
        PythonRuntime.Run(Python.pymoo.check);

        // An explicit seed is used verbatim (the reproducible path); a null seed resolves to an
        // OS-random int (non-reproducible), and the SAME resolved seed is shared with the default
        // RandomSampler.
        int resolvedSeed = seed ?? Random.Shared.Next(1, int.MaxValue);

        _algorithm = algorithm;
        _populationSize = populationSize;
        _options = options ?? new PymooAlgorithmOptions();
        _seed = resolvedSeed;
        _independentSampler = independentSampler ?? new RandomSampler(resolvedSeed);
    }

    public override void ReseedRng()
    {
        _seed = Random.Shared.Next(1, int.MaxValue);
        _independentSampler.ReseedRng();
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, distribution) in _searchSpace.Calculate(study))
        {
            // pymoo cannot optimize a distribution that contains just a single value; skip it (the
            // parameter is sampled in the Trial).
            if (distribution.Single())
                continue;
            searchSpace[name] = distribution;
        }
        return searchSpace;
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        var trans = new SearchSpaceTransform(searchSpace, transform01: true);
        IReadOnlyList<StudyDirection> directions = study.Directions;

        // All trials (any state) — the complete ones drive the tell-replay and seed pymoo's initial
        // population; the any-state count after the last told generation gives the current trial's index
        // within its generation.
        List<FrozenTrial> trials = study.GetTrialsInternal(deepcopy: false, states: null, useCache: false);

        // The pymoo optimizer lives ONLY inside this GIL scope: built, replayed (tell), asked, disposed.
        return PythonRuntime.Run(() =>
        {
            using var optimizer = new Optimizer(
                _algorithm, _options, _populationSize, _seed, trans, searchSpace, directions);

            List<FrozenTrial> complete = optimizer.CollectComplete(trials);
            // Generation 0 is not complete yet — defer to the independent sampler (the house pattern:
            // the search space is unknown until trials complete, so pymoo cannot own generation 0).
            if (complete.Count < optimizer.PopulationSize)
                return new OrderedDict<string, object?>();

            // Rebuild pymoo with generation 0 injected as its initial population, replay, dole.
            optimizer.Reinitialize(complete);
            int lastToldTrialNumber = optimizer.Tell(complete);
            return optimizer.Ask(trials, lastToldTrialNumber);
        });
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        // The initial generation and any out-of-space parameter are sampled independently — same as the
        // native NSGAIISampler / AgeMoea2Sampler (which do not warn: independent sampling is expected).
        => _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);

    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => _independentSampler.AfterTrial(study, trial, state, values);

    // ------------------------------------------------------------------------------------------------
    //  Builds the real pymoo Problem + Algorithm and drives its native ask/tell. Every member touches
    //  PyObjects and is called under the GIL (from SampleRelative's PythonRuntime.Run body). The
    //  optimizer object never escapes the Run scope (Dispose frees the problem + algorithm).
    // ------------------------------------------------------------------------------------------------
    private sealed class Optimizer : IDisposable
    {
        private static readonly TrialState[] CompleteOnly = { TrialState.Complete };

        private readonly PymooAlgorithm _algorithm;
        private readonly PymooAlgorithmOptions _options;
        private readonly int _populationSize;
        private readonly int _seed;
        private readonly SearchSpaceTransform _trans;
        private readonly OrderedDict<string, BaseDistribution> _searchSpace;
        private readonly IReadOnlyList<StudyDirection> _directions;
        private readonly int _nBounds;
        private readonly int _nObj;

        private readonly PyObject _problem;
        private PyObject _algo;
        private readonly int _popsize;

        internal int PopulationSize => _popsize;

        internal Optimizer(
            PymooAlgorithm algorithm,
            PymooAlgorithmOptions options,
            int populationSize,
            int seed,
            SearchSpaceTransform trans,
            OrderedDict<string, BaseDistribution> searchSpace,
            IReadOnlyList<StudyDirection> directions)
        {
            _algorithm = algorithm;
            _options = options;
            _populationSize = populationSize;
            _seed = seed;
            _trans = trans;
            _searchSpace = searchSpace;
            _directions = directions;
            _nObj = directions.Count;
            _nBounds = trans.RawBounds.GetLength(0);

            // Under transform01 the parameters live in the unit cube, so the pymoo box is [0,1]^n.
            var xl = new double[_nBounds];
            var xu = new double[_nBounds];
            for (int i = 0; i < _nBounds; i++)
            {
                xl[i] = 0.0;
                xu[i] = 1.0;
            }
            using (PyObject xlPy = PymooInterop.ToNumpy(xl))
            using (PyObject xuPy = PymooInterop.ToNumpy(xu))
                _problem = Python.pymoo.Problem(_nBounds, _nObj, xlPy, xuPy);

            // Phase 1: build without an injected initial population, only to read pymoo's authoritative
            // population size (derived, for the reference-direction / reference-point algorithms — it is
            // NOT reliably a simple formula, so we read it rather than compute it).
            _algo = BuildAlgorithm(sampling: null);
            _popsize = (int)Python.pymoo.Algorithm.pop_size_t(_algo);
        }

        /// <summary>The completed, compatible trials, in trial-number (= doling) order.</summary>
        internal List<FrozenTrial> CollectComplete(IReadOnlyList<FrozenTrial> trials)
        {
            List<FrozenTrial> complete = CollectTargetTrials(trials, lastTold: -1, CompleteOnly);
            complete.Sort((a, b) => a.Number.CompareTo(b.Number));
            return complete;
        }

        /// <summary>Rebuild pymoo with the first <see cref="PopulationSize" /> completed trials injected as
        /// its initial population (their transformed parameters), so the tell-replay is coherent.</summary>
        internal void Reinitialize(IReadOnlyList<FrozenTrial> complete)
        {
            var gen0 = new double[_popsize, _nBounds];
            for (int k = 0; k < _popsize; k++)
            {
                double[] x = _trans.Transform(complete[k].ParamsDict).ToArray<double>();
                for (int d = 0; d < _nBounds; d++)
                    gen0[k, d] = x[d];
            }

            _algo.Dispose();
            using PyObject sampling = PymooInterop.ToNumpy(gen0);
            _algo = BuildAlgorithm(sampling);
        }

        /// <summary>
        ///     Replay every full generation of completed trials into pymoo: for each generation
        ///     <c>ask()</c> the infill population (deterministic under the seed — the exact rows doled out
        ///     earlier; generation 0 is the injected initial population), stamp its objective matrix
        ///     <c>F</c> from the trials' stored values (with pymoo's minimization sign flip), and
        ///     <c>tell()</c> it. Returns the number of the last trial told (or -1 if none). The completed
        ///     trials are grouped into generations by trial-number order, which equals the doling order
        ///     under sequential execution (the bridge's scope).
        /// </summary>
        internal int Tell(IReadOnlyList<FrozenTrial> complete)
        {
            int generation = complete.Count / _popsize;
            int lastToldTrialNumber = -1;
            for (int g = 0; g < generation; g++)
            {
                var f = new double[_popsize, _nObj];
                for (int k = 0; k < _popsize; k++)
                {
                    FrozenTrial t = complete[g * _popsize + k];
                    IReadOnlyList<double> values = t.Values!;
                    for (int j = 0; j < _nObj; j++)
                        // pymoo MINIMIZES every objective; a MAXIMIZE direction flips the sign.
                        f[k, j] = _directions[j] == StudyDirection.Maximize ? -values[j] : values[j];
                    if (t.Number > lastToldTrialNumber)
                        lastToldTrialNumber = t.Number;
                }

                using PyObject pop = Python.pymoo.Algorithm.ask(_algo);
                using PyObject fNumpy = PymooInterop.ToNumpy(f);
                Python.pymoo.Population.set(pop, "F", fNumpy);
                Python.pymoo.Algorithm.tell(_algo, pop);
            }
            return lastToldTrialNumber;
        }

        /// <summary>
        ///     The next individual's parameters. The index within the current generation is the count of
        ///     compatible trials (any state, their parameters already present) after the last told trial —
        ///     the current trial itself has no parameters yet, so it is not counted (sequential: 0 for the
        ///     first of a generation, 1 for the second, …). Under parallel over-subscription the index can
        ///     exceed the population; it is wrapped into the current infill population (a safe clamp — the
        ///     bridge targets sequential / mildly-parallel execution).
        /// </summary>
        internal OrderedDict<string, object?> Ask(IReadOnlyList<FrozenTrial> trials, int lastToldTrialNumber)
        {
            int individualIndex = CollectTargetTrials(trials, lastToldTrialNumber, targetStates: null).Count;
            individualIndex %= _popsize;

            double[] flat;
            using (PyObject pop = Python.pymoo.Algorithm.ask(_algo))
            using (PyObject xNumpy = Python.pymoo.Population.get(pop, "X"))
                flat = PymooInterop.ToNDArray(xNumpy).ToArray<double>();   // C-order (popsize * nBounds)

            var row = new double[_nBounds];
            Array.Copy(flat, individualIndex * _nBounds, row, 0, _nBounds);
            return _trans.Untransform(np.array(row));
        }

        /// <summary>Build the pymoo algorithm (with an optional injected initial population) and
        /// <c>setup</c> it against the problem + seed. Under the GIL.</summary>
        private PyObject BuildAlgorithm(PyObject? sampling)
        {
            (string module, string className) = PymooAlgorithmMap.Resolve(_algorithm);
            using PyDict kwargs = BuildAlgorithmKwargs(_algorithm, _options, _populationSize, _nObj, _seed, _directions);
            if (sampling is not null)
                kwargs["sampling"] = sampling;
            PyObject algo = Python.pymoo.CreateAlgorithm(module, className, kwargs);
            Python.pymoo.Algorithm.setup(algo, _problem, _seed);
            return algo;
        }

        /// <summary>
        ///     Build the family-specific keyword arguments for the pymoo algorithm constructor: a
        ///     <c>pop_size</c> for the population-size algorithms; <c>ref_dirs</c> (Das–Dennis or Riesz
        ///     energy) for the reference-direction algorithms; <c>ref_points</c> (+ <c>epsilon</c> /
        ///     <c>pop_size</c> or <c>pop_per_ref_point</c>) for the reference-point algorithms. The caller
        ///     disposes the returned dict. Under the GIL.
        /// </summary>
        private static PyDict BuildAlgorithmKwargs(
            PymooAlgorithm algorithm, PymooAlgorithmOptions options, int populationSize, int nObj, int seed,
            IReadOnlyList<StudyDirection> directions)
        {
            var kwargs = new PyDict();
            try
            {
                if (PymooAlgorithmMap.UsesReferenceDirections(algorithm))
                {
                    using PyObject refDirs = BuildReferenceDirections(options, nObj, populationSize, seed);
                    kwargs["ref_dirs"] = refDirs;
                }
                else if (PymooAlgorithmMap.UsesReferencePoints(algorithm))
                {
                    if (options.ReferencePoints is null || options.ReferencePoints.Length == 0)
                        throw new NumSharp.ValueError(
                            $"{algorithm} requires reference points; set PymooAlgorithmOptions.ReferencePoints.");
                    using PyObject refPoints = BuildReferencePoints(options.ReferencePoints, nObj, directions);
                    kwargs["ref_points"] = refPoints;
                    if (algorithm == PymooAlgorithm.RNsga3)
                    {
                        using PyObject popPerRef = new PyInt(options.PopulationPerReferencePoint);
                        kwargs["pop_per_ref_point"] = popPerRef;
                    }
                    else // RNsga2
                    {
                        using (PyObject popSize = new PyInt(populationSize))
                            kwargs["pop_size"] = popSize;
                        // (PyFloat is ambiguous with OptunaSharp.Distributions.PyFloat, and the root
                        //  `Python` class shadows the Python.Runtime namespace here — cross via ToPython.)
                        using (PyObject epsilon = options.Epsilon.ToPython())
                            kwargs["epsilon"] = epsilon;
                    }
                }
                else // population-size family
                {
                    using PyObject popSize = new PyInt(populationSize);
                    kwargs["pop_size"] = popSize;
                }
                return kwargs;
            }
            catch
            {
                kwargs.Dispose();
                throw;
            }
        }

        /// <summary><c>get_reference_directions("das-dennis", n_obj, n_partitions=…)</c> or
        /// <c>("energy", n_obj, n_points, seed=…)</c>. Dispose the result. Under the GIL.</summary>
        private static PyObject BuildReferenceDirections(
            PymooAlgorithmOptions options, int nObj, int populationSize, int seed)
        {
            using var kwargs = new PyDict();
            if (options.ReferenceDirectionsMethod == PymooReferenceDirections.Energy)
            {
                int nPoints = options.NumberOfReferenceDirections ?? populationSize;
                using PyObject nPointsPy = new PyInt(nPoints);
                using (PyObject seedPy = new PyInt(seed))
                    kwargs["seed"] = seedPy;
                return Python.pymoo.get_reference_directions("energy", nObj, kwargs, nPointsPy);
            }

            int nPartitions = options.NumberOfPartitions ?? 12;
            using (PyObject nPartitionsPy = new PyInt(nPartitions))
                kwargs["n_partitions"] = nPartitionsPy;
            return Python.pymoo.get_reference_directions("das-dennis", nObj, kwargs);
        }

        /// <summary>A <c>double[][]</c> of reference points → an independent numpy <c>(k, n_obj)</c> array
        /// in pymoo's (minimized) objective space, applying the maximize sign flip. Dispose the result.
        /// Under the GIL.</summary>
        private static PyObject BuildReferencePoints(
            double[][] refPoints, int nObj, IReadOnlyList<StudyDirection> directions)
        {
            var matrix = new double[refPoints.Length, nObj];
            for (int i = 0; i < refPoints.Length; i++)
            {
                if (refPoints[i].Length != nObj)
                    throw new NumSharp.ValueError(
                        $"Each reference point must have {nObj} objectives; point {i} has {refPoints[i].Length}.");
                for (int j = 0; j < nObj; j++)
                    matrix[i, j] = directions[j] == StudyDirection.Maximize ? -refPoints[i][j] : refPoints[i][j];
            }
            return PymooInterop.ToNumpy(matrix);
        }

        /// <summary>Every search-space parameter is present in the trial, its distribution is compatible,
        /// and its value lies inside the distribution (Optuna's <c>_is_compatible</c>). The current
        /// running trial — whose parameters are not yet set — fails the presence check, so it is excluded
        /// from the generation bookkeeping.</summary>
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

        public void Dispose()
        {
            _algo.Dispose();
            _problem.Dispose();
        }
    }
}
