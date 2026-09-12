using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Samplers;

/// <summary>
///     A Quasi Monte Carlo sampler that generates low-discrepancy sequences. Port of
///     <c>optuna/samplers/_qmc.py: QMCSampler</c> (experimental since Optuna 3.0.0).
///
///     <para>Optuna's sampler wraps <c>scipy.stats.qmc</c>. OptunaSharp reproduces the non-scrambled
///     Sobol' and Halton sequences bit-for-bit through <see cref="QuasiMonteCarlo" />. The search
///     space is fixed by the first trial of the study; the first trial (and any parameter outside the
///     relative search space) is drawn by the <c>independent_sampler</c> (a <see cref="RandomSampler" />
///     by default, so those draws stay on MT19937).</para>
///
///     <para><b>Scrambling (PORTING_PLAN Phase 2.12 — complete).</b> Both randomizations are ported
///     bit-exact vs scipy, seeded by <c>numpy.random.default_rng(seed)</c>: scrambled Sobol' through
///     <see cref="Internal.ScrambledSobol" /> (the linear-matrix-scramble + digital shift) and scrambled
///     Halton through <see cref="Internal.ScrambledHalton" /> (Owen's digit permutations — scipy's
///     <c>_van_der_corput_permutations</c> over NumSharp's <c>Generator.shuffle</c>). Every
///     <c>(qmc_type, scramble)</c> combination is supported.</para>
/// </summary>
public class QMCSampler : BaseSampler
{
    private static readonly TrialState[] SuggestedStates = { TrialState.Complete, TrialState.Pruned };
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.samplers._qmc");

    private readonly bool _scramble;
    private readonly long _seed;
    private readonly BaseSampler _independentSampler;
    private readonly bool _warnIndependentSampling;
    private readonly string _qmcType;

    /// <param name="qmcType">The QMC sequence: <c>"sobol"</c> (default) or <c>"halton"</c>.</param>
    /// <param name="scramble">Apply scrambling (randomization) to the QMC sequence; seeded by <paramref name="seed" />.</param>
    /// <param name="seed">Seed for the independent sampler and, when <paramref name="scramble" /> is set, the QMC scrambling.</param>
    /// <param name="independentSampler">Sampler for the first trial / non-relative params. Defaults to <see cref="RandomSampler" />.</param>
    /// <param name="warnAsynchronousSeeding">Warn when scrambling is on without a manual seed.</param>
    /// <param name="warnIndependentSampling">Warn when a parameter falls back to independent sampling.</param>
    public QMCSampler(
        string qmcType = "sobol",
        bool scramble = false,
        int? seed = null,
        BaseSampler? independentSampler = null,
        bool warnAsynchronousSeeding = true,
        bool warnIndependentSampling = true)
    {
        // Optuna decorates QMCSampler with @experimental_class("3.0.0"), which raises an
        // ExperimentalWarning on construction. OptunaSharp targets the Optuna 5.x surface and treats
        // this sampler as stable, so that warning is intentionally omitted (deliberate divergence).

        if (qmcType is "halton" or "sobol")
            _qmcType = qmcType;
        else
            throw new ValueError($"The `qmc_type='{qmcType}'` is invalid. Choose either `halton` or `sobol`.");

        _scramble = scramble;
        // Optuna sets `_seed = np.random.PCG64().random_raw() if seed is None else seed` — a value
        // fixed for this sampler instance (so its scrambling is stable across trials) but random across
        // processes when unseeded. Non-scrambled sequences never consult the seed.
        if (seed is not null)
            _seed = seed.Value;
        else if (scramble)
            // Non-reproducible by design (Optuna warns via _log_asynchronous_seeding below). Drawn from
            // OS entropy through NumSharp's PCG64 (`default_rng()`) — never System.Random.
            _seed = (int)np.random.default_rng().integers(int.MaxValue);
        else
            _seed = 0;  // unused when scramble=False (both the hash and the engine ignore it).
        _independentSampler = independentSampler ?? new RandomSampler(seed);
        _warnIndependentSampling = warnIndependentSampling;

        if (seed is null && scramble && warnAsynchronousSeeding)
            LogAsynchronousSeeding();
    }

    public override void ReseedRng() => _independentSampler.ReseedRng();

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var pastTrials = study.GetTrialsInternal(deepcopy: false, states: SuggestedStates, useCache: true);
        // The initial trial is sampled by the independent sampler.
        if (pastTrials.Count == 0)
        {
            var pendingTrials = study.GetTrialsInternal(
                deepcopy: false, states: new[] { TrialState.Running }, useCache: true);
            if (pendingTrials.Count == 0)
                return new OrderedDict<string, BaseDistribution>();

            var unionSearchSpace = new OrderedDict<string, BaseDistribution>();
            HashSet<string>? intersectionKeys = null;
            foreach (var t in pendingTrials)
            {
                var space = t.DistributionsDict;
                foreach (var (name, dist) in space)
                    unionSearchSpace[name] = dist;
                if (intersectionKeys is null)
                    intersectionKeys = new HashSet<string>(space.Keys);
                else if (space.Count > 0)
                    intersectionKeys.IntersectWith(space.Keys);
            }

            if (intersectionKeys is not null && !intersectionKeys.SetEquals(unionSearchSpace.Keys))
                _logger.Warning(
                    "`QMCSampler` assumes that the search space does not include any conditions. " +
                    "Please make sure the provided search space is non-conditional.");
            return unionSearchSpace;
        }

        // If an initial trial was already made, construct the search space from the first trial.
        var firstTrial = pastTrials.OrderBy(t => t.Number).First();
        var result = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, dist) in firstTrial.DistributionsDict)
            result[name] = dist;
        return result;
    }

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        if (study.GetTrialsInternal(deepcopy: false, states: SuggestedStates, useCache: true).Count > 0)
        {
            if (_warnIndependentSampling)
                LogIndependentSampling(trial, paramName);
        }

        return _independentSampler.SampleIndependent(study, trial, paramName, paramDistribution);
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
    {
        if (searchSpace.Count == 0)
            return new OrderedDict<string, object?>();

        // Map each categorical parameter to FloatDistribution(0, C): it takes one QMC coordinate in
        // [0, C), and int(value) (a floor, since the engines emit values in [0, 1)) gives a uniform
        // choice index in [0, C - 1]. IntDistribution would round to nearest and bias the boundaries.
        var categoricalSpace = new OrderedDict<string, CategoricalDistribution>();
        var combined = new OrderedDict<string, BaseDistribution>();
        foreach (var (name, dist) in searchSpace)
        {
            if (dist is CategoricalDistribution cat)
            {
                categoricalSpace[name] = cat;
                combined[name] = new FloatDistribution(0, cat.Choices.Count);
            }
            else
            {
                combined[name] = dist;
            }
        }

        double[] qmcPoint = SampleQmc(study, searchSpace);
        var trans = new SearchSpaceTransform(combined, transform01: true);
        var sample = trans.Untransform(np.array(qmcPoint));

        var result = new OrderedDict<string, object?>();
        foreach (var (name, value) in sample)
        {
            if (categoricalSpace.TryGetValue(name, out var cat))
                result[name] = cat.ToExternalRepr((long)(double)value!);  // int(value): truncate.
            else
                result[name] = value;
        }
        return result;
    }

    public override void BeforeTrial(Study study, FrozenTrial trial)
        => _independentSampler.BeforeTrial(study, trial);

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => _independentSampler.AfterTrial(study, trial, state, values);

    /// <summary>Python <c>_sample_qmc</c>: the QMC point at this trial's <c>sample_id</c>.</summary>
    private double[] SampleQmc(Study study, OrderedDict<string, BaseDistribution> searchSpace)
    {
        int sampleId = FindSampleId(study, searchSpace);  // `sample_id` starts from 0.
        int d = searchSpace.Count;
        // Optuna does `fast_forward(sample_id)` then `random(1)`, i.e. the point at index sample_id.
        if (_qmcType == "sobol")
        {
            if (!_scramble)
                return QuasiMonteCarlo.SobolPoint(d, (ulong)sampleId);
            // Scrambled Sobol': `Sobol(d, seed, scramble=True).fast_forward(sample_id).random(1)` is the
            // point at index sample_id of the scrambled sequence (verified bit-exact vs scipy). The
            // engine is stateless here — reconstructed per call from (d, seed) — so parallel trials need
            // no lock (the sample_id counter itself is already serialized through the storage attr).
            double[,] pts = ScrambledSobol.Random(d, sampleId + 1, (int)_seed);
            var row = new double[d];
            for (int j = 0; j < d; j++)
                row[j] = pts[sampleId, j];
            return row;
        }
        // Halton.
        if (!_scramble)
            return QuasiMonteCarlo.HaltonPoint(d, (ulong)sampleId);
        // Scrambled Halton: `Halton(d, seed, scramble=True).fast_forward(sample_id).random(1)` is the
        // Owen-permuted point at index sample_id (verified bit-exact vs scipy). Like scrambled Sobol' the
        // engine is stateless — rebuilt per call from (d, seed) — and permutation init is its only rng draw.
        return ScrambledHalton.Point(d, (ulong)sampleId, (int)_seed);
    }

    /// <summary>Python <c>_find_sample_id</c>: a per-(engine, search space[, seed]) counter in the study attrs.</summary>
    private int FindSampleId(Study study, OrderedDict<string, BaseDistribution> searchSpace)
    {
        // Sort by keys to make the order static.
        var searchSpaceStr = new OrderedDict<string, object?>();
        foreach (var name in searchSpace.Keys.OrderBy(k => k, StringComparer.Ordinal))
            searchSpaceStr[name] = searchSpace[name].ToString();

        var qmcVars = new OrderedDict<string, object?>
        {
            ["qmc_type"] = _qmcType,
            ["search_space"] = searchSpaceStr,
        };
        // Sobol/Halton sequences without scrambling do not use the seed.
        if (_scramble)
        {
            qmcVars["scramble"] = true;
            qmcVars["seed"] = _seed;
        }
        else
        {
            qmcVars["scramble"] = false;
        }

        byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(PyJson.Dumps(qmcVars)));
        string keyQmcId = "qmc:" + Convert.ToHexString(hash).ToLowerInvariant();

        var studyAttrs = study.Storage.GetStudySystemAttrs(study.StudyId);
        int sampleId = (studyAttrs.TryGetValue(keyQmcId, out var v) && v is not null
            ? Convert.ToInt32(v)
            : -1) + 1;
        study.Storage.SetStudySystemAttr(study.StudyId, keyQmcId, sampleId);
        return sampleId;
    }

    private static void LogAsynchronousSeeding()
        => _logger.Warning(
            "No seed is provided for `QMCSampler` and the seed is set randomly. " +
            "If you are running multiple `QMCSampler`s in parallel and/or distributed " +
            " environment, the same seed must be used in all samplers to ensure that resulting " +
            "samples are taken from the same QMC sequence. ");

    private void LogIndependentSampling(FrozenTrial trial, string paramName)
        => _logger.Warning(string.Format(
            IndependentSamplingWarningTemplate,
            paramName,
            trial.Number,
            _independentSampler.GetType().Name,
            GetType().Name,
            "dynamic search space is not supported by `QMCSampler`"));
}
