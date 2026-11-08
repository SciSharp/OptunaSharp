using System;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers;

/// <summary>
///     Sampler using random sampling. Port of <c>optuna/samplers/_random.py: RandomSampler</c>.
///
///     This sampler is based on <i>independent sampling</i>.
///     See also <see cref="BaseSampler" /> for more details of 'independent sampling'.
/// </summary>
/// <example>
///     <code>
///     var study = Optuna.CreateStudy(sampler: new RandomSampler());
///     study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -5, 5), 2), nTrials: 10);
///     </code>
/// </example>
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/samplers/generated/optuna.samplers.RandomSampler.html">Official Optuna RandomSampler reference.</seealso>
public class RandomSampler : BaseSampler
{
    private readonly LazyRandomState _rng;

    /// <param name="seed">Seed for random number generator.</param>
    /// <summary>Creates an independent sampler with a lazily initialized NumPy-compatible random stream.</summary>
    public RandomSampler(int? seed = null)
    {
        _rng = new LazyRandomState(seed);
    }

    /// <summary>Shared lazy NumPy-compatible random-state holder.</summary>
    internal LazyRandomState Rng => _rng;

    /// <inheritdoc />
    public override void ReseedRng()
    {
        // Python: self._rng.rng.seed(), fresh OS entropy: 128 bits through numpy's own key construction,
        // so concurrent workers never share a stream (THREADING.md §6.3).
        OsEntropySeed.Reseed(_rng.Rng);
    }

    /// <summary>
    ///     A new <see cref="RandomSampler" /> with a generator of its own, for a <see cref="SamplerConcurrency.PerWorker" />
    ///     worker. Independent draws need no shared state, so separate instances never correlate (0 duplicate samples in
    ///     the measured eight-instance runs).
    /// </summary>
    /// <returns>A fresh instance, or <see langword="null" /> for a subclass, which keeps shared access.</returns>
    internal override BaseSampler? CreateWorkerInstance()
        // The executor reseeds it from OS entropy before each trial, so the seed given here does not matter.
        => GetType() == typeof(RandomSampler) ? new RandomSampler() : null;

    /// <inheritdoc />
    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => new();

    /// <inheritdoc />
    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => new();

    /// <inheritdoc />
    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        var searchSpace = new OrderedDict<string, BaseDistribution> { [paramName] = paramDistribution };
        var trans = new SearchSpaceTransform(searchSpace);
        var bounds = trans.Bounds;
        // uniform(bounds[:, 0], bounds[:, 1]) — one MT19937 draw per bound row, in row order,
        // exactly as NumPy broadcasts.
        var transParams = _rng.Rng.uniform(bounds[":,0"], bounds[":,1"]);

        return trans.Untransform(transParams)[paramName];
    }
}
