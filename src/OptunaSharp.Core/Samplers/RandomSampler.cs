using System;
using System.Security.Cryptography;
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
public class RandomSampler : BaseSampler
{
    private readonly LazyRandomState _rng;

    /// <param name="seed">Seed for random number generator.</param>
    public RandomSampler(int? seed = null)
    {
        _rng = new LazyRandomState(seed);
    }

    internal LazyRandomState Rng => _rng;

    public override void ReseedRng()
    {
        // Python: self._rng.rng.seed() — reseed from OS entropy.
        Span<byte> buffer = stackalloc byte[4];
        RandomNumberGenerator.Fill(buffer);
        _rng.Rng.seed(BitConverter.ToUInt32(buffer));
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => new();

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => new();

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
