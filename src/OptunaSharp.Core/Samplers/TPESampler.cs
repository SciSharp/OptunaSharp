using System;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Samplers;

/// <summary>
///     Optuna's default sampler (Tree-structured Parzen Estimator).
///
///     NOT PORTED YET — arrives with PORTING_PLAN Phase 2.8. Until then, constructing it (or a
///     study without an explicit sampler, since <see cref="Study" /> defaults to TPE like Optuna
///     does) throws. Pass an explicit sampler, e.g. <see cref="RandomSampler" />.
/// </summary>
public class TPESampler : BaseSampler
{
    public TPESampler()
    {
        throw new NotImplementedException(
            "TPESampler is not ported yet (PORTING_PLAN Phase 2.8). Pass an explicit sampler " +
            "(e.g. new RandomSampler(seed)) to Optuna.CreateStudy / new Study(...).");
    }

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => throw new NotImplementedException();

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => throw new NotImplementedException();

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => throw new NotImplementedException();
}
