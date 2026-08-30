using System;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Samplers;

/// <summary>
///     Optuna's default sampler (Tree-structured Parzen Estimator).
///
///     NOT PORTED YET — arrives with PORTING_PLAN Phase 2.8. Mirroring Optuna, a study
///     constructed without an explicit sampler carries a TPESampler; construction is benign
///     (so load/copy/summaries workflows over existing studies work), but any attempt to
///     actually sample throws. Pass an explicit sampler (e.g. <see cref="RandomSampler" />)
///     to optimize in the meantime.
/// </summary>
public class TPESampler : BaseSampler
{
    private static NotImplementedException NotPorted() => new(
        "TPESampler is not ported yet (PORTING_PLAN Phase 2.8). Pass an explicit sampler " +
        "(e.g. new RandomSampler(seed)) to Optuna.CreateStudy / new Study(...).");

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
        => throw NotPorted();

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        => throw NotPorted();

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
        => throw NotPorted();
}
