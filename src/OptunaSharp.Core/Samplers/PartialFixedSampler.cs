using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp.Samplers;

/// <summary>
///     Sampler with partially fixed parameters.
///     Port of <c>optuna/samplers/_partial_fixed.py: PartialFixedSampler</c> (experimental since
///     2.4.0).
///
///     <para>After several steps of optimization you can fix the value of some parameters and
///     re-optimize the rest, delegating the unfixed parameters to a wrapped base sampler:</para>
///     <code>
///     var study = Optuna.CreateStudy();
///     study.Optimize(Objective, nTrials: 10);
///
///     var fixedParams = new Dictionary&lt;string, object?&gt; { ["y"] = study.BestParams["y"] };
///     study.Sampler = new PartialFixedSampler(fixedParams, study.Sampler);
///     study.Optimize(Objective, nTrials: 10);
///     </code>
/// </summary>
public class PartialFixedSampler : BaseSampler
{
    private readonly IReadOnlyDictionary<string, object?> _fixedParams;
    private readonly BaseSampler _baseSampler;

    /// <param name="fixedParams">A dictionary of parameters to be fixed.</param>
    /// <param name="baseSampler">A sampler which samples unfixed parameters.</param>
    public PartialFixedSampler(IReadOnlyDictionary<string, object?> fixedParams, BaseSampler baseSampler)
    {
        _fixedParams = fixedParams;
        _baseSampler = baseSampler;
    }

    public override void ReseedRng() => _baseSampler.ReseedRng();

    public override OrderedDict<string, BaseDistribution> InferRelativeSearchSpace(Study study, FrozenTrial trial)
    {
        var searchSpace = _baseSampler.InferRelativeSearchSpace(study, trial);

        // Remove fixed params from relative search space to return fixed values.
        foreach (var paramName in _fixedParams.Keys)
        {
            if (searchSpace.ContainsKey(paramName))
                searchSpace.Remove(paramName);
        }

        return searchSpace;
    }

    public override OrderedDict<string, object?> SampleRelative(
        Study study, FrozenTrial trial, OrderedDict<string, BaseDistribution> searchSpace)
        // Fixed params are never sampled here.
        => _baseSampler.SampleRelative(study, trial, searchSpace);

    public override object? SampleIndependent(
        Study study, FrozenTrial trial, string paramName, BaseDistribution paramDistribution)
    {
        if (!_fixedParams.ContainsKey(paramName))
        {
            // Unfixed params are sampled here.
            return _baseSampler.SampleIndependent(study, trial, paramName, paramDistribution);
        }

        // Fixed params are sampled here.
        // Check if a parameter value is contained in the range of this distribution.
        var paramValue = _fixedParams[paramName];

        double paramValueInInternalRepr = paramDistribution.ToInternalRepr(paramValue);
        bool contained = paramDistribution.Contains(paramValueInInternalRepr);

        if (!contained)
        {
            OptunaWarnings.Warn(
                $"Fixed parameter '{paramName}' with value {PythonRepr.Str(paramValue)} is out of range " +
                $"for distribution {paramDistribution}.");
        }
        return paramValue;
    }

    public override void BeforeTrial(Study study, FrozenTrial trial) => _baseSampler.BeforeTrial(study, trial);

    public override void AfterTrial(Study study, FrozenTrial trial, TrialState state, IReadOnlyList<double>? values)
        => _baseSampler.AfterTrial(study, trial, state, values);
}
