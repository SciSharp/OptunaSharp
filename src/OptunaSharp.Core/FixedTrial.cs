using System;
using System.Collections.Generic;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;

namespace OptunaSharp;

/// <summary>
///     A trial class which suggests a fixed value for each parameter.
///     Port of <c>optuna/trial/_fixed.py: FixedTrial</c>.
///
///     This object has the same methods as <see cref="Trial" />, and it suggests pre-defined
///     parameter values. The parameter values can be determined at the construction of the
///     <see cref="FixedTrial" /> object. In contrast to <see cref="Trial" />,
///     <see cref="FixedTrial" /> does not depend on <see cref="Study" />, and it is useful for
///     deploying optimization results.
/// </summary>
/// <example>
///     <code>
///     double Objective(BaseTrial trial)
///     {
///         var x = trial.SuggestFloat("x", -100, 100);
///         var y = trial.SuggestCategorical("y", new long[] { -1, 0, 1 });
///         return x * x + y;
///     }
///
///     var result = Objective(new FixedTrial(new() { ["x"] = 1.0, ["y"] = 0L }));  // == 1
///     </code>
/// </example>
public class FixedTrial : BaseTrial
{
    private readonly OrderedDict<string, object?> _params;
    private readonly OrderedDict<string, object?> _suggestedParams = new();
    private readonly OrderedDict<string, BaseDistribution> _distributions = new();
    private readonly OrderedDict<string, object?> _userAttrs = new();
    private readonly OrderedDict<string, object?> _systemAttrs = new();
    private readonly DateTime? _datetimeStart = DateTime.Now;
    private readonly int _number;

    public FixedTrial(OrderedDict<string, object?> @params, int number = 0)
    {
        _params = @params;
        _number = number;
    }

    public override double SuggestFloat(string name, double low, double high, double? step = null, bool log = false)
        => PyFloat.TryConvert(Suggest(name, new FloatDistribution(low, high, log: log, step: step)), out double v)
            ? v
            : throw new TypeError("float-castable value is expected.");

    public override long SuggestInt(string name, long low, long high, long step = 1, bool log = false)
    {
        var value = Suggest(name, new IntDistribution(low, high, log: log, step: step));
        return value switch
        {
            long l => l,
            double d => (long)d,
            bool b => b ? 1L : 0L,
            _ => Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture),
        };
    }

    public override CategoricalChoice SuggestCategorical(string name, IEnumerable<CategoricalChoice> choices)
        => CategoricalChoice.From(Suggest(name, new CategoricalDistribution(choices)));

    public override void Report(double value, int step)
    {
    }

    public override bool ShouldPrune() => false;

    public override void SetUserAttr(string key, object? value) => _userAttrs[key] = value;

    [Obsolete("Deprecated in v3.1.0. This feature will be removed in v5.0.0.")]
    public override void SetSystemAttr(string key, object? value) => _systemAttrs[key] = value;

    private object? Suggest(string name, BaseDistribution distribution)
    {
        if (!_params.ContainsKey(name))
            throw new ValueError(
                $"The value of the parameter '{name}' is not found. Please set it at " +
                "the construction of the FixedTrial object.");

        var value = _params[name];
        double paramValueInInternalRepr = distribution.ToInternalRepr(value);
        if (!distribution.Contains(paramValueInInternalRepr))
            OptunaWarnings.Warn(
                $"The value {PythonRepr.Str(value)} of the parameter '{name}' is out of " +
                $"the range of the distribution {distribution}.");

        if (_distributions.TryGetValue(name, out var existing))
            Distribution.CheckDistributionCompatibility(existing, distribution);

        _suggestedParams[name] = value;
        _distributions[name] = distribution;

        return value;
    }

    public override IReadOnlyDictionary<string, object?> Params => _suggestedParams;

    public override IReadOnlyDictionary<string, BaseDistribution> Distributions => _distributions;

    public override IReadOnlyDictionary<string, object?> UserAttrs => _userAttrs;

    public override IReadOnlyDictionary<string, object?> SystemAttrs => _systemAttrs;

    public override DateTime? DatetimeStart => _datetimeStart;

    public override int Number => _number;

    /// <summary>
    ///     Returns constraint values. The trial is considered feasible when all constraint values
    ///     are zero or less.
    /// </summary>
    public override OrderedDict<string, double> Constraints
        => ConstrainedOptimization.GetConstraintsFromSystemAttrs(_systemAttrs);

    /// <summary>
    ///     Set a constraint value for the trial. The trial is considered feasible when all
    ///     constraint values are zero or less.
    /// </summary>
    public override void SetConstraint(string key, double value)
    {
        if (double.IsNaN(value))
            throw new ValueError($"Attempted to set a constraint for {PythonRepr.StringRepr(key)}, but NaN is not allowed.");

        string constraintKey = $"{ConstrainedOptimization.ConstraintsKey}:{key}";

        if (_systemAttrs.ContainsKey(constraintKey))
        {
            // Do nothing if already set.
            OptunaWarnings.Warn(
                $"The constraint value is ignored because this constraint `key={PythonRepr.Repr(key)}` is already set.");
            return;
        }

        _systemAttrs[constraintKey] = value;
    }
}
