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
/// <seealso href="https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FixedTrial.html">Official Optuna fixed-trial contract.</seealso>
public class FixedTrial : BaseTrial
{
    private readonly OrderedDict<string, object?> _params;
    private readonly OrderedDict<string, object?> _suggestedParams = new();
    private readonly OrderedDict<string, BaseDistribution> _distributions = new();
    private readonly OrderedDict<string, object?> _userAttrs = new();
    private readonly OrderedDict<string, object?> _systemAttrs = new();
    private readonly DateTime? _datetimeStart = DateTime.Now;
    private readonly int _number;

    /// <param name="params">Parameter names mapped to their external values.</param>
    /// <summary>Creates an objective replay using the supplied parameter dictionary by reference.</summary>
    /// <param name="number">Trial number exposed to the objective; defaults to zero.</param>
    /// <remarks>Params contains only parameters that have been requested, not every value supplied to the constructor.</remarks>
    public FixedTrial(OrderedDict<string, object?> @params, int number = 0)
    {
        _params = @params;
        _number = number;
    }

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive float grid spacing, or null for a continuous range; cannot be combined with log.</param>
    /// <param name="log">Whether to sample in the logarithmic domain, which requires a strictly positive lower endpoint.</param>
    /// <summary>Replays a stored floating-point parameter value for the objective.</summary>
    /// <returns>The stored floating-point value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="NumSharp.TypeError">The stored or sampled value cannot be converted to a float.</exception>
    /// <remarks>Numeric values outside the requested range are returned with a warning when replaying stored parameters.</remarks>
    /// <exception cref="System.FormatException">A stepped bound or step is nonfinite and cannot be parsed as a decimal.</exception>
    /// <exception cref="System.DivideByZeroException">The positive step rounds to zero in decimal arithmetic while aligning the upper endpoint.</exception>
    public override double SuggestFloat(string name, double low, double high, double? step = null, bool log = false)
        => PyFloat.TryConvert(Suggest(name, new FloatDistribution(low, high, log: log, step: step)), out double v)
            ? v
            : throw new TypeError("float-castable value is expected.");

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="low">Inclusive lower endpoint of the parameter range.</param>
    /// <param name="high">Inclusive upper endpoint; a stepped range may lower it to the last reachable grid point.</param>
    /// <param name="step">Positive integer grid spacing; must be one for logarithmic sampling.</param>
    /// <param name="log">Whether to sample in the logarithmic domain, which requires a strictly positive lower endpoint.</param>
    /// <summary>Replays a stored integer parameter value for the objective.</summary>
    /// <returns>The stored integer value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="System.FormatException">A stored or sampled string is not an integer.</exception>
    /// <exception cref="System.OverflowException">An integer conversion exceeds Int64.</exception>
    /// <exception cref="System.InvalidCastException">The value does not support integer conversion.</exception>
    /// <remarks>Numeric values outside the requested range are returned with a warning when replaying stored parameters.</remarks>
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

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="choices">Nonempty sequence of categorical candidates; order is retained and determines stored indices.</param>
    /// <summary>Replays a stored categorical parameter value for the objective.</summary>
    /// <returns>The stored categorical value.</returns>
    /// <exception cref="NumSharp.ValueError">The distribution is invalid, a required stored parameter is missing, or a repeated suggestion changes its distribution kind, logarithmic scale, or categorical choices.</exception>
    /// <exception cref="NumSharp.TypeError">A stored or sampled value is not a supported categorical scalar.</exception>
    /// <exception cref="System.ArgumentNullException">The choices sequence is null.</exception>
    public override CategoricalChoice SuggestCategorical(string name, IEnumerable<CategoricalChoice> choices)
        => CategoricalChoice.From(Suggest(name, new CategoricalDistribution(choices)));

    /// <param name="value">Intermediate objective value; ignored by replay trials.</param>
    /// <param name="step">Zero-based reporting step; ignored by replay trials.</param>
    /// <summary>Accepts an intermediate observation without recording it; replay trials do not prune.</summary>
    public override void Report(double value, int step)
    {
    }

    /// <summary>Returns false because a replay trial does not run a pruning algorithm.</summary>
    /// <returns>Always false.</returns>
    public override bool ShouldPrune() => false;

    /// <param name="key">Attribute name; an existing attribute is overwritten.</param>
    /// <param name="value">Value retained in this replay record's local attribute dictionary.</param>
    /// <summary>Updates the local user attribute by name.</summary>
    public override void SetUserAttr(string key, object? value) => _userAttrs[key] = value;

    // FixedTrial.set_system_attr (deprecated 3.1.0) was removed in Optuna 5.0.0 (#6834); the read-only
    // SystemAttrs view stays, as it does in Optuna. Constraints still land in it via SetConstraint.

    /// <param name="name">Parameter name; repeated suggestions refer to the same parameter.</param>
    /// <param name="distribution">Distribution defining the parameter representation and allowed values.</param>
    /// <summary>Resolves a parameter through the recorded distribution and caches its external value.</summary>
    /// <returns>The external parameter value; an out-of-range fixed numeric value may be returned with a warning.</returns>
    /// <exception cref="NumSharp.ValueError">A required parameter is absent, its representation is invalid, or distributions are incompatible.</exception>
    private object? Suggest(string name, BaseDistribution distribution)
    {
        if (!_params.ContainsKey(name))
            throw new ValueError(
                $"The value of the parameter '{name}' is not found. Please set it at " +
                "the construction of the FixedTrial object.");

        // Write-boundary normalize: a boxed enum fixed value becomes its member-NAME string here, so
        // both the distribution conversion below and the recorded Params see the same string a
        // SuggestCategorical<TEnum> would — and CategoricalChoice.From (inside ToInternalRepr) never
        // meets an enum (it still rejects them, matching Optuna). Non-enum values pass through unchanged.
        var value = EnumCategorical.NormalizeParam(_params[name]);
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

    /// <summary>Gets parameters already suggested through this trial, using their external values.</summary>
    public override IReadOnlyDictionary<string, object?> Params => _suggestedParams;

    /// <summary>Gets the distribution recorded for each suggested parameter.</summary>
    public override IReadOnlyDictionary<string, BaseDistribution> Distributions => _distributions;

    /// <summary>Gets the user metadata attached to this trial.</summary>
    public override IReadOnlyDictionary<string, object?> UserAttrs => _userAttrs;

    /// <summary>
    ///     Gets the internal metadata recorded on this fixed trial (e.g. the constraint values written by
    ///     <see cref="SetConstraint" />). Still part of Optuna 5.0.0's <c>FixedTrial</c>; only the setter was
    ///     removed (#6834).
    /// </summary>
    public IReadOnlyDictionary<string, object?> SystemAttrs => _systemAttrs;

    /// <summary>Gets the local time at which this replay trial was constructed.</summary>
    public override DateTime? DatetimeStart => _datetimeStart;

    /// <summary>Gets the zero-based trial number within the study or replay context.</summary>
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
    /// <param name="key">Constraint name; later writes of the same name are ignored with a warning.</param>
    /// <param name="value">Constraint value; zero or negative is feasible and NaN is forbidden.</param>
    /// <exception cref="NumSharp.ValueError">The constraint value is NaN.</exception>
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
