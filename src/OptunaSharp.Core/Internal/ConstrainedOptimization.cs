using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>Port of <c>optuna/study/_constrained_optimization.py</c>.</summary>
internal static class ConstrainedOptimization
{
    /// <summary>The legacy system-attribute key and prefix used for persisted constraint values.</summary>
    internal const string ConstraintsKey = "constraints";

    /// <summary>Return whether the given trials are created in constrained optimization.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <returns>True when at least one trial contains constraint values.</returns>
    internal static bool IsConstrainedOptimization(IEnumerable<FrozenTrial> trials)
        => trials.Any(trial => trial.Constraints.Count > 0);

    /// <summary>Return feasible trials from given trials.</summary>
    /// <param name="trials">Trial snapshots to inspect; their order is preserved unless the operation explicitly ranks them.</param>
    /// <returns>The input trials for which every recorded constraint is nonpositive.</returns>
    internal static List<FrozenTrial> GetFeasibleTrials(IEnumerable<FrozenTrial> trials)
    {
        var feasibleTrials = new List<FrozenTrial>();
        foreach (var trial in trials)
        {
            if (trial.Constraints.Values.All(x => x <= 0.0))
                feasibleTrials.Add(trial);
        }
        return feasibleTrials;
    }

    /// <summary>
    ///     Whether <see cref="GetConstraintsFromSystemAttrs" /> would return any constraint: the old-format list key holds
    ///     a value, or some key carries the per-constraint prefix. Allocates nothing.
    /// </summary>
    /// <param name="systemAttrs">A trial's system attributes.</param>
    /// <returns>False exactly when the constraints dictionary would be empty.</returns>
    /// <remarks>
    ///     The same two tests the dictionary's construction makes before adding an entry, so a false answer means that
    ///     dictionary would be empty (an old-format list that is present but empty also reads as constraints here, and
    ///     the caller then builds the empty dictionary as before).
    /// </remarks>
    internal static bool HasConstraints(IReadOnlyDictionary<string, object?> systemAttrs)
    {
        if (systemAttrs.TryGetValue(ConstraintsKey, out var con) && con is not null)
            return true;
        string prefix = ConstraintsKey + ":";
        foreach (string key in systemAttrs.Keys)
        {
            if (key.StartsWith(prefix, System.StringComparison.Ordinal))
                return true;
        }
        return false;
    }

    /// <summary>Python <c>_get_constraints_from_system_attrs</c>.</summary>
    /// <param name="systemAttrs">The stored attributes carrying legacy and named constraint values.</param>
    /// <returns>The constraint values keyed by constraint name; named entries override legacy indexed entries.</returns>
    internal static OrderedDict<string, double> GetConstraintsFromSystemAttrs(
        IReadOnlyDictionary<string, object?> systemAttrs)
    {
        var constraintsDict = new OrderedDict<string, double>();

        // Load constraints from old format (list).
        if (systemAttrs.TryGetValue(ConstraintsKey, out var con) && con is not null)
        {
            int i = 0;
            foreach (var c in (IEnumerable<object?>)con)
            {
                constraintsDict[i.ToString(System.Globalization.CultureInfo.InvariantCulture)] = ToDouble(c);
                i++;
            }
        }

        // Load constraints from new format (individual keys).
        // Keys first, the value only for a match: the same entries in the same order as walking the pairs, but a
        // compact finished trial (plan 13, item S12) then rebuilds no stored string it does not need. TPE reads every
        // trial's constraints on every ask, and its trials carry their relative parameters as long JSON strings.
        string prefix = ConstraintsKey + ":";
        foreach (string key in systemAttrs.Keys)
        {
            if (key.StartsWith(prefix, System.StringComparison.Ordinal))
            {
                string constraintKey = key.Substring(prefix.Length);
                if (constraintsDict.ContainsKey(constraintKey))
                    OptunaWarnings.Warn("Overwrite an old format constraint.");
                constraintsDict[constraintKey] = ToDouble(systemAttrs[key]);
            }
        }
        return constraintsDict;
    }

    /// <summary>Returns the converted numeric value, or NaN when conversion is unsupported.</summary>
    /// <param name="value">The value to convert, format, or compare using the operation's stated convention.</param>
    /// <returns>The converted numeric value, or NaN when conversion is unsupported.</returns>
    private static double ToDouble(object? value)
        => PyFloat.TryConvert(value, out double d) ? d : double.NaN;
}
