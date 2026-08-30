using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;

namespace OptunaSharp.Internal;

/// <summary>Port of <c>optuna/study/_constrained_optimization.py</c>.</summary>
internal static class ConstrainedOptimization
{
    internal const string ConstraintsKey = "constraints";

    /// <summary>Return whether the given trials are created in constrained optimization.</summary>
    internal static bool IsConstrainedOptimization(IEnumerable<FrozenTrial> trials)
        => trials.Any(trial => trial.Constraints.Count > 0);

    /// <summary>Return feasible trials from given trials.</summary>
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

    /// <summary>Python <c>_get_constraints_from_system_attrs</c>.</summary>
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
        string prefix = ConstraintsKey + ":";
        foreach (var (key, value) in systemAttrs)
        {
            if (key.StartsWith(prefix, System.StringComparison.Ordinal))
            {
                string constraintKey = key.Substring(prefix.Length);
                if (constraintsDict.ContainsKey(constraintKey))
                    OptunaWarnings.Warn("Overwrite an old format constraint.");
                constraintsDict[constraintKey] = ToDouble(value);
            }
        }
        return constraintsDict;
    }

    private static double ToDouble(object? value)
        => PyFloat.TryConvert(value, out double d) ? d : double.NaN;
}
