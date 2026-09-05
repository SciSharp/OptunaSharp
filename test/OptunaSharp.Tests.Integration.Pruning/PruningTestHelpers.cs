using System;
using System.Collections.Generic;
using OptunaSharp.Internal;

namespace OptunaSharp.Tests.Integration.Pruning;

/// <summary>
///     Shared helpers for the named-pruning-callback tests. <see cref="CaptureWarnings" /> intercepts the
///     internal <see cref="OptunaWarnings.WarningRaised" /> event (this assembly is a declared Core
///     friend) so the Keras / tf.keras / Lightning missing-metric <c>UserWarning</c> can be asserted
///     verbatim, the way real Optuna's <c>warnings.warn</c> would be caught by <c>pytest.warns</c>.
/// </summary>
internal static class PruningTestHelpers
{
    internal static IReadOnlyList<string> CaptureWarnings(Action act)
    {
        var messages = new List<string>();
        void Handler(string message, WarningCategory category) => messages.Add(message);
        OptunaWarnings.WarningRaised += Handler;
        try
        {
            act();
        }
        finally
        {
            OptunaWarnings.WarningRaised -= Handler;
        }
        return messages;
    }

    internal static IReadOnlyDictionary<string, IReadOnlyList<double>> MetricHistory(
        params (string Metric, double[] History)[] entries)
    {
        var dict = new Dictionary<string, IReadOnlyList<double>>();
        foreach (var (metric, history) in entries)
            dict[metric] = history;
        return dict;
    }

    internal static IReadOnlyDictionary<string, IReadOnlyDictionary<string, IReadOnlyList<double>>> Metrics(
        params (string ValidName, IReadOnlyDictionary<string, IReadOnlyList<double>> ByMetric)[] entries)
    {
        var dict = new Dictionary<string, IReadOnlyDictionary<string, IReadOnlyList<double>>>();
        foreach (var (validName, byMetric) in entries)
            dict[validName] = byMetric;
        return dict;
    }
}
