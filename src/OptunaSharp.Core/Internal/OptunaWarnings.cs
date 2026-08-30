using System;
using OptunaSharp.Logging;

namespace OptunaSharp.Internal;

/// <summary>
///     Categories mirroring the Python warning classes Optuna raises through
///     <c>optuna_warn(message, category)</c>.
/// </summary>
internal enum WarningCategory
{
    /// <summary>Python <c>UserWarning</c> (the <c>optuna_warn</c> default).</summary>
    UserWarning,

    /// <summary>Python <c>RuntimeWarning</c>.</summary>
    RuntimeWarning,

    /// <summary>Python <c>FutureWarning</c>.</summary>
    FutureWarning,

    /// <summary>Python <c>DeprecationWarning</c>.</summary>
    DeprecationWarning,

    /// <summary>Optuna's own <c>optuna.exceptions.ExperimentalWarning</c>.</summary>
    ExperimentalWarning,
}

/// <summary>
///     Port of optuna/_warnings.py — the seam every Optuna <c>warnings.warn</c> call goes
///     through. .NET has no warnings module, so warnings surface on the logging seam at
///     WARNING level; tests can intercept them via <see cref="WarningRaised" />.
/// </summary>
internal static class OptunaWarnings
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna");

    /// <summary>Raised for every warning, regardless of verbosity. Used by tests (`pytest.warns` analog).</summary>
    internal static event Action<string, WarningCategory>? WarningRaised;

    /// <summary>Python: <c>optuna_warn(message, category)</c>.</summary>
    internal static void Warn(string message, WarningCategory category = WarningCategory.UserWarning)
    {
        WarningRaised?.Invoke(message, category);
        _logger.Warning(category == WarningCategory.UserWarning ? message : $"{category}: {message}");
    }
}
