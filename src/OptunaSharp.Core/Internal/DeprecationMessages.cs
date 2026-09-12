namespace OptunaSharp.Internal;

/// <summary>
///     Verbatim runtime-deprecation messages, byte-identical to what Optuna's
///     <c>@deprecated_class</c> / <c>@deprecated_func</c> decorators emit through
///     <c>warnings.warn(message, FutureWarning)</c> (Optuna's
///     <c>_DEPRECATION_WARNING_TEMPLATE</c> in <c>optuna/_deprecated.py</c>, optionally with the
///     decorator's extra <c>text</c> appended after a single space).
///
///     Each constant is used in two places for a given deprecated member: as the
///     <see cref="System.ObsoleteAttribute" /> message (a compile-time warning, never an error) and
///     as the argument to <see cref="OptunaWarnings.Warn(string, WarningCategory)" /> with
///     <see cref="WarningCategory.FutureWarning" /> in the member body (the runtime warning Optuna
///     raises). Both are single-sourced here so the two never drift.
///
///     The message uses Optuna's Python member names (e.g. <c>suggest_uniform</c>,
///     <c>system_attrs</c>) and its raw reStructuredText role markup
///     (<c>:class:`~optuna.distributions.FloatDistribution`</c>) exactly as Optuna emits them, so an
///     interop parity test that captures the warning matches byte-for-byte.
/// </summary>
internal static class DeprecationMessages
{
    // The shared body of Optuna's `_DEPRECATION_WARNING_TEMPLATE` for each (deprecated, removed)
    // version pair; the member name is prepended and any extra `text` is appended after one space.
    private const string DeprecatedV30V60 =
        " has been deprecated in v3.0.0. This feature will be removed in v6.0.0. " +
        "See https://github.com/optuna/optuna/releases/tag/v3.0.0.";

    private const string DeprecatedV31V50 =
        " has been deprecated in v3.1.0. This feature will be removed in v5.0.0. " +
        "See https://github.com/optuna/optuna/releases/tag/v3.1.0.";

    // Distributions (deprecated_class, v3.0.0 -> v6.0.0).
    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>UniformDistribution</c>.</summary>
    internal const string UniformDistribution =
        "UniformDistribution" + DeprecatedV30V60 + " Use :class:`~optuna.distributions.FloatDistribution` instead.";

    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>LogUniformDistribution</c>.</summary>
    internal const string LogUniformDistribution =
        "LogUniformDistribution" + DeprecatedV30V60 + " Use :class:`~optuna.distributions.FloatDistribution` instead.";

    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>DiscreteUniformDistribution</c>.</summary>
    internal const string DiscreteUniformDistribution =
        "DiscreteUniformDistribution" + DeprecatedV30V60 + " Use :class:`~optuna.distributions.FloatDistribution` instead.";

    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>IntUniformDistribution</c>.</summary>
    internal const string IntUniformDistribution =
        "IntUniformDistribution" + DeprecatedV30V60 + " Use :class:`~optuna.distributions.IntDistribution` instead.";

    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>IntLogUniformDistribution</c>.</summary>
    internal const string IntLogUniformDistribution =
        "IntLogUniformDistribution" + DeprecatedV30V60 + " Use :class:`~optuna.distributions.IntDistribution` instead.";

    // Trial suggest_* aliases (deprecated_func, v3.0.0 -> v6.0.0).
    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>SuggestUniform</c>.</summary>
    internal const string SuggestUniform =
        "suggest_uniform" + DeprecatedV30V60 + " Use suggest_float instead.";

    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>SuggestLogUniform</c>.</summary>
    internal const string SuggestLogUniform =
        "suggest_loguniform" + DeprecatedV30V60 + " Use suggest_float(..., log=True) instead.";

    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>SuggestDiscreteUniform</c>.</summary>
    internal const string SuggestDiscreteUniform =
        "suggest_discrete_uniform" + DeprecatedV30V60 + " Use suggest_float(..., step=...) instead.";

    // system-attrs members (deprecated_func, v3.1.0 -> v5.0.0, no extra text).
    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>SetSystemAttr</c>.</summary>
    internal const string SetSystemAttr = "set_system_attr" + DeprecatedV31V50;

    /// <summary>The shared compile-time and runtime Optuna deprecation text for <c>SystemAttrs</c>.</summary>
    internal const string SystemAttrs = "system_attrs" + DeprecatedV31V50;
}
