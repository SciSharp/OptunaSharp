namespace OptunaSharp;

/// <summary>
///     Version metadata for OptunaSharp — the .NET port of Optuna. This mirrors Optuna's
///     <c>optuna/version.py</c> (<c>__version__</c>) and is deliberately the only symbol in the
///     scaffolding: no Optuna algorithms are ported yet.
/// </summary>
public static class OptunaSharpVersion
{
    /// <summary>The OptunaSharp package/assembly version.</summary>
    public const string Version = "0.1.0";

    /// <summary>The Optuna release this port tracks as its behavioral source of truth.</summary>
    public const string OptunaCompatibilityTarget = "5.0";
}
