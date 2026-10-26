// Core does not enable nullable reference types project-wide. Opting this file in keeps the `?`
// annotations below meaningful, and keeps them from raising CS8632 like an un-opted file would.
#nullable enable

using System.Reflection;

namespace OptunaSharp;

/// <summary>
///     Version metadata for OptunaSharp — the .NET port of Optuna. This mirrors Optuna's
///     <c>optuna/version.py</c> (<c>__version__</c>): the package version and the Optuna release
///     this port tracks as its behavioral source of truth.
/// </summary>
public static class OptunaSharpVersion
{
    /// <summary>
    ///     The version of the OptunaSharp assembly that is actually loaded, i.e. the NuGet package
    ///     version (<c>1.2.3</c>, <c>1.2.3-rc.1</c>) without its <c>+commit</c> build metadata. It is
    ///     the analogue of Python's <c>optuna.__version__</c>.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         Read once, at first use, from the assembly's <see cref="AssemblyInformationalVersionAttribute" />.
    ///         The build stamps that attribute from the MSBuild <c>Version</c> property, which the release
    ///         pipeline sets from the pushed <c>v*</c> tag. So this value cannot drift from the package
    ///         version the way a hand-maintained literal does.
    ///     </para>
    ///     <para>
    ///         It is deliberately a <c>static readonly</c> field, not a <c>const</c>. The compiler copies a
    ///         <c>const</c> into every consumer at the consumer's own compile time. An application built
    ///         against one OptunaSharp and run against another would then report the old number, and so
    ///         would every database the SQLite and EF Core storages create: they record
    ///         <c>"OptunaSharp " + Version</c> in <c>version_info.library_version</c>. The trade-off is that
    ///         the value cannot appear in attribute arguments, <c>switch</c> labels or other constant
    ///         expressions.
    ///     </para>
    ///     <para>
    ///         The fallbacks matter only for hand-rolled builds that disable the generated attribute
    ///         (<c>GenerateAssemblyInformationalVersionAttribute=false</c>). The value then falls back to
    ///         the numeric assembly version, and finally to <c>"0.0.0"</c>, so it is never <see langword="null" />.
    ///     </para>
    /// </remarks>
    public static readonly string Version = ReadLoadedVersion();

    /// <summary>The Optuna release this port tracks as its behavioral source of truth.</summary>
    public const string OptunaCompatibilityTarget = "5.0";

    /// <summary>
    ///     Resolves <see cref="Version" /> from the metadata of the assembly that defines this type,
    ///     which is the OptunaSharp assembly the process actually loaded.
    /// </summary>
    /// <returns>
    ///     The informational version with any <c>+build-metadata</c> suffix removed. Failing that, the
    ///     three-part assembly version; failing that, <c>"0.0.0"</c>.
    /// </returns>
    private static string ReadLoadedVersion()
    {
        Assembly assembly = typeof(OptunaSharpVersion).Assembly;

        // Since the .NET 8 SDK the informational version carries the source revision as SemVer build
        // metadata ("1.2.3+<40-hex sha>"). Python's __version__ has no such suffix, and the RDB
        // version_info row must not change from one commit to the next, so the suffix is cut at the
        // first '+'. SemVer forbids '+' anywhere else, so a prerelease label ("-rc.1") survives intact.
        string? informational = assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion;
        if (!string.IsNullOrWhiteSpace(informational))
        {
            int metadataStart = informational.IndexOf('+');
            return metadataStart >= 0 ? informational[..metadataStart] : informational;
        }

        // AssemblyVersion is four-part ("1.2.3.0"); three parts match the package version's shape.
        System.Version? numeric = assembly.GetName().Version;
        return numeric is null ? "0.0.0" : numeric.ToString(3);
    }
}
