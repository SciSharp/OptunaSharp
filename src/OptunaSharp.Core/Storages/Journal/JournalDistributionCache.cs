using System;
using System.Collections.Generic;
using OptunaSharp.Distributions;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     Turns the distribution JSON a journal record or snapshot carries into a distribution object, giving every trial
///     with the same text one shared object instead of parsing the text again each time.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why (plan 10, item 2.7; the owner's choice "Share one object", 2026-11-07).</b> Every parameter record and
///         every parameter of a snapshot trial carries its distribution as JSON text, and a study repeats a handful of
///         them. Parsing each one cost 57 ms for 40,000 parameters of a 20,000-trial study (Release build); looking it
///         up by its text cost 3 ms.
///     </para>
///     <para>
///         <b>What changes.</b> Values do not: <see cref="Distribution.FromJson" /> is a pure function of the text, so a
///         shared object equals the one a new parse would build. Object identity does: two trials' distributions for the
///         same parameter may now be the same object, where Python's <c>json_to_distribution</c> builds a new one each
///         time. OptunaSharp already treats distributions as immutable and shares them, in
///         <see cref="FrozenTrial.DeepCopy" /> for one.
///     </para>
///     <para>
///         <b>What is never shared.</b> Only <see cref="FloatDistribution" />, <see cref="IntDistribution" /> and
///         <see cref="CategoricalDistribution" /> themselves are (<see cref="IsShareable" />). The deprecated types are
///         built per use as before, so their FutureWarning fires as often as it did, and
///         <see cref="DiscreteUniformDistribution" />, whose <see cref="DiscreteUniformDistribution.Q" /> has a public
///         setter, is never handed to two trials.
///     </para>
///     <para>
///         <b>Bounded.</b> At most <see cref="Capacity" /> texts are kept; a study whose bounds change endlessly simply
///         parses the texts past that, as before. Not thread-safe: each <see cref="JournalStorageReplayResult" /> owns
///         one, used under its storage's lock or during a restore no other thread sees.
///     </para>
/// </remarks>
internal sealed class JournalDistributionCache
{
    /// <summary>The most distinct texts kept; later texts are parsed per use.</summary>
    internal const int Capacity = 4096;

    private readonly Dictionary<string, BaseDistribution> _byJson = new(StringComparer.Ordinal);

    /// <summary>
    ///     The distribution for a JSON text: the shared object when the text was seen before, else a new parse, kept
    ///     for later when its type may be shared.
    /// </summary>
    /// <param name="json">The text, as <see cref="Distribution.ToJson" /> writes it.</param>
    /// <returns>A distribution equal to <c>Distribution.FromJson(json)</c>; possibly the object an earlier call returned.</returns>
    /// <exception cref="NumSharp.ValueError">The text is not a valid distribution (the same errors as <see cref="Distribution.FromJson" />; a failed parse is never cached).</exception>
    /// <exception cref="InvalidCastException">A JSON field has an incompatible runtime type.</exception>
    /// <exception cref="KeyNotFoundException">A required distribution field is absent.</exception>
    /// <exception cref="FormatException">A numeric field cannot be converted.</exception>
    /// <exception cref="OverflowException">A numeric field exceeds the supported conversion range.</exception>
    /// <exception cref="NumSharp.TypeError">A categorical choice is not a supported scalar.</exception>
    internal BaseDistribution FromJson(string json)
    {
        if (_byJson.TryGetValue(json, out BaseDistribution? shared))
            return shared;
        BaseDistribution distribution = Distribution.FromJson(json);
        if (IsShareable(distribution) && _byJson.Count < Capacity)
            _byJson[json] = distribution;
        return distribution;
    }

    /// <summary>
    ///     Whether trials may share one object of this distribution: the three current types exactly, whose properties
    ///     have no public setters, never their deprecated subclasses.
    /// </summary>
    /// <param name="distribution">The distribution.</param>
    /// <returns><c>true</c> for <see cref="FloatDistribution" />, <see cref="IntDistribution" /> and <see cref="CategoricalDistribution" /> themselves.</returns>
    internal static bool IsShareable(BaseDistribution distribution)
    {
        // Exact types: the deprecated aliases derive from these, warn when built, and DiscreteUniformDistribution's Q
        // can be set, so sharing them would change how often a warning fires or let one trial's change reach another.
        Type type = distribution.GetType();
        return type == typeof(FloatDistribution) || type == typeof(IntDistribution) || type == typeof(CategoricalDistribution);
    }
}
