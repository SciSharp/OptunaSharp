using System;
using System.Collections.Generic;
using System.Globalization;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;

namespace OptunaSharp.Distributions;

/// <summary>
///     Module-level functions of <c>optuna/distributions.py</c>: JSON round-trip
///     (<c>distribution_to_json</c> / <c>json_to_distribution</c>) and
///     <c>check_distribution_compatibility</c>. The storage layer depends on the JSON forms.
/// </summary>
public static class Distribution
{
    /// <summary>
    ///     Serialize a distribution to JSON format (Python <c>distribution_to_json</c>).
    /// </summary>
    /// <param name="dist">A distribution to be serialized.</param>
    /// <returns>A JSON string of a given distribution, byte-identical to Python's output.</returns>
    public static string ToJson(BaseDistribution dist)
    {
        var payload = new OrderedDict<string, object?>
        {
            ["name"] = dist.GetType().Name,
            ["attributes"] = dist.AsDict(),
        };
        return PyJson.Dumps(payload);
    }

    /// <summary>
    ///     Deserialize a distribution in JSON format (Python <c>json_to_distribution</c>).
    /// </summary>
    /// <param name="jsonStr">A JSON-serialized distribution.</param>
    /// <returns>A deserialized distribution.</returns>
    public static BaseDistribution FromJson(string jsonStr)
    {
        if (PyJson.Loads(jsonStr) is not OrderedDict<string, object?> jsonDict)
            throw new ValueError("Unknown distribution format.");

        if (jsonDict.ContainsKey("name"))
        {
            string name = (string)jsonDict["name"]!;
            var attributes = (OrderedDict<string, object?>)jsonDict["attributes"]!;

#pragma warning disable CS0618  // Deprecated aliases round-trip by design.
            switch (name)
            {
                case nameof(IntDistribution):
                    return new IntDistribution(
                        low: GetLong(attributes, "low"),
                        high: GetLong(attributes, "high"),
                        log: GetBool(attributes, "log", false),
                        step: GetLong(attributes, "step", 1));
                case nameof(IntLogUniformDistribution):
                    return new IntLogUniformDistribution(
                        low: GetLong(attributes, "low"),
                        high: GetLong(attributes, "high"),
                        step: GetLong(attributes, "step", 1));
                case nameof(IntUniformDistribution):
                    return new IntUniformDistribution(
                        low: GetLong(attributes, "low"),
                        high: GetLong(attributes, "high"),
                        step: GetLong(attributes, "step", 1));
                case nameof(FloatDistribution):
                    return new FloatDistribution(
                        low: GetDouble(attributes, "low"),
                        high: GetDouble(attributes, "high"),
                        log: GetBool(attributes, "log", false),
                        step: GetNullableDouble(attributes, "step"));
                case nameof(UniformDistribution):
                    return new UniformDistribution(
                        low: GetDouble(attributes, "low"),
                        high: GetDouble(attributes, "high"));
                case nameof(LogUniformDistribution):
                    return new LogUniformDistribution(
                        low: GetDouble(attributes, "low"),
                        high: GetDouble(attributes, "high"));
                case nameof(DiscreteUniformDistribution):
                    return new DiscreteUniformDistribution(
                        low: GetDouble(attributes, "low"),
                        high: GetDouble(attributes, "high"),
                        q: GetDouble(attributes, "q"));
                case nameof(CategoricalDistribution):
                    return new CategoricalDistribution(ToChoices(attributes["choices"]));
                default:
                    throw new ValueError($"Unknown distribution class: {name}");
            }
#pragma warning restore CS0618
        }
        else
        {
            // Deserialize a distribution from an abbreviated format.
            string type = (string)jsonDict["type"]!;
            if (type == "categorical")
                return new CategoricalDistribution(ToChoices(jsonDict["choices"]));
            if (type is "float" or "int")
            {
                var step = jsonDict.GetValueOrDefault("step");
                bool log = GetBool(jsonDict, "log", false);

                if (type == "float")
                    return new FloatDistribution(
                        GetDouble(jsonDict, "low"),
                        GetDouble(jsonDict, "high"),
                        log: log,
                        step: step is null ? null : ToDouble(step));

                return new IntDistribution(
                    low: GetLong(jsonDict, "low"),
                    high: GetLong(jsonDict, "high"),
                    log: log,
                    step: step is null ? 1 : ToLong(step));
            }

            throw new ValueError($"Unknown distribution type: {type}");
        }
    }

    /// <summary>
    ///     A function to check compatibility of two distributions
    ///     (Python <c>check_distribution_compatibility</c>).
    ///
    ///     It checks whether <paramref name="distOld" /> and <paramref name="distNew" /> are the
    ///     same kind of distributions. If <paramref name="distOld" /> is
    ///     <see cref="CategoricalDistribution" />, it further checks <c>choices</c> are the same
    ///     between both. Note that this method is not supposed to be called by library users.
    /// </summary>
    /// <param name="distOld">A distribution previously recorded in storage.</param>
    /// <param name="distNew">A distribution newly added to storage.</param>
    public static void CheckDistributionCompatibility(BaseDistribution distOld, BaseDistribution distNew)
    {
        if (distOld.GetType() != distNew.GetType())
            throw new ValueError("Cannot set different distribution kind to the same parameter name.");

        if (distOld is FloatDistribution floatOld && distNew is FloatDistribution floatNew)
        {
            if (floatOld.Log != floatNew.Log)
                throw new ValueError("Cannot set different log configuration to the same parameter name.");
        }
        else if (distOld is IntDistribution intOld && distNew is IntDistribution intNew)
        {
            if (intOld.Log != intNew.Log)
                throw new ValueError("Cannot set different log configuration to the same parameter name.");
        }

        if (distOld is not CategoricalDistribution)
            return;
        if (distNew is not CategoricalDistribution)
            return;
        if (!distOld.Equals(distNew))
            throw new ValueError(nameof(CategoricalDistribution) + " does not support dynamic value space.");
    }

    /// <summary>Python <c>_get_single_value(distribution)</c>.</summary>
    internal static object? GetSingleValue(BaseDistribution distribution)
    {
        System.Diagnostics.Debug.Assert(distribution.Single());

        return distribution switch
        {
            FloatDistribution f => f.Low,
            IntDistribution i => i.Low,
            CategoricalDistribution c => c.Choices[0].Value,
            _ => throw new InvalidOperationException("Should not reach."),
        };
    }

    /// <summary>
    ///     Python <c>_convert_old_distribution_to_new_distribution</c> — maps deprecated aliases
    ///     onto <see cref="FloatDistribution" /> / <see cref="IntDistribution" />.
    /// </summary>
    internal static BaseDistribution ConvertOldDistributionToNewDistribution(
        BaseDistribution distribution, bool suppressWarning = false)
    {
        BaseDistribution newDistribution;

#pragma warning disable CS0618
        switch (distribution)
        {
            // Float distributions.
            case UniformDistribution u:
                newDistribution = new FloatDistribution(u.Low, u.High, log: false, step: null);
                break;
            case LogUniformDistribution lu:
                newDistribution = new FloatDistribution(lu.Low, lu.High, log: true, step: null);
                break;
            case DiscreteUniformDistribution du:
                newDistribution = new FloatDistribution(du.Low, du.High, log: false, step: du.Q);
                break;

            // Integer distributions.
            case IntUniformDistribution iu:
                newDistribution = new IntDistribution(iu.Low, iu.High, log: false, step: iu.Step);
                break;
            case IntLogUniformDistribution ilu:
                newDistribution = new IntDistribution(ilu.Low, ilu.High, log: true, step: ilu.Step);
                break;

            // Categorical distribution (and already-new types).
            default:
                newDistribution = distribution;
                break;
        }
#pragma warning restore CS0618

        if (!newDistribution.Equals(distribution) && !suppressWarning)
        {
            OptunaWarnings.Warn(
                $"{distribution} is deprecated and internally converted to" +
                $" {newDistribution}. See https://github.com/optuna/optuna/issues/2941.",
                WarningCategory.FutureWarning);
        }

        return newDistribution;
    }

    private static IEnumerable<CategoricalChoice> ToChoices(object? value)
    {
        var list = (List<object?>)value!;
        foreach (var item in list)
            yield return CategoricalChoice.From(item);
    }

    private static double ToDouble(object? value) => value switch
    {
        double d => d,
        long l => l,
        bool b => b ? 1.0 : 0.0,
        _ => Convert.ToDouble(value, CultureInfo.InvariantCulture),
    };

    private static long ToLong(object? value) => value switch
    {
        long l => l,
        double d => (long)d,
        bool b => b ? 1L : 0L,
        _ => Convert.ToInt64(value, CultureInfo.InvariantCulture),
    };

    private static double GetDouble(OrderedDict<string, object?> dict, string key) => ToDouble(dict[key]);

    private static double? GetNullableDouble(OrderedDict<string, object?> dict, string key)
    {
        var value = dict.GetValueOrDefault(key);
        return value is null ? null : ToDouble(value);
    }

    private static long GetLong(OrderedDict<string, object?> dict, string key) => ToLong(dict[key]);

    private static long GetLong(OrderedDict<string, object?> dict, string key, long defaultValue)
    {
        var value = dict.GetValueOrDefault(key);
        return value is null ? defaultValue : ToLong(value);
    }

    private static bool GetBool(OrderedDict<string, object?> dict, string key, bool defaultValue)
    {
        var value = dict.GetValueOrDefault(key);
        return value is null ? defaultValue : (bool)value;
    }
}
