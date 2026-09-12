using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Cli;

/// <summary>
///     Port of the <c>ask</c> command's sampler construction:
///     <c>getattr(optuna.samplers, name)(**json.loads(sampler_kwargs))</c>. Resolves
///     <c>OptunaSharp.Samplers.&lt;Name&gt;</c> by reflection and binds the JSON kwargs (snake_case keys)
///     to a constructor's parameters. Scalar kwargs (the realistic CLI case, e.g. <c>{"seed": 0}</c>)
///     are supported; anything that cannot bind raises a clear error, as Optuna's <c>TypeError</c> would.
/// </summary>
internal static class SamplerFactory
{
    /// <summary>Resolves a built-in sampler type and invokes a constructor bound from JSON keyword arguments.</summary>
    /// <param name="name">The unqualified built-in sampler class name.</param>
    /// <param name="kwargsJson">An optional JSON object containing sampler constructor arguments.</param>
    /// <returns>The constructed built-in sampler instance.</returns>
    /// <exception cref="NumSharp.AttributeError">No concrete built-in sampler has the requested name.</exception>
    /// <exception cref="NumSharp.ValueError">The keyword JSON is not an object or no constructor can bind all supplied arguments.</exception>
    /// <exception cref="System.Reflection.TargetInvocationException">The selected sampler constructor throws; its cause is preserved as the inner exception.</exception>
    internal static BaseSampler Create(string name, string? kwargsJson)
    {
        Type? type = typeof(BaseSampler).Assembly.GetType("OptunaSharp.Samplers." + name, throwOnError: false);
        if (type is null || !typeof(BaseSampler).IsAssignableFrom(type) || type.IsAbstract)
            throw new NumSharp.AttributeError($"module 'optuna.samplers' has no attribute '{name}'");

        var kwargs = ParseKwargs(kwargsJson);
        return (BaseSampler)Construct(type, name, kwargs);
    }

    /// <summary>Decodes the optional JSON object containing sampler constructor arguments.</summary>
    /// <param name="kwargsJson">An optional JSON object containing sampler constructor arguments.</param>
    /// <returns>The decoded keyword dictionary, or an empty dictionary when omitted.</returns>
    /// <exception cref="NumSharp.ValueError">The supplied JSON is malformed or is not an object.</exception>
    private static Dictionary<string, object?> ParseKwargs(string? kwargsJson)
    {
        var result = new Dictionary<string, object?>();
        if (string.IsNullOrEmpty(kwargsJson))
            return result;
        if (PyJson.Loads(kwargsJson) is not OrderedDict<string, object?> dict)
            throw new NumSharp.ValueError("`--sampler-kwargs` must be a JSON object.");
        foreach (var (k, v) in dict)
            result[k] = v;
        return result;
    }

    /// <summary>Selects the widest public constructor that consumes every supplied keyword and can bind every parameter.</summary>
    /// <param name="type">The concrete sampler type whose public constructors are considered.</param>
    /// <param name="name">The unqualified built-in sampler class name.</param>
    /// <param name="kwargs">Decoded constructor arguments, using Python snake_case or CLR parameter names.</param>
    /// <returns>The sampler instance returned by the matching constructor.</returns>
    /// <exception cref="NumSharp.ValueError">No public constructor can bind all supplied arguments.</exception>
    /// <exception cref="System.Reflection.TargetInvocationException">The selected sampler constructor throws.</exception>
    private static object Construct(Type type, string name, Dictionary<string, object?> kwargs)
    {
        // Try constructors widest-first; take the first that binds every parameter AND consumes every
        // provided kwarg (so an unknown kwarg is an error, like Python's `unexpected keyword argument`).
        foreach (var ctor in type.GetConstructors().OrderByDescending(c => c.GetParameters().Length))
        {
            var parameters = ctor.GetParameters();
            var args = new object?[parameters.Length];
            var used = new HashSet<string>();
            bool ok = true;

            for (int i = 0; i < parameters.Length; i++)
            {
                var p = parameters[i];
                string? key = kwargs.Keys.FirstOrDefault(k => Normalize(k) == Normalize(p.Name!));
                if (key is not null)
                {
                    if (!TryCoerce(kwargs[key], p.ParameterType, out args[i]))
                    {
                        ok = false;
                        break;
                    }
                    used.Add(key);
                }
                else if (p.HasDefaultValue)
                {
                    args[i] = p.DefaultValue;
                }
                else
                {
                    ok = false;
                    break;
                }
            }

            if (ok && used.Count == kwargs.Count)
                return ctor.Invoke(args)!;
        }

        throw new NumSharp.ValueError(
            $"Could not construct sampler '{name}' from the given --sampler-kwargs "
            + $"({{{string.Join(", ", kwargs.Keys)}}}).");
    }

    /// <summary>Removes underscores and folds case to match Python keyword names to CLR parameter names.</summary>
    /// <param name="s">The constructor keyword or parameter name to normalize.</param>
    /// <returns>The lowercase name with underscores removed.</returns>
    private static string Normalize(string s) => s.Replace("_", "").ToLowerInvariant();

    /// <summary>Attempts a supported scalar conversion for a reflected constructor parameter.</summary>
    /// <param name="value">The scalar or composite value to parse, classify, or render.</param>
    /// <param name="target">The requested CLR parameter type.</param>
    /// <param name="result">Receives the converted value on success, or null when conversion fails.</param>
    /// <returns>Whether conversion succeeded and produced a value acceptable to the target parameter.</returns>
    private static bool TryCoerce(object? value, Type target, out object? result)
    {
        Type t = Nullable.GetUnderlyingType(target) ?? target;

        if (value is null)
        {
            result = null;
            return !target.IsValueType || Nullable.GetUnderlyingType(target) is not null;
        }

        try
        {
            if (t == typeof(bool)) { result = Convert.ToBoolean(value, CultureInfo.InvariantCulture); return value is bool; }
            if (t == typeof(string)) { result = value as string; return value is string; }
            if (t == typeof(int)) { result = Convert.ToInt32(value, CultureInfo.InvariantCulture); return value is long or int; }
            if (t == typeof(long)) { result = Convert.ToInt64(value, CultureInfo.InvariantCulture); return value is long or int; }
            if (t == typeof(double)) { result = Convert.ToDouble(value, CultureInfo.InvariantCulture); return value is long or int or double; }
            if (t == typeof(float)) { result = Convert.ToSingle(value, CultureInfo.InvariantCulture); return value is long or int or double; }
            if (t.IsEnum)
            {
                result = value is string es ? Enum.Parse(t, es, ignoreCase: true) : Enum.ToObject(t, value);
                return true;
            }
            result = Convert.ChangeType(value, t, CultureInfo.InvariantCulture);
            return true;
        }
        catch
        {
            result = null;
            return false;
        }
    }
}
