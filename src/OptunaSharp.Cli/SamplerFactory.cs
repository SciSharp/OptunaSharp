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
    internal static BaseSampler Create(string name, string? kwargsJson)
    {
        Type? type = typeof(BaseSampler).Assembly.GetType("OptunaSharp.Samplers." + name, throwOnError: false);
        if (type is null || !typeof(BaseSampler).IsAssignableFrom(type) || type.IsAbstract)
            throw new NumSharp.AttributeError($"module 'optuna.samplers' has no attribute '{name}'");

        var kwargs = ParseKwargs(kwargsJson);
        return (BaseSampler)Construct(type, name, kwargs);
    }

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

    private static string Normalize(string s) => s.Replace("_", "").ToLowerInvariant();

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
