using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Cli;

/// <summary>
///     Port of the <c>ask</c> command's sampler construction:
///     <c>getattr(optuna.samplers, name)(**json.loads(sampler_kwargs))</c>. Resolves
///     <c>OptunaSharp.Samplers.&lt;Name&gt;</c> by reflection and binds the JSON kwargs (snake_case keys)
///     to a constructor's parameters: scalars (<c>{"seed": 0}</c>) and, as Python's <c>**kwargs</c> passes them
///     unchanged, JSON objects and arrays (<c>GridSampler</c>'s <c>search_space</c>, <c>CmaEsSampler</c>'s <c>x0</c>,
///     <c>NSGAIIISampler</c>'s <c>reference_points</c>; QA-64). Anything that cannot bind raises a clear error, as
///     Optuna's <c>TypeError</c> would.
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

    /// <summary>
    ///     Binds a JSON object to a dictionary parameter with string keys (<c>IReadOnlyDictionary</c>, <c>IDictionary</c>
    ///     or <c>Dictionary</c> of <c>string</c> to some value type), converting each value with <see cref="TryCoerce" />.
    ///     Insertion order is kept (a <see cref="Dictionary{TKey, TValue}" /> that only receives additions enumerates in
    ///     insertion order).
    /// </summary>
    /// <param name="json">The decoded JSON object.</param>
    /// <param name="target">The parameter type, nullable unwrapped.</param>
    /// <param name="result">Receives the dictionary on success, or null.</param>
    /// <returns>Whether the parameter takes a string-keyed dictionary and every value converted.</returns>
    private static bool TryCoerceMapping(OrderedDict<string, object?> json, Type target, out object? result)
    {
        result = null;
        Type? valueType = DictionaryValueType(target);
        if (valueType is null)
            return false;

        var dict = (IDictionary)Activator.CreateInstance(typeof(Dictionary<,>).MakeGenericType(typeof(string), valueType))!;
        foreach (var (key, item) in json)
        {
            if (!TryCoerce(item, valueType, out object? converted))
                return false;
            dict[key] = converted;
        }

        if (!target.IsInstanceOfType(dict))
            return false;
        result = dict;
        return true;
    }

    /// <summary>
    ///     The value type of a string-keyed dictionary type the binder can build: <c>IReadOnlyDictionary&lt;string, V&gt;</c>,
    ///     <c>IDictionary&lt;string, V&gt;</c> or <c>Dictionary&lt;string, V&gt;</c>.
    /// </summary>
    /// <param name="target">The parameter type.</param>
    /// <returns><c>V</c>, or <c>null</c> when the type is not such a dictionary.</returns>
    private static Type? DictionaryValueType(Type target)
    {
        if (!target.IsGenericType)
            return null;
        Type def = target.GetGenericTypeDefinition();
        Type[] args = target.GetGenericArguments();
        bool dictionaryShaped = def == typeof(IReadOnlyDictionary<,>) || def == typeof(IDictionary<,>)
                                || def == typeof(Dictionary<,>);
        return dictionaryShaped && args[0] == typeof(string) ? args[1] : null;
    }

    /// <summary>
    ///     Binds a JSON array to a sequence parameter: <c>T[]</c>, a list or collection interface of <c>T</c>
    ///     (<see cref="List{T}" /> is built), or a rectangular <c>double[,]</c> from an array of equally long arrays
    ///     (NSGA-III's reference points). Each element is converted with <see cref="TryCoerce" />.
    /// </summary>
    /// <param name="json">The decoded JSON array.</param>
    /// <param name="target">The parameter type, nullable unwrapped.</param>
    /// <param name="result">Receives the sequence on success, or null.</param>
    /// <returns>Whether the parameter takes such a sequence and every element converted.</returns>
    private static bool TryCoerceSequence(List<object?> json, Type target, out object? result)
    {
        result = null;
        if (target == typeof(double[,]))
        {
            // A ragged or non-numeric matrix does not bind, as np.asarray would not give a 2-D float array.
            int rows = json.Count;
            int cols = rows == 0 ? 0 : (json[0] as List<object?>)?.Count ?? -1;
            if (cols < 0)
                return false;
            var matrix = new double[rows, cols];
            for (int r = 0; r < rows; r++)
            {
                if (json[r] is not List<object?> row || row.Count != cols)
                    return false;
                for (int c = 0; c < cols; c++)
                {
                    if (!TryCoerce(row[c], typeof(double), out object? cell))
                        return false;
                    matrix[r, c] = (double)cell!;
                }
            }
            result = matrix;
            return true;
        }

        Type? elementType = SequenceElementType(target);
        if (elementType is null)
            return false;
        var items = (IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(elementType))!;
        foreach (object? item in json)
        {
            if (!TryCoerce(item, elementType, out object? converted))
                return false;
            items.Add(converted);
        }

        if (target.IsArray)
        {
            var array = Array.CreateInstance(elementType, items.Count);
            items.CopyTo(array, 0);
            result = array;
            return true;
        }
        if (!target.IsInstanceOfType(items))
            return false;
        result = items;
        return true;
    }

    /// <summary>
    ///     The element type of a one-dimensional sequence type the binder can build: <c>T[]</c>, <see cref="List{T}" />,
    ///     or <c>IList</c>, <c>IReadOnlyList</c>, <c>ICollection</c>, <c>IReadOnlyCollection</c>, <c>IEnumerable</c> of
    ///     <c>T</c>. Strings are not sequences here.
    /// </summary>
    /// <param name="target">The parameter type.</param>
    /// <returns><c>T</c>, or <c>null</c> when the type is not such a sequence.</returns>
    private static Type? SequenceElementType(Type target)
    {
        if (target.IsArray)
            return target.GetArrayRank() == 1 ? target.GetElementType() : null;
        if (!target.IsGenericType || target == typeof(string))
            return null;
        Type def = target.GetGenericTypeDefinition();
        bool listShaped = def == typeof(List<>) || def == typeof(IList<>) || def == typeof(IReadOnlyList<>)
                          || def == typeof(ICollection<>) || def == typeof(IReadOnlyCollection<>)
                          || def == typeof(IEnumerable<>);
        return listShaped ? target.GetGenericArguments()[0] : null;
    }

    /// <summary>Removes underscores and folds case to match Python keyword names to CLR parameter names.</summary>
    /// <param name="s">The constructor keyword or parameter name to normalize.</param>
    /// <returns>The lowercase name with underscores removed.</returns>
    private static string Normalize(string s) => s.Replace("_", "").ToLowerInvariant();

    /// <summary>
    ///     Converts a decoded JSON value for a reflected constructor parameter: scalars to the parameter's scalar type,
    ///     JSON objects to dictionaries and JSON arrays to lists, arrays or a rectangular <c>double[,]</c>, element by
    ///     element, and any value to <see cref="CategoricalChoice" /> where a grid takes one.
    /// </summary>
    /// <param name="value">The decoded JSON value: <c>null</c>, bool, long, double, string, <see cref="List{T}" /> of values, or <see cref="OrderedDict{TKey, TValue}" /> of string keys.</param>
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

        // A value the parameter already accepts goes through unchanged: CmaEsSampler's x0, an
        // IReadOnlyDictionary<string, object?>, receives the decoded JSON object as Optuna's receives the dict.
        if (target.IsInstanceOfType(value))
        {
            result = value;
            return true;
        }

        // Python passes JSON objects and arrays to the constructor as dicts and lists (QA-64); bind them element-wise.
        if (value is OrderedDict<string, object?> jsonObject)
            return TryCoerceMapping(jsonObject, t, out result);
        if (value is List<object?> jsonArray)
            return TryCoerceSequence(jsonArray, t, out result);
        if (t == typeof(CategoricalChoice))
        {
            // A grid point: Python equality semantics live in CategoricalChoice, so any JSON scalar is a valid choice.
            result = CategoricalChoice.From(value);
            return true;
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
