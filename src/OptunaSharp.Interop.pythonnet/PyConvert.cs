using System;
using System.Collections.Generic;
using Python.Runtime;
using OptunaSharp.Distributions;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     Marshals the small set of Optuna objects the OptunaHub bridge exchanges with Python — parameter
///     <b>values</b> (Python's <c>None</c>/<c>bool</c>/<c>int</c>/<c>float</c>/<c>str</c>, incl. the
///     numpy scalar spellings a sampler may return) and <b>distributions</b> — between CLR types and
///     live <c>optuna.distributions.*</c> objects.
///     <para>Every method must be called under the GIL (see <see cref="PythonRuntime.Run(Action)" />).</para>
/// </summary>
public static class PyConvert
{
    private static PyObject Distributions => PythonRuntime.Optuna.GetAttr("distributions");

    // ---- values ---------------------------------------------------------------------------------

    /// <summary>CLR parameter value (from a suggest call / <see cref="CategoricalChoice" />) → Python object.</summary>
    public static PyObject ToPy(object? value)
    {
        switch (value)
        {
            case null:
                return ((object?)null).ToPython(); // Python None
            case CategoricalChoice choice:
                return ToPy(choice.Value);
            case bool b:
                return b.ToPython();
            case string s:
                return s.ToPython();
            case double d:
                return d.ToPython();
            case float f:
                return ((double)f).ToPython();
            case sbyte or byte or short or ushort or int or uint or long:
                return Convert.ToInt64(value).ToPython();
            default:
                return value.ToPython();
        }
    }

    /// <summary>
    ///     Python value → CLR (<c>null</c>, <see cref="bool" />, <see cref="long" />,
    ///     <see cref="double" /> or <see cref="string" />), recognizing numpy scalar types too.
    /// </summary>
    public static object? ToClr(PyObject value)
    {
        if (value is null || value.IsNone())
            return null;

        string kind;
        using (var type = value.GetPythonType())
        using (var nameObj = type.GetAttr("__name__"))
            kind = nameObj.As<string>() ?? "";

        return kind switch
        {
            "bool" or "bool_" => value.As<bool>(),
            "int" or "int64" or "int32" or "int16" or "int8"
                or "uint64" or "uint32" or "uint16" or "uint8" or "long" => value.As<long>(),
            "float" or "float64" or "float32" or "float16" or "double" => value.As<double>(),
            "str" or "str_" => value.As<string>(),
            "NoneType" => null,
            // Fallback: try float, then string.
            _ => TryFloatElseString(value),
        };
    }

    /// <summary>Python value → a typed <see cref="CategoricalChoice" /> (Python-equality semantics apply downstream).</summary>
    public static CategoricalChoice ToChoice(PyObject value) => CategoricalChoice.From(ToClr(value));

    private static object? TryFloatElseString(PyObject value)
    {
        try { return value.As<double>(); }
        catch { return value.As<string>(); }
    }

    // ---- distributions --------------------------------------------------------------------------

    /// <summary>OptunaSharp distribution → a live <c>optuna.distributions.*</c> object.</summary>
    public static PyObject ToPy(BaseDistribution dist)
    {
        var dists = Distributions;
        switch (dist)
        {
            case FloatDistribution f:
            {
                using var ctor = dists.GetAttr("FloatDistribution");
                using var kwargs = new PyDict();
                kwargs["low"] = f.Low.ToPython();
                kwargs["high"] = f.High.ToPython();
                kwargs["log"] = f.Log.ToPython();
                kwargs["step"] = f.Step is { } s ? s.ToPython() : ((object?)null).ToPython();
                return ctor.Invoke(Array.Empty<PyObject>(), kwargs);
            }
            case IntDistribution i:
            {
                using var ctor = dists.GetAttr("IntDistribution");
                using var kwargs = new PyDict();
                kwargs["low"] = i.Low.ToPython();
                kwargs["high"] = i.High.ToPython();
                kwargs["log"] = i.Log.ToPython();
                kwargs["step"] = i.Step.ToPython();
                return ctor.Invoke(Array.Empty<PyObject>(), kwargs);
            }
            case CategoricalDistribution c:
            {
                using var ctor = dists.GetAttr("CategoricalDistribution");
                var items = new PyObject[c.Choices.Count];
                for (int k = 0; k < items.Length; k++)
                    items[k] = ToPy(c.Choices[k].Value);
                using var choices = new PyList(items);
                foreach (var it in items) it.Dispose();
                return ctor.Invoke(choices);
            }
            default:
                throw new OptunaHubInteropException(
                    $"Cannot marshal distribution of type {dist.GetType().Name} to Python (supported: " +
                    "FloatDistribution, IntDistribution, CategoricalDistribution).");
        }
    }

    /// <summary>A live <c>optuna.distributions.*</c> object → the matching OptunaSharp distribution.</summary>
    public static BaseDistribution ToClrDistribution(PyObject dist)
    {
        string name;
        using (var type = dist.GetPythonType())
        using (var nameObj = type.GetAttr("__name__"))
            name = nameObj.As<string>() ?? "";

        switch (name)
        {
            case "FloatDistribution":
            case "UniformDistribution":
            case "LogUniformDistribution":
            case "DiscreteUniformDistribution":
            {
                double low = dist.GetAttr("low").As<double>();
                double high = dist.GetAttr("high").As<double>();
                bool log = HasTrue(dist, "log");
                double? step = ReadNullableDouble(dist, "step");
                return new FloatDistribution(low, high, log, step);
            }
            case "IntDistribution":
            case "IntUniformDistribution":
            case "IntLogUniformDistribution":
            {
                long low = dist.GetAttr("low").As<long>();
                long high = dist.GetAttr("high").As<long>();
                bool log = HasTrue(dist, "log");
                long step = ReadNullableLong(dist, "step") ?? 1L;
                return new IntDistribution(low, high, log, step);
            }
            case "CategoricalDistribution":
            {
                using var choices = dist.GetAttr("choices");
                long n = choices.Length();
                var list = new List<CategoricalChoice>((int)n);
                for (long k = 0; k < n; k++)
                {
                    using var item = choices[new PyInt(k)];
                    list.Add(ToChoice(item));
                }
                return new CategoricalDistribution(list);
            }
            default:
                throw new OptunaHubInteropException(
                    $"Cannot marshal Python distribution '{name}' to OptunaSharp (supported: Float/Int/Categorical).");
        }
    }

    private static bool HasTrue(PyObject obj, string attr)
    {
        if (!obj.HasAttr(attr))
            return false;
        using var a = obj.GetAttr(attr);
        return !a.IsNone() && a.As<bool>();
    }

    private static double? ReadNullableDouble(PyObject obj, string attr)
    {
        if (!obj.HasAttr(attr))
            return null;
        using var a = obj.GetAttr(attr);
        return a.IsNone() ? null : a.As<double>();
    }

    private static long? ReadNullableLong(PyObject obj, string attr)
    {
        if (!obj.HasAttr(attr))
            return null;
        using var a = obj.GetAttr(attr);
        return a.IsNone() ? null : a.As<long>();
    }
}
