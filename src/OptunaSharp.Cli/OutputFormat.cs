using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Globalization;
using System.Linq;
using System.Text;
using OptunaSharp;
using OptunaSharp.Internal;
using YamlDotNet.Serialization;

namespace OptunaSharp.Cli;

/// <summary>
///     Python <c>tuple</c>. Kept distinct from a list so it renders as <c>('a',)</c> (Python tuple
///     repr) in the value/table formats, while still serializing to a JSON array (it is an
///     <see cref="IEnumerable" />, so <see cref="PyJson" /> writes it as one). The only CLI value that
///     is a tuple is <c>studies</c>' <c>direction</c> column.
/// </summary>
internal sealed class PyTuple : IReadOnlyList<object?>
{
    private readonly IReadOnlyList<object?> _items;

    internal PyTuple(IEnumerable<object?> items) => _items = items.ToList();

    public int Count => _items.Count;
    public object? this[int index] => _items[index];
    public IEnumerator<object?> GetEnumerator() => _items.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => _items.GetEnumerator();
}

/// <summary>
///     Faithful port of the output-formatting half of <c>optuna/cli.py</c> — <c>_format_value</c>,
///     <c>_convert_to_dict</c>, <c>_dump_value</c>, <c>_dump_table</c>, <c>CellValue</c> and
///     <c>_format_output</c>. The <c>value</c>/<c>table</c> formats render through Core's
///     <see cref="PythonRepr" /> (CPython <c>str()</c>/<c>repr()</c>), <c>json</c> through Core's
///     <see cref="PyJson" /> (<c>json.dumps</c>), so the stdout bytes match the <c>optuna</c> command.
///     <c>yaml</c> is best-effort over YamlDotNet (PyYAML's exact byte output is not a parity target).
/// </summary>
internal static class OutputFormat
{
    private const string DatetimeFormat = "yyyy-MM-dd HH:mm:ss"; // Python "%Y-%m-%d %H:%M:%S".

    // A column key: (name, nested). `nested` is "" (non-nested), a string (param / user_attr / metric
    // name), or a boxed int index (objective `values`). A record maps column keys to raw cell values.
    internal readonly record struct Column(string Name, object Nested);

    internal enum ValueType { None = 0, Numeric = 1, String = 2 }

    // ------------------------------------------------------------------ _format_value

    /// <summary>
    ///     Python <c>_format_value</c>: pass numbers/bools/None through; datetimes to the CLI datetime
    ///     string; recurse into lists/tuples/dicts; everything else (incl. <c>timedelta</c>) to
    ///     <c>str(value)</c>.
    /// </summary>
    internal static object? FormatValue(object? value)
    {
        switch (value)
        {
            case null:
                return null;
            // Python: `isinstance(value, (int, float))` — bool is an int subclass, so it passes through.
            case bool:
            case sbyte or byte or short or ushort or int or uint or long or ulong:
            case float or double or decimal:
                return value;
            case DateTime dt:
                return dt.ToString(DatetimeFormat, CultureInfo.InvariantCulture);
            case TimeSpan ts:
                // timedelta hits Python's else-branch too: `str(timedelta)`.
                return PyTimedelta.Str(ts);
            case string s:
                // Python's else-branch: `str(value)` of a str is the str itself.
                return s;
            case PyTuple tup:
                return new PyTuple(tup.Select(FormatValue));
            case IDictionary dict:
            {
                var ret = new OrderedDictionary();
                foreach (DictionaryEntry e in dict)
                    ret[FormatValue(e.Key) ?? "None"] = FormatValue(e.Value);
                return ret;
            }
            case IEnumerable seq:
                return seq.Cast<object?>().Select(FormatValue).ToList();
            default:
                // timedelta and any other type: Python `str(value)`.
                return PyStr(value);
        }
    }

    // ------------------------------------------------------------------ str()

    /// <summary>CPython <c>str(value)</c>, tuple-aware (a <see cref="PyTuple" /> renders as <c>(a,)</c>).</summary>
    internal static string PyStr(object? value)
        => value is PyTuple tup ? PythonRepr.TupleRepr(tup) : PythonRepr.Str(value);

    // ------------------------------------------------------------------ _convert_to_dict

    /// <summary>Python <c>_convert_to_dict</c>: (records, columns) -> (row dicts keyed by header, header).</summary>
    internal static (List<OrderedDictionary> Values, List<string> Header) ConvertToDict(
        List<Dictionary<Column, object?>> records, List<Column> columns, bool flatten)
    {
        var header = new List<string>();
        var ret = new List<OrderedDictionary>();

        if (flatten)
        {
            foreach (var column in columns)
            {
                if (NestedNonEmpty(column.Nested))
                {
                    header.Add($"{column.Name}_{NestedToStr(column.Nested)}");
                }
                else if (records.Any(r => r.TryGetValue(column, out var v) && IsListOrTuple(v)))
                {
                    int maxLength = records
                        .Where(r => r.ContainsKey(column))
                        .Select(r => SeqLen(r[column]))
                        .DefaultIfEmpty(0)
                        .Max();
                    for (int i = 0; i < maxLength; i++)
                        header.Add($"{column.Name}_{i}");
                }
                else
                {
                    header.Add(column.Name);
                }
            }

            foreach (var record in records)
            {
                var row = new OrderedDictionary();
                foreach (var column in columns)
                {
                    if (!record.TryGetValue(column, out var raw))
                        continue;
                    object? value = FormatValue(raw);
                    if (NestedNonEmpty(column.Nested))
                    {
                        row[$"{column.Name}_{NestedToStr(column.Nested)}"] = value;
                    }
                    else if (records.Any(r => r.TryGetValue(column, out var v) && IsListOrTuple(v)))
                    {
                        int i = 0;
                        foreach (var v in (IEnumerable)value!)
                            row[$"{column.Name}_{i++}"] = v;
                    }
                    else
                    {
                        row[column.Name] = value;
                    }
                }
                ret.Add(row);
            }
        }
        else
        {
            foreach (var column in columns)
                if (!header.Contains(column.Name))
                    header.Add(column.Name);

            foreach (var record in records)
            {
                var attrs = new OrderedDictionary();
                foreach (var name in header)
                    attrs[name] = new OrderedDictionary();

                foreach (var column in columns)
                {
                    if (!record.TryGetValue(column, out var raw))
                        continue;
                    object? value = FormatValue(raw);
                    if (column.Nested is long idx)
                    {
                        // Reconstruct a list of objective values (the second key is a list index).
                        if (attrs[column.Name] is OrderedDictionary { Count: 0 })
                            attrs[column.Name] = new List<object?>();
                        var list = (List<object?>)attrs[column.Name]!;
                        while (list.Count < idx + 1)
                            list.Add(null);
                        list[(int)idx] = value;
                    }
                    else if (NestedNonEmpty(column.Nested))
                    {
                        ((OrderedDictionary)attrs[column.Name]!)[(string)column.Nested] = value;
                    }
                    else
                    {
                        attrs[column.Name] = value;
                    }
                }
                ret.Add(attrs);
            }
        }

        return (ret, header);
    }

    private static bool NestedNonEmpty(object nested) => !(nested is string s && s.Length == 0);
    private static string NestedToStr(object nested) => Convert.ToString(nested, CultureInfo.InvariantCulture)!;

    // Python's `isinstance(v, (list, tuple))` — a dict is neither, so it must NOT be expanded by index
    // (a nested="" dict column such as `user_attrs` stays a dict in the flatten path).
    private static bool IsListOrTuple(object? v) => v is PyTuple || (v is IEnumerable && v is not string && v is not IDictionary);
    private static int SeqLen(object? v) => ((IEnumerable)v!).Cast<object?>().Count();

    // ------------------------------------------------------------------ _dump_value

    /// <summary>Python <c>_dump_value</c>: space-joined cells per row, None -> "".</summary>
    private static string DumpValue(List<OrderedDictionary> records, List<string> header)
    {
        var rows = new List<string>();
        foreach (var record in records)
        {
            var row = header.Select(cn =>
            {
                object? v = record.Contains(cn) ? record[cn] : null;
                return v is null ? "" : PyStr(v);
            });
            rows.Add(string.Join(" ", row));
        }
        return string.Join("\n", rows);
    }

    // ------------------------------------------------------------------ _dump_table

    private sealed class CellValue
    {
        internal readonly object? Value;
        internal readonly ValueType Type;

        internal CellValue(object? value)
        {
            Value = value;
            Type = value switch
            {
                null => ValueType.None,
                bool or sbyte or byte or short or ushort or int or uint or long or ulong
                    or float or double or decimal => ValueType.Numeric,
                _ => ValueType.String,
            };
        }

        internal int Width() => PyStr(Value).Length;

        internal string GetString(ValueType valueType, int width)
        {
            if (Value is null)
                return new string(' ', width);
            string value = PyStr(Value);
            return valueType == ValueType.Numeric ? value.PadLeft(width) : value.PadRight(width);
        }
    }

    /// <summary>Python <c>_dump_table</c>: the boxed, per-column-aligned ASCII table.</summary>
    private static string DumpTable(List<OrderedDictionary> records, List<string> header)
    {
        var rows = records
            .Select(record => header.Select(cn => new CellValue(record.Contains(cn) ? record[cn] : null)).ToList())
            .ToList();

        var separator = new StringBuilder("+");
        var headerString = new StringBuilder("|");
        var rowsString = rows.Select(_ => new StringBuilder("|")).ToList();

        for (int column = 0; column < header.Count; column++)
        {
            // Numeric unless any cell in the column is a string (None cells do not force string).
            var valueType = ValueType.Numeric;
            foreach (var row in rows)
                if (row[column].Type == ValueType.String)
                    valueType = ValueType.String;

            int maxWidth = rows.Count == 0
                ? header[column].Length
                : Math.Max(header[column].Length, rows.Max(row => row[column].Width()));

            separator.Append('-', maxWidth + 2).Append('+');
            headerString.Append(' ')
                .Append(valueType == ValueType.Numeric ? header[column].PadLeft(maxWidth) : header[column].PadRight(maxWidth))
                .Append(" |");
            for (int i = 0; i < rows.Count; i++)
                rowsString[i].Append(' ').Append(rows[i][column].GetString(valueType, maxWidth)).Append(" |");
        }

        var ret = new StringBuilder();
        ret.Append(separator).Append('\n');
        ret.Append(headerString).Append('\n');
        ret.Append(separator).Append('\n');
        foreach (var rowString in rowsString)
            ret.Append(rowString).Append('\n');
        ret.Append(separator).Append('\n');
        return ret.ToString();
    }

    // ------------------------------------------------------------------ _format_output

    /// <summary>Python <c>_format_output</c> for a list of records.</summary>
    internal static string FormatList(List<Dictionary<Column, object?>> records, List<Column> columns, string format, bool flatten)
        => Format(records, columns, format, flatten, single: false);

    /// <summary>Python <c>_format_output</c> for a single record (json/yaml emit the bare object).</summary>
    internal static string FormatSingle(Dictionary<Column, object?> record, List<Column> columns, string format, bool flatten)
        => Format(new List<Dictionary<Column, object?>> { record }, columns, format, flatten, single: true);

    private static string Format(List<Dictionary<Column, object?>> records, List<Column> columns, string format, bool flatten, bool single)
    {
        var (values, header) = ConvertToDict(records, columns, flatten);
        switch (format)
        {
            case "value":
                return DumpValue(values, header).Trim();
            case "table":
                return DumpTable(values, header).Trim();
            case "json":
                return (single ? PyJson.Dumps(values[0]) : PyJson.Dumps(values)).Trim();
            case "yaml":
                return DumpYaml(single ? (object)values[0] : values).Trim();
            default:
                throw new CLIUsageError($"Optuna CLI does not supported the {format} format.");
        }
    }

    // ------------------------------------------------------------------ yaml (best-effort)

    // WithQuotingNecessaryStrings(true) quotes scalars that YAML 1.1's implicit resolver (PyYAML's
    // schema) would otherwise reinterpret — timestamps, sexagesimals, bools, etc. — approximating
    // yaml.safe_dump's quoting. YAML byte-parity with PyYAML is best-effort, not a hard target.
    private static readonly ISerializer YamlSerializer =
        new SerializerBuilder().WithQuotingNecessaryStrings(quoteYaml1_1Strings: true).Build();

    private static string DumpYaml(object value) => YamlSerializer.Serialize(ToPlainForYaml(value));

    // YamlDotNet serializes generic collections; convert the OrderedDictionary/list/tuple tree to
    // Dictionary/List (keys sorted, approximating PyYAML's sort_keys=True default).
    private static object? ToPlainForYaml(object? value)
    {
        switch (value)
        {
            case OrderedDictionary od:
            {
                var dict = new Dictionary<string, object?>();
                foreach (DictionaryEntry e in od)
                    dict[Convert.ToString(e.Key, CultureInfo.InvariantCulture)!] = ToPlainForYaml(e.Value);
                return dict.OrderBy(kv => kv.Key, StringComparer.Ordinal).ToDictionary(kv => kv.Key, kv => kv.Value);
            }
            case string s:
                return s;
            case PyTuple tup:
                return tup.Select(ToPlainForYaml).ToList();
            case IEnumerable seq:
                return seq.Cast<object?>().Select(ToPlainForYaml).ToList();
            default:
                return value;
        }
    }
}
