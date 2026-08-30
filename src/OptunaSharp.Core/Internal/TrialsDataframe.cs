using System;
using System.Collections.Generic;
using System.Linq;
using OptunaSharp.Collections;

namespace OptunaSharp;

/// <summary>
///     A plain, dependency-free table of trials — the Core-tier stand-in for pandas in
///     <c>Study.trials_dataframe</c> (plan §2.4: the <c>Microsoft.Data.Analysis</c> adapter is a
///     Tier X package that converts from this).
/// </summary>
public sealed class TrialsTable
{
    internal TrialsTable(List<(string Column, string Nested)> columns, List<object?[]> rows, bool multiIndex)
    {
        MultiColumns = columns;
        Rows = rows;
        MultiIndex = multiIndex;
    }

    /// <summary>The two-level column labels, e.g. <c>("params", "x")</c> or <c>("number", "")</c>.</summary>
    public IReadOnlyList<(string Column, string Nested)> MultiColumns { get; }

    /// <summary>Whether the caller requested MultiIndex-style columns.</summary>
    public bool MultiIndex { get; }

    /// <summary>
    ///     Flattened column names (<c>params_x</c>) as produced by Python's
    ///     <c>_flatten_columns</c>; equals the two-level labels joined on <c>_</c> with empty
    ///     parts dropped.
    /// </summary>
    public IReadOnlyList<string> Columns
        => MultiColumns.Select(c => string.Join("_", new[] { c.Column, c.Nested }.Where(p => p.Length > 0))).ToList();

    /// <summary>Row-major cell values, one array per trial, aligned with <see cref="MultiColumns" />.</summary>
    public IReadOnlyList<object?[]> Rows { get; }

    public int RowCount => Rows.Count;

    public int ColumnCount => MultiColumns.Count;

    /// <summary>Cell accessor by row index and flattened column name.</summary>
    public object? this[int row, string flatColumn]
    {
        get
        {
            int idx = Columns.ToList().IndexOf(flatColumn);
            if (idx < 0)
                throw new NumSharp.KeyError(flatColumn);
            return Rows[row][idx];
        }
    }
}

/// <summary>Port of <c>optuna/study/_dataframe.py</c> targeting <see cref="TrialsTable" />.</summary>
internal static class TrialsDataframe
{
    internal static readonly string[] DefaultAttrs =
    {
        "number",
        "value",
        "datetime_start",
        "datetime_complete",
        "duration",
        "params",
        "user_attrs",
        "system_attrs",
        "state",
    };

    /// <summary>Python <c>_trials_dataframe(study, attrs, multi_index)</c>.</summary>
    internal static TrialsTable Build(Study study, string[] attrs, bool multiIndex)
    {
        // If no trials, return an empty table.
        if (study.GetTrials(deepcopy: false).Count == 0)
            return new TrialsTable(new List<(string, string)>(), new List<object?[]>(), multiIndex);

        if (attrs.Contains("value") && study.IsMultiObjective())
            attrs = attrs.Select(attr => attr == "value" ? "values" : attr).ToArray();

        var (records, columns) = CreateRecordsAndAggregateColumn(study, attrs);

        var rows = records
            .Select(record => columns.Select(col => record.TryGetValue(col, out var v) ? v : null).ToArray())
            .ToList();

        return new TrialsTable(columns, rows, multiIndex);
    }

    /// <summary>Python <c>_create_records_and_aggregate_column</c>.</summary>
    private static (List<Dictionary<(string, string), object?>> Records, List<(string, string)> Columns)
        CreateRecordsAndAggregateColumn(Study study, string[] attrs)
    {
        var attrsToDfColumns = new OrderedDict<string, string>();
        foreach (var attr in attrs)
        {
            // Python conventional underscores are omitted in the dataframe.
            attrsToDfColumns[attr] = attr.StartsWith("_", StringComparison.Ordinal) ? attr.Substring(1) : attr;
        }

        // columnAgg aggregates dataframe columns per FrozenTrial attribute.
        var columnAgg = new Dictionary<string, HashSet<(string, string)>>();
        const string nonNestedAttr = "";

        var metricNames = study.MetricNames;

        var records = new List<Dictionary<(string, string), object?>>();
        foreach (var trial in study.GetTrials(deepcopy: false))
        {
            var record = new Dictionary<(string, string), object?>();
            foreach (var (attr, dfColumn) in attrsToDfColumns)
            {
                HashSet<(string, string)> Agg()
                {
                    if (!columnAgg.TryGetValue(attr, out var set))
                        columnAgg[attr] = set = new HashSet<(string, string)>();
                    return set;
                }

                switch (attr)
                {
                    case "values":
                    {
                        // Expand trial.values; should be null when the trial's state is FAIL or PRUNED.
                        var trialValues = trial.Values is null
                            ? Enumerable.Repeat((object?)null, study.Directions.Count).ToList()
                            : trial.Values.Cast<object?>().ToList();
                        for (int i = 0; i < trialValues.Count; i++)
                        {
                            string nestedAttr = metricNames is null
                                ? i.ToString(System.Globalization.CultureInfo.InvariantCulture)
                                : metricNames[i];
                            record[(dfColumn, nestedAttr)] = trialValues[i];
                            Agg().Add((dfColumn, nestedAttr));
                        }
                        break;
                    }
                    case "value":
                    {
                        string nestedAttr = metricNames is null ? nonNestedAttr : metricNames[0];
                        record[(dfColumn, nestedAttr)] = trial.State == TrialState.Fail ? null : TryGetValue(trial);
                        Agg().Add((dfColumn, nestedAttr));
                        break;
                    }
                    case "params":
                    {
                        foreach (var (nestedAttr, nestedValue) in trial.ParamsDict)
                        {
                            record[(dfColumn, nestedAttr)] = nestedValue;
                            Agg().Add((dfColumn, nestedAttr));
                        }
                        break;
                    }
                    case "user_attrs":
                    {
                        foreach (var (nestedAttr, nestedValue) in trial.UserAttrsDict)
                        {
                            record[(dfColumn, nestedAttr)] = nestedValue;
                            Agg().Add((dfColumn, nestedAttr));
                        }
                        break;
                    }
                    case "system_attrs":
                    {
                        foreach (var (nestedAttr, nestedValue) in trial.SystemAttrsDict)
                        {
                            record[(dfColumn, nestedAttr)] = nestedValue;
                            Agg().Add((dfColumn, nestedAttr));
                        }
                        break;
                    }
                    case "intermediate_values":
                    {
                        foreach (var (step, nestedValue) in trial.IntermediateValues)
                        {
                            string nestedAttr = step.ToString(System.Globalization.CultureInfo.InvariantCulture);
                            record[(dfColumn, nestedAttr)] = nestedValue;
                            Agg().Add((dfColumn, nestedAttr));
                        }
                        break;
                    }
                    default:
                    {
                        record[(dfColumn, nonNestedAttr)] = GetScalarAttr(trial, attr);
                        Agg().Add((dfColumn, nonNestedAttr));
                        break;
                    }
                }
            }

            records.Add(record);
        }

        // Build column list preserving the order of `attrs` and, for multi-objective studies with
        // metric names set, the order of `metric_names` instead of alphabetical sorting (GH #6785).
        var columns = new List<(string, string)>();
        foreach (var k in attrs)
        {
            if (!columnAgg.ContainsKey(k))
                continue;
            if (k == "values" && metricNames is not null)
            {
                string dfCol = attrsToDfColumns[k];
                columns.AddRange(metricNames.Select(name => (dfCol, name)));
            }
            else
            {
                columns.AddRange(columnAgg[k]
                    .OrderBy(c => c.Item1, StringComparer.Ordinal)
                    .ThenBy(c => c.Item2, NestedComparer.Instance));
            }
        }

        return (records, columns);
    }

    private static object? TryGetValue(FrozenTrial trial)
    {
        try
        {
            return trial.Value;
        }
        catch (NumSharp.RuntimeError)
        {
            return null;
        }
    }

    private static object? GetScalarAttr(FrozenTrial trial, string attr) => attr switch
    {
        "number" => (long)trial.Number,
        "state" => trial.State.PythonName(),
        "datetime_start" => trial.DatetimeStart,
        "datetime_complete" => trial.DatetimeComplete,
        "duration" => trial.Duration,
        "_trial_id" or "trial_id" => (long)trial.TrialId,
        "last_step" => trial.LastStep is null ? null : (long)trial.LastStep.Value,
        _ => throw new NumSharp.ValueError($"Unsupported attr: {attr}"),
    };

    /// <summary>Sorts nested labels numerically when both parse as ints (Python tuple sort over int keys).</summary>
    private sealed class NestedComparer : IComparer<string>
    {
        internal static readonly NestedComparer Instance = new();

        public int Compare(string? x, string? y)
        {
            if (long.TryParse(x, out long lx) && long.TryParse(y, out long ly))
                return lx.CompareTo(ly);
            return string.CompareOrdinal(x, y);
        }
    }
}
