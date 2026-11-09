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
    /// <summary>Stores trial row values alongside the ordered column descriptors used by dataframe-style access.</summary>
    /// <param name="columns">The ordered attribute/key column descriptors of the trial table.</param>
    /// <param name="rows">One cell-value array per trial, aligned with the supplied column descriptors.</param>
    /// <param name="multiIndex">Whether the result retains separate attribute and nested-key column levels.</param>
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

    /// <summary>The number of trial records in the table.</summary>
    public int RowCount => Rows.Count;

    /// <summary>The number of attribute/key columns in the table.</summary>
    public int ColumnCount => MultiColumns.Count;

    /// <summary>Cell accessor by row index and flattened column name.</summary>
    /// <param name="row">The zero-based row index of the trial record.</param>
    /// <param name="flatColumn">The display name of a flattened attribute/key column.</param>
    /// <returns>The cell value at the requested row and column, including null for a missing attribute value.</returns>
    /// <exception cref="NumSharp.KeyError">The flattened column name does not exist.</exception>
    /// <exception cref="System.ArgumentOutOfRangeException">The row index is outside the table.</exception>
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
    /// <summary>The default trial attributes exported by dataframe-style trial inspection.</summary>
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
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="attrs">The trial attributes to include as table column groups.</param>
    /// <param name="multiIndex">Whether the result retains separate attribute and nested-key column levels.</param>
    /// <returns>The trial records and ordered column descriptors represented as a managed table.</returns>
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
    /// <param name="study">The study supplying trial history, objective directions, and storage.</param>
    /// <param name="attrs">The trial attributes to include as table column groups.</param>
    /// <returns>The row dictionaries and combined attribute/key columns in display order.</returns>
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
                        foreach (var (nestedAttr, nestedValue) in trial.Params)
                        {
                            record[(dfColumn, nestedAttr)] = nestedValue;
                            Agg().Add((dfColumn, nestedAttr));
                        }
                        break;
                    }
                    case "user_attrs":
                    {
                        foreach (var (nestedAttr, nestedValue) in trial.UserAttrs)
                        {
                            record[(dfColumn, nestedAttr)] = nestedValue;
                            Agg().Add((dfColumn, nestedAttr));
                        }
                        break;
                    }
                    case "system_attrs":
                    {
                        foreach (var (nestedAttr, nestedValue) in trial.SystemAttrs)
                        {
                            record[(dfColumn, nestedAttr)] = nestedValue;
                            Agg().Add((dfColumn, nestedAttr));
                        }
                        break;
                    }
                    case "intermediate_values":
                    {
                        foreach (var (step, nestedValue) in trial.IntermediateValuesForRead)
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

    /// <summary>Returns the requested row value, or null when that column is absent.</summary>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <returns>The requested row value, or null when that column is absent.</returns>
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

    /// <summary>Returns the managed value of the requested trial attribute.</summary>
    /// <param name="trial">The trial whose recorded state and values are used.</param>
    /// <param name="attr">The trial attribute whose scalar or mapping value is requested.</param>
    /// <returns>The managed value of the requested trial attribute.</returns>
    /// <exception cref="NumSharp.ValueError">The requested attribute name is not supported by the managed trial table.</exception>
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
        /// <summary>The shared stateless comparer used for deterministic ordering.</summary>
        internal static readonly NestedComparer Instance = new();

        /// <summary>Returns the lexicographic comparison of the two column descriptors.</summary>
        /// <param name="x">The numerical input values, in the order used by the operation.</param>
        /// <param name="y">The second numerical input, aligned with the first input where both are sequences.</param>
        /// <returns>The lexicographic comparison of the two column descriptors.</returns>
        public int Compare(string? x, string? y)
        {
            if (long.TryParse(x, out long lx) && long.TryParse(y, out long ly))
                return lx.CompareTo(ly);
            return string.CompareOrdinal(x, y);
        }
    }
}
