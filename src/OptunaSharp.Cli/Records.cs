using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;

namespace OptunaSharp.Cli;

using Column = OutputFormat.Column;

/// <summary>
///     Port of <c>optuna/study/_dataframe.py: _create_records_and_aggregate_column</c> that the
///     <c>trials</c>/<c>best-trial</c>/<c>best-trials</c> CLI commands call. Re-ported here (over Core's
///     public <see cref="Study" />/<see cref="FrozenTrial" /> API) rather than reusing Core's internal
///     <c>TrialsDataframe</c> so the objective-<c>values</c> columns keep an <b>integer</b> second key —
///     Python's <c>_convert_to_dict</c> reconstructs a list from int indices, which the CLI's non-flatten
///     json/table output depends on (Core's table stringifies the index for its pandas-free
///     <c>trials_dataframe</c>).
/// </summary>
internal static class Records
{
    /// <summary>The attribute set the CLI trial-listing commands request.</summary>
    /// <param name="study">The study whose metadata and trial snapshots supply the output records.</param>
    /// <returns>The CLI trial-attribute names, selecting value or values according to objective count.</returns>
    internal static string[] TrialAttrs(Study study) => new[]
    {
        "number",
        study.Directions.Count > 1 ? "values" : "value",
        "datetime_start",
        "datetime_complete",
        "duration",
        "params",
        "user_attrs",
        "state",
    };

    /// <summary>Builds trial output rows and aggregates their nested column keys in display order.</summary>
    /// <param name="study">The study whose metadata and trial snapshots supply the output records.</param>
    /// <param name="attrs">The ordered trial attribute names requested by the listing command.</param>
    /// <returns>The trial row dictionaries and the ordered union of output columns.</returns>
    internal static (List<Dictionary<Column, object?>> Records, List<Column> Columns)
        CreateRecordsAndAggregateColumn(Study study, string[] attrs)
    {
        var attrsToDfColumns = new List<(string Attr, string DfColumn)>();
        foreach (var attr in attrs)
            attrsToDfColumns.Add((attr, attr.StartsWith("_", StringComparison.Ordinal) ? attr.Substring(1) : attr));

        var columnAgg = new Dictionary<string, HashSet<Column>>();
        var metricNames = study.MetricNames;
        int nDirections = study.Directions.Count;

        var records = new List<Dictionary<Column, object?>>();
        foreach (var trial in study.GetTrials(deepcopy: false))
        {
            var record = new Dictionary<Column, object?>();
            foreach (var (attr, dfColumn) in attrsToDfColumns)
            {
                HashSet<Column> Agg()
                {
                    if (!columnAgg.TryGetValue(attr, out var set))
                        columnAgg[attr] = set = new HashSet<Column>();
                    return set;
                }

                switch (attr)
                {
                    case "values":
                    {
                        // trial.values is None when the state is FAIL or PRUNED.
                        var trialValues = trial.Values is null
                            ? Enumerable.Repeat((object?)null, nDirections).ToList()
                            : trial.Values.Cast<object?>().ToList();
                        for (int i = 0; i < trialValues.Count; i++)
                        {
                            object nested = metricNames is null ? (long)i : metricNames[i];
                            var col = new Column(dfColumn, nested);
                            record[col] = trialValues[i];
                            Agg().Add(col);
                        }
                        break;
                    }
                    case "value":
                    {
                        object nested = metricNames is null ? "" : metricNames[0];
                        var col = new Column(dfColumn, nested);
                        record[col] = trial.Values is null ? null : trial.Values[0];
                        Agg().Add(col);
                        break;
                    }
                    case "params":
                    {
                        foreach (var (name, value) in trial.ParamsDict)
                        {
                            var col = new Column(dfColumn, name);
                            record[col] = value;
                            Agg().Add(col);
                        }
                        break;
                    }
                    case "user_attrs":
                    {
                        foreach (var (name, value) in trial.UserAttrsDict)
                        {
                            var col = new Column(dfColumn, name);
                            record[col] = value;
                            Agg().Add(col);
                        }
                        break;
                    }
                    default:
                    {
                        var col = new Column(dfColumn, "");
                        record[col] = GetScalarAttr(trial, attr);
                        Agg().Add(col);
                        break;
                    }
                }
            }
            records.Add(record);
        }

        var columns = new List<Column>();
        foreach (var k in attrs)
        {
            if (!columnAgg.ContainsKey(k))
                continue;
            if (k == "values" && metricNames is not null)
                columns.AddRange(metricNames.Select(name => new Column("values", name)));
            else
                columns.AddRange(columnAgg[k].OrderBy(c => c, ColumnComparer.Instance));
        }

        return (records, columns);
    }

    /// <summary>Reads a supported scalar trial field used by CLI table output.</summary>
    /// <param name="trial">The frozen trial whose scalar attribute is requested.</param>
    /// <param name="attr">A supported scalar trial attribute name.</param>
    /// <returns>The requested scalar value, or null when the trial has no corresponding timestamp or duration.</returns>
    /// <exception cref="NumSharp.ValueError">The requested attribute is not one of the supported scalar fields.</exception>
    private static object? GetScalarAttr(FrozenTrial trial, string attr) => attr switch
    {
        "number" => (long)trial.Number,
        "state" => StateName(trial.State),
        "datetime_start" => trial.DatetimeStart,
        "datetime_complete" => trial.DatetimeComplete,
        "duration" => trial.Duration,
        _ => throw new NumSharp.ValueError($"Unsupported attr: {attr}"),
    };

    /// <summary>Python <c>TrialState.name</c> (the value written into the <c>state</c> column).</summary>
    /// <param name="state">The trial lifecycle state to display.</param>
    /// <returns>The uppercase Python spelling of the trial state.</returns>
    internal static string StateName(TrialState state) => state switch
    {
        TrialState.Running => "RUNNING",
        TrialState.Complete => "COMPLETE",
        TrialState.Pruned => "PRUNED",
        TrialState.Fail => "FAIL",
        TrialState.Waiting => "WAITING",
        _ => state.ToString().ToUpperInvariant(),
    };

    /// <summary>Python <c>sorted(set of (df_column, nested))</c>: numeric second key when both are ints.</summary>
    private sealed class ColumnComparer : IComparer<Column>
    {
        /// <summary>The shared comparer ordering compound output-column keys.</summary>
        internal static readonly ColumnComparer Instance = new();

        /// <summary>Orders columns by their top-level name, then numeric or ordinal nested key.</summary>
        /// <param name="x">The first compound column key to compare.</param>
        /// <param name="y">The second compound column key to compare.</param>
        /// <returns>A negative value, zero, or positive value according to column ordering.</returns>
        public int Compare(Column x, Column y)
        {
            int c = string.CompareOrdinal(x.Name, y.Name);
            if (c != 0)
                return c;
            if (x.Nested is long lx && y.Nested is long ly)
                return lx.CompareTo(ly);
            return string.CompareOrdinal(
                Convert.ToString(x.Nested, CultureInfo.InvariantCulture),
                Convert.ToString(y.Nested, CultureInfo.InvariantCulture));
        }
    }
}
