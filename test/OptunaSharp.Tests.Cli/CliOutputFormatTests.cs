using System;
using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Cli;

namespace OptunaSharp.Tests.Cli;

using Column = OptunaSharp.Cli.OutputFormat.Column;

/// <summary>
///     Python-free unit tests for the CLI's output formatter (the port of <c>optuna/cli.py</c>'s
///     <c>_format_output</c> family). The expected strings are the byte-exact output of the
///     <c>optuna</c> command, cross-checked live against optuna 5.0.0.dev (see the parity suite).
/// </summary>
[TestClass]
public class CliOutputFormatTests
{
    private static Dictionary<Column, object?> Record(params (string Name, object Nested, object? Value)[] cells)
    {
        var record = new Dictionary<Column, object?>();
        foreach (var (name, nested, value) in cells)
            record[new Column(name, nested)] = value;
        return record;
    }

    // ------------------------------------------------------------------ studies-style output

    [TestMethod]
    public void Studies_Table_RendersTupleDirection_AndNumericAlignment()
    {
        var columns = new List<Column>
        {
            new("name", ""), new("direction", ""), new("n_trials", ""), new("datetime_start", ""),
        };
        var records = new List<Dictionary<Column, object?>>
        {
            Record(
                ("name", "", "demo"),
                ("direction", "", new PyTuple(new object?[] { "MINIMIZE" })),
                ("n_trials", "", 1L),
                ("datetime_start", "", "2020-01-01 00:00:00")),
        };

        string expected = string.Join("\n",
            "+------+---------------+----------+---------------------+",
            "| name | direction     | n_trials | datetime_start      |",
            "+------+---------------+----------+---------------------+",
            "| demo | ('MINIMIZE',) |        1 | 2020-01-01 00:00:00 |",
            "+------+---------------+----------+---------------------+");

        OutputFormat.FormatList(records, columns, "table", flatten: false).Should().Be(expected);
    }

    [TestMethod]
    public void Studies_Json_TupleBecomesArray()
    {
        var columns = new List<Column> { new("name", ""), new("direction", ""), new("n_trials", "") };
        var records = new List<Dictionary<Column, object?>>
        {
            Record(
                ("name", "", "demo"),
                ("direction", "", new PyTuple(new object?[] { "MINIMIZE", "MAXIMIZE" })),
                ("n_trials", "", 2L)),
        };

        OutputFormat.FormatList(records, columns, "json", flatten: false)
            .Should().Be("[{\"name\": \"demo\", \"direction\": [\"MINIMIZE\", \"MAXIMIZE\"], \"n_trials\": 2}]");
    }

    [TestMethod]
    public void Studies_Value_JoinsCellsWithSpace()
    {
        var columns = new List<Column> { new("name", ""), new("n_trials", "") };
        var records = new List<Dictionary<Column, object?>>
        {
            Record(("name", "", "a"), ("n_trials", "", 3L)),
            Record(("name", "", "b"), ("n_trials", "", 7L)),
        };
        OutputFormat.FormatList(records, columns, "value", flatten: false).Should().Be("a 3\nb 7");
    }

    // ------------------------------------------------------------------ multi-objective values

    [TestMethod]
    public void Trials_Json_MultiObjectiveValuesBecomeList()
    {
        // The `values` column keeps INTEGER second keys, so _convert_to_dict reconstructs a list.
        var columns = new List<Column> { new("number", ""), new("values", 0L), new("values", 1L) };
        var records = new List<Dictionary<Column, object?>>
        {
            Record(("number", "", 0L), ("values", 0L, 0.1), ("values", 1L, 0.9)),
        };
        OutputFormat.FormatList(records, columns, "json", flatten: false)
            .Should().Be("[{\"number\": 0, \"values\": [0.1, 0.9]}]");
    }

    [TestMethod]
    public void Trials_Flatten_MultiObjectiveValuesExpandByIndex()
    {
        var columns = new List<Column> { new("number", ""), new("values", 0L), new("values", 1L) };
        var records = new List<Dictionary<Column, object?>>
        {
            Record(("number", "", 0L), ("values", 0L, 0.1), ("values", 1L, 0.9)),
        };
        OutputFormat.FormatList(records, columns, "json", flatten: true)
            .Should().Be("[{\"number\": 0, \"values_0\": 0.1, \"values_1\": 0.9}]");
    }

    // ------------------------------------------------------------------ params dict + absent cells

    [TestMethod]
    public void Trials_NonFlattenJson_ParamsAreNestedDict_AbsentKeysOmitted()
    {
        // Trial A has param x; trial B has param y. Each record only carries its own param key, so the
        // nested params dict differs per row (the absent-vs-None distinction the CLI must preserve).
        var columns = new List<Column> { new("number", ""), new("params", "x"), new("params", "y") };
        var records = new List<Dictionary<Column, object?>>
        {
            Record(("number", "", 0L), ("params", "x", 1.5)),
            Record(("number", "", 1L), ("params", "y", 2.5)),
        };
        OutputFormat.FormatList(records, columns, "json", flatten: false)
            .Should().Be("[{\"number\": 0, \"params\": {\"x\": 1.5}}, {\"number\": 1, \"params\": {\"y\": 2.5}}]");
    }

    // ------------------------------------------------------------------ None cells / alignment

    [TestMethod]
    public void Table_NoneCell_RendersBlank_ColumnStaysNumericAligned()
    {
        // A numeric column with a None cell: None renders as blanks, the present number right-aligns,
        // and the None cell still contributes width 4 (Python's len(str(None)) quirk).
        var columns = new List<Column> { new("number", ""), new("value", "") };
        var records = new List<Dictionary<Column, object?>>
        {
            Record(("number", "", 0L), ("value", "", 0.5)),
            Record(("number", "", 1L), ("value", "", null)),
        };
        string expected = string.Join("\n",
            "+--------+-------+",
            "| number | value |",
            "+--------+-------+",
            "|      0 |   0.5 |",
            "|      1 |       |",
            "+--------+-------+");
        OutputFormat.FormatList(records, columns, "table", flatten: false).Should().Be(expected);
    }

    [TestMethod]
    public void Table_Empty_RendersHeaderOnly()
    {
        var columns = new List<Column> { new("name", "") };
        OutputFormat.FormatList(new List<Dictionary<Column, object?>>(), columns, "table", flatten: false)
            .Should().Be(string.Join("\n", "+------+", "| name |", "+------+", "+------+"));
    }

    // ------------------------------------------------------------------ float repr

    [TestMethod]
    public void Value_FloatUsesPythonRepr()
    {
        var columns = new List<Column> { new("v", "") };
        var records = new List<Dictionary<Column, object?>> { Record(("v", "", 0.5488135039273248)) };
        OutputFormat.FormatList(records, columns, "value", flatten: false).Should().Be("0.5488135039273248");
    }

    // ------------------------------------------------------------------ timedelta

    [DataTestMethod]
    [DataRow(0L, "0:00:00")]
    [DataRow(TimeSpan.TicksPerSecond + TimeSpan.TicksPerSecond / 2, "0:00:01.500000")]
    [DataRow(TimeSpan.TicksPerDay, "1 day, 0:00:00")]
    [DataRow(2 * TimeSpan.TicksPerDay, "2 days, 0:00:00")]
    [DataRow(TimeSpan.TicksPerHour + 2 * TimeSpan.TicksPerMinute + 3 * TimeSpan.TicksPerSecond, "1:02:03")]
    [DataRow(2461890L, "0:00:00.246189")]
    public void PyTimedelta_MatchesPythonStr(long ticks, string expected)
        => PyTimedelta.Str(TimeSpan.FromTicks(ticks)).Should().Be(expected);
}
