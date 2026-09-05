using System.Collections.Generic;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp;
using OptunaSharp.Cli;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Cli;

using Column = OptunaSharp.Cli.OutputFormat.Column;

/// <summary>
///     Unit tests for the CLI's records/columns builder (<see cref="Records" />, the port of
///     <c>_create_records_and_aggregate_column</c>) over in-process studies — no Python. The load-bearing
///     invariants are the objective-<c>value</c> vs <c>values</c> split, the integer second key of the
///     multi-objective <c>values</c> columns, the <c>None</c> value of FAIL/PRUNED trials, the sorted
///     nested keys, and the enum state names.
/// </summary>
[TestClass]
public class CliRecordsTests
{
    private static readonly Column NumberCol = new("number", "");
    private static readonly Column ValueCol = new("value", "");
    private static readonly Column StateCol = new("state", "");

    private static Study NewStudy(params StudyDirection[] directions)
        => Optuna.CreateStudy(
            sampler: new RandomSampler(42),
            directions: directions.Length == 0 ? new[] { StudyDirection.Minimize } : directions);

    private static (List<Dictionary<Column, object?>> Records, List<Column> Columns) Build(Study study)
        => Records.CreateRecordsAndAggregateColumn(study, Records.TrialAttrs(study));

    [TestMethod]
    public void TrialAttrs_SingleObjective_UsesValue_MultiObjective_UsesValues()
    {
        Records.TrialAttrs(NewStudy(StudyDirection.Minimize)).Should().Contain("value").And.NotContain("values");
        Records.TrialAttrs(NewStudy(StudyDirection.Minimize, StudyDirection.Maximize))
            .Should().Contain("values").And.NotContain("value");
    }

    [TestMethod]
    public void CompleteTrial_HasValue_Number_State_Params()
    {
        var study = NewStudy();
        var trial = study.Ask();
        double x = trial.SuggestFloat("x", 0, 1);
        study.Tell(trial, 0.5);

        var (records, columns) = Build(study);
        records.Should().HaveCount(1);
        columns.Should().Contain(new[] { NumberCol, ValueCol, StateCol, new Column("params", "x") });

        var r = records[0];
        r[NumberCol].Should().Be(0L);
        r[ValueCol].Should().Be(0.5);
        r[StateCol].Should().Be("COMPLETE");
        r[new Column("params", "x")].Should().Be(x);
    }

    [TestMethod]
    public void FailedTrial_HasNullValue_And_FailState()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        study.Tell(trial, null, TrialState.Fail);

        var r = Build(study).Records[0];
        r[ValueCol].Should().BeNull();
        r[StateCol].Should().Be("FAIL");
    }

    [TestMethod]
    public void PrunedTrial_HasNullValue_And_PrunedState()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        study.Tell(trial, null, TrialState.Pruned);

        var r = Build(study).Records[0];
        r[ValueCol].Should().BeNull();
        r[StateCol].Should().Be("PRUNED");
    }

    [TestMethod]
    public void MultiObjective_ValuesColumns_HaveIntegerSecondKeys()
    {
        var study = NewStudy(StudyDirection.Minimize, StudyDirection.Maximize);
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        study.Tell(trial, new[] { 0.1, 0.9 });

        var (records, columns) = Build(study);
        columns.Should().Contain(new Column("values", 0L));
        columns.Should().Contain(new Column("values", 1L));
        columns.Should().NotContain(new Column("values", "0")); // string keys would break MO list reconstruction

        var r = records[0];
        r[new Column("values", 0L)].Should().Be(0.1);
        r[new Column("values", 1L)].Should().Be(0.9);
    }

    [TestMethod]
    public void MultiObjective_WithMetricNames_UsesNamedValuesColumns()
    {
        var study = NewStudy(StudyDirection.Minimize, StudyDirection.Maximize);
        study.SetMetricNames(new[] { "loss", "acc" });
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        study.Tell(trial, new[] { 0.1, 0.9 });

        var (records, columns) = Build(study);
        columns.Should().Contain(new[] { new Column("values", "loss"), new Column("values", "acc") });
        records[0][new Column("values", "loss")].Should().Be(0.1);
        records[0][new Column("values", "acc")].Should().Be(0.9);
    }

    [TestMethod]
    public void Params_Columns_AreSortedByName()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestFloat("z", 0, 1);
        trial.SuggestFloat("a", 0, 1);
        trial.SuggestInt("m", 0, 10);
        study.Tell(trial, 1.0);

        var paramColumns = Build(study).Columns.Where(c => c.Name == "params").ToList();
        paramColumns.Should().Equal(new Column("params", "a"), new Column("params", "m"), new Column("params", "z"));
    }

    [TestMethod]
    public void UserAttrs_BecomeNestedColumns()
    {
        var study = NewStudy();
        var trial = study.Ask();
        trial.SuggestFloat("x", 0, 1);
        trial.SetUserAttr("note", "hello");
        study.Tell(trial, 1.0);

        var (records, columns) = Build(study);
        columns.Should().Contain(new Column("user_attrs", "note"));
        records[0][new Column("user_attrs", "note")].Should().Be("hello");
    }

    [TestMethod]
    public void EmptyStudy_ProducesNoRecordsAndNoColumns()
    {
        var (records, columns) = Build(NewStudy());
        records.Should().BeEmpty();
        columns.Should().BeEmpty();
    }

    [DataTestMethod]
    [DataRow(TrialState.Running, "RUNNING")]
    [DataRow(TrialState.Complete, "COMPLETE")]
    [DataRow(TrialState.Pruned, "PRUNED")]
    [DataRow(TrialState.Fail, "FAIL")]
    [DataRow(TrialState.Waiting, "WAITING")]
    public void StateName_MatchesPythonEnumName(TrialState state, string expected)
        => Records.StateName(state).Should().Be(expected);
}
