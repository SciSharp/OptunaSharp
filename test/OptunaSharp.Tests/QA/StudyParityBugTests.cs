using System;
using System.Collections.Generic;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.QA;

/// <summary>
///     Open parity bugs in the study and trial surface (tell, logging text, reprs), found by the 2026-11-02
///     parity QA run (<c>docs/qa/PARITY_QA_2026-11-02.md</c>). Expected values are Optuna 5.0.0's output on
///     CPython 3.12; each test fails today and passes once the divergence is fixed, at which point its
///     <c>[OpenBugs]</c> attribute should be removed.
/// </summary>
[TestClass]
public class StudyParityBugTests
{
    /// <summary>Log handler that keeps the text of every record it receives, in order.</summary>
    /// <remarks>Registered for one test at a time; records from other threads' studies can interleave.</remarks>
    private sealed class CollectingHandler : ILogHandler
    {
        /// <summary>The messages received so far, oldest first.</summary>
        public List<string> Messages { get; } = new();

        /// <summary>Stores the record's message; never throws, so it cannot disturb the code under test.</summary>
        /// <param name="record">The log record delivered by <see cref="OptunaLogging" />.</param>
        public void Emit(in LogRecord record)
        {
            lock (Messages)
                Messages.Add(record.Message);
        }
    }

    /// <summary>
    ///     QA-12: a skipped tell must log the value exactly as it was passed; for the scalar overload Python
    ///     prints <c>2.0</c>, the port prints <c>[2.0]</c>.
    /// </summary>
    /// <remarks>
    ///     Optuna's <c>_tell_with_warning</c> logs <c>f"Skipped telling trial {n} with values {values} ..."</c>
    ///     with the caller's own argument. <c>Study.Tell(Trial, double, bool)</c> wraps the scalar in an array
    ///     before the message is built, so the C# text shows a list the caller never passed.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void TellSkipIfFinished_ScalarValue_LogsScalarLikePython()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        var trial = study.Ask();
        study.Tell(trial, 1.0);

        var handler = new CollectingHandler();
        int verbosity = OptunaLogging.GetVerbosity();
        OptunaLogging.SetVerbosity(LogLevel.Info);
        OptunaLogging.AddHandler(handler);
        try
        {
            study.Tell(trial, 2.0, skipIfFinished: true);
        }
        finally
        {
            // Global logging state: always restore it, or later tests log at the wrong level.
            OptunaLogging.RemoveHandler(handler);
            OptunaLogging.SetVerbosity(verbosity);
        }

        handler.Messages.Should().Contain(
            "Skipped telling trial 0 with values 2.0 and state None since trial was already finished. " +
            "Finished trial has values [1.0] and state TrialState.COMPLETE.");
    }

    /// <summary>
    ///     QA-18: <c>trials_dataframe</c> must sort the nested columns of a dict attribute with Python's
    ///     <c>sorted()</c> on string keys; the port sorts digit-only keys numerically.
    /// </summary>
    /// <remarks>
    ///     Optuna builds the columns with <c>sorted(column_agg[k])</c>: parameter and attribute names are
    ///     strings, so <c>'10' &lt; '2'</c>. The port's <c>NestedComparer</c> parses any digit-only key as a
    ///     number (it exists for the int keys of <c>values</c> and <c>intermediate_values</c>), so parameters
    ///     named <c>"1"</c>, <c>"2"</c>, <c>"10"</c> come out as <c>params_1, params_2, params_10</c>. Python
    ///     5.0.0 for this study: <c>['number', 'params_1', 'params_10', 'params_2', 'user_attrs_11',
    ///     'user_attrs_3']</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void TrialsDataframe_DigitOnlyKeys_SortAsStringsLikePython()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        study.Optimize(t =>
        {
            foreach (var name in new[] { "10", "2", "1" })
                t.SuggestFloat(name, 0, 1);
            t.SetUserAttr("11", 1L);
            t.SetUserAttr("3", 2L);
            return 1.0;
        }, nTrials: 2);

        var table = study.TrialsDataframe(attrs: new[] { "number", "params", "user_attrs" });

        table.Columns.Should().Equal("number", "params_1", "params_10", "params_2", "user_attrs_11", "user_attrs_3");
    }

    /// <summary>
    ///     QA-19: <c>trials_dataframe</c> must accept every <see cref="FrozenTrial" /> attribute Python's
    ///     <c>getattr</c> accepts, including <c>distributions</c> and underscore-prefixed names.
    /// </summary>
    /// <remarks>
    ///     Optuna reads each requested attribute with <c>getattr(trial, attr)</c>: a dict such as
    ///     <c>distributions</c> expands to nested columns, and <c>_number</c> is shown as <c>number</c>. The
    ///     port's fixed attribute switch throws <c>ValueError: Unsupported attr: distributions</c>. Python 5.0.0
    ///     columns for this request: <c>['number', 'distributions_c', 'distributions_x',
    ///     'intermediate_values_0', 'intermediate_values_2', 'intermediate_values_10', 'last_step', 'trial_id',
    ///     'number']</c>.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void TrialsDataframe_DistributionsAndPrivateAttrs_SupportedLikePython()
    {
        var study = Optuna.CreateStudy(sampler: new RandomSampler(seed: 0));
        study.Optimize(t =>
        {
            t.SuggestFloat("x", 0, 1);
            t.SuggestCategorical("c", new[] { "a", "b" });
            t.Report(0.5, 0);
            t.Report(0.25, 10);
            t.Report(0.3, 2);
            return 1.0;
        }, nTrials: 2);

        var table = study.TrialsDataframe(attrs: new[] { "number", "distributions", "intermediate_values", "last_step", "_trial_id", "_number" });

        table.Columns.Should().Equal(
            "number", "distributions_c", "distributions_x", "intermediate_values_0", "intermediate_values_2",
            "intermediate_values_10", "last_step", "trial_id", "number");
    }

    /// <summary>
    ///     QA-13: <see cref="FrozenTrial.ToString" /> must render like Python's <c>FrozenTrial.__repr__</c>: the
    ///     state as <c>&lt;TrialState.COMPLETE: 1&gt;</c> and datetimes as <c>datetime.datetime(...)</c>.
    /// </summary>
    /// <remarks>
    ///     Python's repr formats every field with <c>repr()</c>, so the enum shows its <c>&lt;Name: value&gt;</c>
    ///     form and a datetime its constructor form, dropping a zero microsecond. The port uses
    ///     <c>str(state)</c> (<c>TrialState.COMPLETE</c>) and quoted <c>'yyyy-MM-dd HH:mm:ss.ffffff'</c> strings
    ///     (which also follow the culture's calendar, catalog item 24e). Expected text from Optuna 5.0.0 for the
    ///     same trial.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void FrozenTrialRepr_StateAndDatetimes_RenderLikePython()
    {
        var trial = new FrozenTrial(
            number: 3,
            state: TrialState.Complete,
            value: 0.5,
            datetimeStart: new DateTime(2026, 1, 2, 3, 4, 5, 60),
            datetimeComplete: new DateTime(2026, 1, 2, 3, 4, 6),
            @params: new OrderedDict<string, object?> { ["x"] = 0.25, ["c"] = null },
            distributions: new OrderedDict<string, BaseDistribution>
            {
                ["x"] = new FloatDistribution(0.0, 1.0),
                ["c"] = new CategoricalDistribution(CategoricalChoice.None, "a"),
            },
            userAttrs: new OrderedDict<string, object?> { ["u"] = new List<object?> { 1L, "b" } },
            systemAttrs: new OrderedDict<string, object?>(),
            intermediateValues: new OrderedDict<long, double> { [0] = 1.5, [3] = double.NaN },
            trialId: 7);

        trial.ToString().Should().Be(
            "FrozenTrial(number=3, state=<TrialState.COMPLETE: 1>, values=[0.5], " +
            "datetime_start=datetime.datetime(2026, 1, 2, 3, 4, 5, 60000), " +
            "datetime_complete=datetime.datetime(2026, 1, 2, 3, 4, 6), params={'x': 0.25, 'c': None}, " +
            "user_attrs={'u': [1, 'b']}, system_attrs={}, intermediate_values={0: 1.5, 3: nan}, " +
            "distributions={'x': FloatDistribution(high=1.0, log=False, low=0.0, step=None), " +
            "'c': CategoricalDistribution(choices=(None, 'a'))}, trial_id=7, value=None)");
    }

    /// <summary>
    ///     QA-43 (dataframe): <c>trials_dataframe</c> must order nested columns by Python's string order, which
    ///     compares code points; the port compares UTF-16 code units, so a name with a character above U+FFFF sorts
    ///     before one with a character in U+E000–U+FFFF.
    /// </summary>
    /// <remarks>
    ///     A surrogate pair starts with a code unit in U+D800–U+DBFF, below every code unit from U+E000 on, while its
    ///     code point is above U+FFFF. With parameters <c>b</c>, <c>ｘ</c> (U+FF58, FULLWIDTH LATIN SMALL LETTER X) and
    ///     <c>😀</c> (U+1F600), Optuna 5.0.0 gives the columns <c>params_b, params_ｘ, params_😀</c>; the port gives
    ///     <c>params_b, params_😀, params_ｘ</c>. Every <c>sorted()</c> port that uses <c>StringComparer.Ordinal</c>
    ///     shares the root cause (GridSampler, the importance evaluators' parameter order, the contour / slice / rank /
    ///     parallel-coordinate parameter order).
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void TrialsDataframe_AstralCharacterNames_SortByCodePointLikePython()
    {
        string fullwidthX = "ｘ", grinning = char.ConvertFromUtf32(0x1F600);
        var study = Optuna.CreateStudy();
        var distribution = new FloatDistribution(0.0, 1.0);
        study.AddTrial(FrozenTrial.CreateTrial(value: 1.0,
            @params: new OrderedDict<string, object?> { [grinning] = 0.5, [fullwidthX] = 0.25, ["b"] = 0.75 },
            distributions: new OrderedDict<string, BaseDistribution> { [grinning] = distribution, [fullwidthX] = distribution, ["b"] = distribution }));

        study.TrialsDataframe().Columns.Should().Equal(
            "number", "value", "datetime_start", "datetime_complete", "duration",
            "params_b", "params_" + fullwidthX, "params_" + grinning, "state");
    }

    /// <summary>
    ///     QA-50: <c>enqueue_trial(..., skip_if_exists=True)</c> must compare infinite parameter values the way
    ///     <c>np.isclose</c> does — equal infinities are duplicates, nothing else is close to an infinity — but the
    ///     port evaluates <c>|a - b| &lt;= 1e-5 * |b|</c> on them, which inverts both cases.
    /// </summary>
    /// <remarks>
    ///     <c>Study._should_skip_enqueue</c> calls <c>np.isclose(new, existing, atol=0.0, equal_nan=True)</c>; NumPy only
    ///     applies the tolerance formula to finite pairs and compares the rest with <c>==</c>. In the port's
    ///     <c>ShouldSkipEnqueue</c>, <c>inf - inf</c> is NaN (never within tolerance), so a repeated <c>inf</c> is enqueued
    ///     twice, while <c>|x - inf| = inf &lt;= 1e-5 * inf</c> holds for every <c>x</c>, so <c>-inf</c> or any finite value
    ///     enqueued over a trial whose parameter is <c>inf</c> is dropped as a duplicate. Optuna 5.0.0 keeps 1, 2 and 2
    ///     trials in the three studies below; the port keeps 2, 1 and 1. Found by the D95 probe of the duplicate check.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void EnqueueTrial_SkipIfExists_ComparesInfinitiesLikeNumpyIsclose()
    {
        int TrialsAfter(double existing, double enqueued)
        {
            var study = Optuna.CreateStudy();
            study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = existing });
            study.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = enqueued }, skipIfExists: true);
            return study.Trials.Count;
        }

        new[]
        {
            TrialsAfter(double.PositiveInfinity, double.PositiveInfinity),
            TrialsAfter(double.PositiveInfinity, double.NegativeInfinity),
            TrialsAfter(double.PositiveInfinity, 0.5),
        }.Should().Equal(1, 2, 2);
    }

    /// <summary>
    ///     QA-51: <c>create_study</c> rejecting a <c>NOT_SET</c> direction must print the directions list the way
    ///     Python's f-string does, with each enum member's <c>repr</c> (<c>&lt;StudyDirection.NOT_SET: 0&gt;</c>); the port
    ///     prints <c>StudyDirection.NOT_SET</c>, the member's <c>str</c>.
    /// </summary>
    /// <remarks>
    ///     <c>_create_study</c> raises <c>f"... but got {directions}. ..."</c>, and a list formats its items with
    ///     <c>repr()</c>; members passed as enums (the only way a C# caller can pass them) therefore appear as
    ///     <c>&lt;StudyDirection.MINIMIZE: 1&gt;</c>. <c>Optuna.CreateStudy</c> joins <c>"StudyDirection." + PythonName()</c>.
    ///     Found by the D96 argument probe; every other <c>create_study</c> / <c>optimize</c> argument error there
    ///     (both directions given, empty directions, <c>n_jobs</c> 0 or -2, negative <c>n_trials</c> / <c>timeout</c>)
    ///     matches Optuna 5.0.0.
    /// </remarks>
    [TestMethod]
    [OpenBugs]
    public void CreateStudy_NotSetDirection_MessageListsEnumReprsLikePython()
    {
        ((Action)(() => Optuna.CreateStudy(directions: new[] { StudyDirection.Minimize, StudyDirection.NotSet })))
            .Should().Throw<NumSharp.ValueError>().Which.Message.Should().Be(
                "`directions` must be a list of `minimize` or `maximize`, but got " +
                "[<StudyDirection.MINIMIZE: 1>, <StudyDirection.NOT_SET: 0>]. " +
                "For single-objective optimization, please use `direction` instead of `directions`.");
    }
}
