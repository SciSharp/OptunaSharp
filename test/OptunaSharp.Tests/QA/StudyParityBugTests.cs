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
}
