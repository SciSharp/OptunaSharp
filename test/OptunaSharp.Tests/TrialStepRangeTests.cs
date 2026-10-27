using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using AwesomeAssertions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests;

/// <summary>
///     Review item B-2 (pass 2B): trial steps are 64-bit. Python's <c>step</c> is an unbounded <c>int</c>, and its
///     journal, SQLite and in-memory storages keep steps past 2^31. With 32-bit steps every C# load of a journal
///     holding such a step threw <see cref="OverflowException" />, so no C# worker or CLI could open the study, and
///     the SQLite reader silently wrapped the step (3,000,000,000 became -1,294,967,296). The SQLite and gRPC cases
///     live in the storage suite; the round trips against live Python in the interop suite.
/// </summary>
[TestClass]
public class TrialStepRangeTests
{
    private const long PastInt32 = 3_000_000_000L;

    /// <summary>
    ///     An in-memory trial stores a report at a step past Int32 under that exact step, reports it as its last step,
    ///     and the pruners' interval arithmetic handles it: a step that is not the first report of its 1,000-step
    ///     interval is not judged again, exactly as Python's <c>_is_first_in_interval_step</c> decides.
    /// </summary>
    [TestMethod]
    public void Report_StepPastInt32_IsStoredExactly_AndPrunersUseIt()
    {
        var study = Optuna.CreateStudy(
            sampler: new RandomSampler(0), pruner: new MedianPruner(nStartupTrials: 1, intervalSteps: 1000));
        foreach (double value in new[] { 1.0, 3.0 })
        {
            var done = study.Ask();
            done.Report(value, PastInt32);
            study.Tell(done, value);
        }

        var trial = study.Ask();
        trial.Report(5.0, PastInt32);
        bool prunedAtFirstReport = trial.ShouldPrune();
        trial.Report(5.0, PastInt32 + 1);
        bool prunedWithinTheInterval = trial.ShouldPrune();

        study.Trials[0].IntermediateValues.Should().ContainKey(PastInt32).WhoseValue.Should().Be(1.0);
        study.Trials[0].LastStep.Should().Be(PastInt32);
        prunedAtFirstReport.Should().BeTrue("5 is worse than the median of 1 and 3 at that step");
        prunedWithinTheInterval.Should().BeFalse("the step is not the first report of its 1,000-step interval");
    }

    /// <summary>
    ///     The review's journal repro, written the way Python writes it: <c>trial.report(0.5, step=3_000_000_000)</c>
    ///     becomes a <c>"step": 3000000000</c> record. Loading the file used to throw <see cref="OverflowException" />
    ///     ("Value was either too large or too small for an Int32") on every attempt; now the trial carries the step.
    /// </summary>
    [TestMethod]
    public void Journal_PythonWrittenStepPastInt32_LoadsExactly()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-steps-{Guid.NewGuid():N}.log");
        try
        {
            File.WriteAllText(
                path,
                "{\"op_code\":0,\"worker_id\":\"w\",\"study_name\":\"big-steps\",\"directions\":[1]}\n" +
                "{\"op_code\":4,\"worker_id\":\"w\",\"study_id\":0,\"datetime_start\":\"2026-01-01T00:00:00.000000+00:00\"}\n" +
                "{\"op_code\":7,\"worker_id\":\"w\",\"trial_id\":0,\"step\":3000000000,\"intermediate_value\":0.5}\n");

            var storage = new JournalStorage(new JournalFileBackend(path, lockObj: new JournalFileOpenLock(path)));
            var trial = storage.GetAllTrials(storage.GetStudyIdFromName("big-steps")).Single();

            trial.IntermediateValues.Should().Equal(new Dictionary<long, double> { [PastInt32] = 0.5 });
            trial.LastStep.Should().Be(PastInt32);
        }
        finally
        {
            foreach (string file in Directory.GetFiles(Path.GetTempPath(), Path.GetFileName(path) + "*"))
                File.Delete(file);
        }
    }

    /// <summary>
    ///     A C# writer's 64-bit steps survive a journal round trip: a second storage replaying the file sees the same
    ///     steps, including the largest one Python's SQLite storage can hold.
    /// </summary>
    [TestMethod]
    public void Journal_StepsPastInt32_RoundTripThroughTheFile()
    {
        string path = Path.Combine(Path.GetTempPath(), $"optunasharp-steps-{Guid.NewGuid():N}.log");
        try
        {
            JournalStorage Open() => new(new JournalFileBackend(path, lockObj: new JournalFileOpenLock(path)));
            var study = Optuna.CreateStudy(storage: Open(), studyName: "round-trip", sampler: new RandomSampler(0));
            var trial = study.Ask();
            trial.Report(0.25, 0);
            trial.Report(0.5, PastInt32);
            trial.Report(0.75, long.MaxValue);
            study.Tell(trial, 0.75);

            var replayed = Optuna.LoadStudy("round-trip", Open()).Trials.Single();

            replayed.IntermediateValues.Should().Equal(new Dictionary<long, double>
            {
                [0] = 0.25, [PastInt32] = 0.5, [long.MaxValue] = 0.75,
            });
            replayed.LastStep.Should().Be(long.MaxValue);
        }
        finally
        {
            foreach (string file in Directory.GetFiles(Path.GetTempPath(), Path.GetFileName(path) + "*"))
                File.Delete(file);
        }
    }
}
