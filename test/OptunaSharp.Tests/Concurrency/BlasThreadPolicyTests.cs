using System;
using System.Collections.Concurrent;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using NumSharp;
using OptunaSharp.Internal;
using OptunaSharp.Samplers;

namespace OptunaSharp.Tests.Concurrency;

/// <summary>
///     <see cref="TrialExecutorOptions.BlasThreads" /> (THREADING.md §9): the automatic policy leaves OpenBLAS alone with
///     <c>nJobs: 1</c> and runs it single-threaded while several workers run, an explicit count holds for any
///     <c>nJobs</c>, concurrent calls run with the smallest count any of them holds, and the count in effect before the
///     first call is restored when the last one ends, its environment variable included.
/// </summary>
/// <remarks>
///     The OpenBLAS thread count is one native setting for the whole process, so the class never runs in parallel with
///     other tests. Each test starts from a known count of 3 (with <c>OPENBLAS_NUM_THREADS</c> unset) and puts back the
///     count and the variable it found.
/// </remarks>
[TestClass]
[DoNotParallelize]
public class BlasThreadPolicyTests
{
    private const string ThreadsVariable = "OPENBLAS_NUM_THREADS";

    // The count each test starts from: not 1 (the automatic policy's), not 2 (the explicit count the tests ask for).
    private const int Baseline = 3;

    private int _foundThreads;
    private string? _foundVariable;
    private int _baseline;

    /// <summary>
    ///     Records the live count and the variable, then sets the known starting count and unsets the variable, so the
    ///     policy's restore of both can be checked.
    /// </summary>
    /// <exception cref="AssertInconclusiveException">No OpenBLAS is loaded, or it cannot run more than one thread.</exception>
    [TestInitialize]
    public void StartFromAKnownCount()
    {
        if (LiveThreads() is not int found)
        {
            Assert.Inconclusive("No OpenBLAS is loaded in this process: the policy has nothing to set.");
            return;
        }

        _foundThreads = found;
        _foundVariable = Environment.GetEnvironmentVariable(ThreadsVariable);
        TensorEngine.Threading.SetThreads(TensorEngine.Threading.OpenBlas, Baseline);
        Environment.SetEnvironmentVariable(ThreadsVariable, null);
        // OpenBLAS may cap a count above what it was built for; the tests compare against what it actually runs.
        _baseline = LiveThreads() ?? 1;
        if (_baseline < 2)
            Assert.Inconclusive("OpenBLAS runs one thread here, so single-threading it cannot be observed.");
    }

    /// <summary>Puts back the count and the variable the test found.</summary>
    [TestCleanup]
    public void RestoreTheFoundCount()
    {
        TensorEngine.Threading.SetThreads(TensorEngine.Threading.OpenBlas, _foundThreads);
        Environment.SetEnvironmentVariable(ThreadsVariable, _foundVariable);
    }

    /// <summary>The live OpenBLAS thread count, read through NumSharp's knob.</summary>
    /// <returns>The count, or null when no OpenBLAS is loaded.</returns>
    private static int? LiveThreads()
        => TensorEngine.Threading.TryGet(TensorEngine.Threading.OpenBlas, out TensorEngine.Threading.Variable knob)
            ? knob.Threads
            : null;

    /// <summary>Runs a study and records the OpenBLAS thread count each trial saw.</summary>
    /// <param name="nJobs">The call's worker count.</param>
    /// <param name="options">The call's executor options, or null for the defaults.</param>
    /// <returns>The distinct counts the trials saw.</returns>
    private static int[] CountsSeenByTrials(int nJobs, TrialExecutorOptions? options = null)
    {
        var seen = new ConcurrentDictionary<int, bool>();
        var study = Optuna.CreateStudy(sampler: new RandomSampler(5));
        study.Optimize(t =>
        {
            seen[LiveThreads() ?? -1] = true;
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 12, nJobs: nJobs, executorOptions: options);
        return seen.Keys.ToArrayOrdered();
    }

    /// <summary>
    ///     With one worker the automatic policy touches nothing: the trials run with the count OpenBLAS already had, which
    ///     is what keeps a sequential run's linear algebra as NumPy's.
    /// </summary>
    [TestMethod]
    public void OneJob_Automatic_LeavesOpenBlasAsItIs()
    {
        CountsSeenByTrials(nJobs: 1).Should().Equal(_baseline);

        LiveThreads().Should().Be(_baseline);
        Environment.GetEnvironmentVariable(ThreadsVariable).Should().BeNull();
    }

    /// <summary>
    ///     With several workers the automatic policy runs OpenBLAS single-threaded for the whole call, then restores the
    ///     count and removes the variable NumSharp's knob wrote, since it was not set before.
    /// </summary>
    /// <param name="kind">The executor the workers run on.</param>
    [TestMethod]
    [DataRow(TrialExecutorKind.DedicatedThreads)]
    [DataRow(TrialExecutorKind.ThreadPool)]
    public void SeveralJobs_Automatic_RunSingleThreaded_ThenRestore(TrialExecutorKind kind)
    {
        CountsSeenByTrials(nJobs: 4, new TrialExecutorOptions { Executor = kind }).Should().Equal(1);

        LiveThreads().Should().Be(_baseline);
        Environment.GetEnvironmentVariable(ThreadsVariable).Should().BeNull("the variable was not set before the call");
    }

    /// <summary>An explicit count holds for any <c>nJobs</c>, one included, and the original is restored afterwards.</summary>
    /// <param name="nJobs">The call's worker count.</param>
    [TestMethod]
    [DataRow(1)]
    [DataRow(4)]
    public void Explicit_HoldsForAnyNJobs_ThenRestores(int nJobs)
    {
        CountsSeenByTrials(nJobs, new TrialExecutorOptions { BlasThreads = 2 }).Should().Equal(2);

        LiveThreads().Should().Be(_baseline);
    }

    /// <summary>
    ///     A variable that was set before the call is left set (to the restored count): only a variable the policy itself
    ///     introduced is removed.
    /// </summary>
    [TestMethod]
    public void AVariableSetBeforeTheCall_IsKept()
    {
        Environment.SetEnvironmentVariable(ThreadsVariable, "7");

        CountsSeenByTrials(nJobs: 4).Should().Equal(1);

        Environment.GetEnvironmentVariable(ThreadsVariable).Should().NotBeNull();
        LiveThreads().Should().Be(_baseline);
    }

    /// <summary>
    ///     A call that ends with its objective's exception restores the count all the same: the hold is released on every
    ///     path out of <c>Optimize</c>.
    /// </summary>
    /// <exception cref="InvalidOperationException">
    ///     Thrown by the objective on purpose and caught by the assertion; it escapes only if <c>Optimize</c> stopped
    ///     raising its objective's error, which fails the test.
    /// </exception>
    /// <remarks>
    ///     Every trial fails, so the first trial the scheduling loop collects is a failure and the call raises it. A single
    ///     failing trial would not do: as in Python, a failure the loop has not collected when the last trial is submitted
    ///     (the final batch) is logged, not raised, so whether the call raised would depend on timing.
    /// </remarks>
    [TestMethod]
    public void TheCountIsRestored_WhenTheObjectiveThrows()
    {
        using var quiet = new QuietLogs();
        var study = Optuna.CreateStudy(sampler: new RandomSampler(6));
        // Typed as single-objective explicitly: a throw-only lambda has no return type to pick an overload by.
        Func<Trial, double> failing = _ => throw new InvalidOperationException("objective failed");

        Action run = () => study.Optimize(failing, nTrials: 12, nJobs: 4);

        run.Should().Throw<InvalidOperationException>().WithMessage("objective failed");
        LiveThreads().Should().Be(_baseline);
    }

    /// <summary>
    ///     Two calls at once, one asking for 2 threads and one running the automatic policy at <c>nJobs: 4</c>: while both
    ///     run, OpenBLAS runs 1 thread (the smallest held); when the parallel call ends the other gets its 2 back; when the
    ///     last ends the original returns.
    /// </summary>
    [TestMethod]
    public void ConcurrentCalls_RunWithTheSmallestCount_AndTheLastRestores()
    {
        using var parallelRunning = new ManualResetEventSlim();
        using var countRead = new ManualResetEventSlim();
        using var parallelDone = new ManualResetEventSlim();
        int whileParallel = 0, afterParallel = 0;

        var holder = Task.Factory.StartNew(() =>
        {
            var study = Optuna.CreateStudy(sampler: new RandomSampler(7));
            study.Optimize(t =>
            {
                parallelRunning.Wait(TimeSpan.FromSeconds(30));
                whileParallel = LiveThreads() ?? -1;
                countRead.Set();
                parallelDone.Wait(TimeSpan.FromSeconds(30));
                afterParallel = LiveThreads() ?? -1;
                return t.SuggestFloat("x", 0, 1);
            }, nTrials: 1, nJobs: 1, executorOptions: new TrialExecutorOptions { BlasThreads = 2 });
        }, TaskCreationOptions.LongRunning);

        var parallel = Optuna.CreateStudy(sampler: new RandomSampler(8));
        parallel.Optimize(t =>
        {
            parallelRunning.Set();
            countRead.Wait(TimeSpan.FromSeconds(30));
            return t.SuggestFloat("x", 0, 1);
        }, nTrials: 4, nJobs: 4);
        parallelDone.Set();
        holder.Wait(TimeSpan.FromSeconds(60)).Should().BeTrue();

        whileParallel.Should().Be(1, "the smallest count held while both calls run");
        afterParallel.Should().Be(2, "the remaining call's own count once the other released its hold");
        LiveThreads().Should().Be(_baseline);
    }

    /// <summary>
    ///     The policy's holds nest by count: the smallest held is in effect, a release reapplies the next smallest, an
    ///     equal second hold changes nothing, a null hold does nothing, and the last release restores the original.
    /// </summary>
    [TestMethod]
    public void Holds_ApplyTheSmallestCount_AndTheLastReleaseRestores()
    {
        using (BlasThreadPolicy.Hold(null))
            LiveThreads().Should().Be(_baseline, "a null hold leaves OpenBLAS alone");

        BlasThreadPolicy.Lease two = BlasThreadPolicy.Hold(2);
        LiveThreads().Should().Be(2);
        BlasThreadPolicy.Lease alsoTwo = BlasThreadPolicy.Hold(2);
        BlasThreadPolicy.Lease one = BlasThreadPolicy.Hold(1);
        LiveThreads().Should().Be(1);

        one.Dispose();
        LiveThreads().Should().Be(2);
        two.Dispose();
        LiveThreads().Should().Be(2, "an equal hold is still held");
        alsoTwo.Dispose();
        LiveThreads().Should().Be(_baseline);
        Environment.GetEnvironmentVariable(ThreadsVariable).Should().BeNull();
    }
}

/// <summary>Small helpers for the policy tests.</summary>
internal static class BlasTestExtensions
{
    /// <summary>The keys in ascending order.</summary>
    /// <param name="keys">The keys of a concurrent set.</param>
    /// <returns>A new ascending array.</returns>
    internal static int[] ToArrayOrdered(this System.Collections.Generic.ICollection<int> keys)
    {
        var array = new int[keys.Count];
        keys.CopyTo(array, 0);
        Array.Sort(array);
        return array;
    }
}
