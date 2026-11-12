using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using AwesomeAssertions;
using OptunaSharp.Collections;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages.Journal;

namespace OptunaSharp.Tests.Storages;

/// <summary>
///     Concurrency of the journal file storage across topologies: several workers on one journal, each with its own
///     <see cref="JournalStorage" /> (the in-process shape of several processes), several storages over one
///     backend, and several studies in one journal. Every test checks the result on a fresh replay of the file,
///     because a run that throws nothing can still leave a broken study behind.
/// </summary>
/// <remarks>
///     <para>
///         The lock tests are the regression for the 2026-10 race: both locks decided "contention or real error"
///         by re-checking the lock path after a failed create, so a holder releasing in between made the waiter
///         throw <c>IOException: … already exists</c>. 16 separate workers lost 7–10 of 16 to it. The sizes here
///         make the open-lock test fail on that code with a probability above 99%.
///     </para>
///     <para>
///         The workers run as threads with separate storage and lock instances. To the journal that is the same as
///         separate processes: exclusion between them goes only through the lock file, never through a shared monitor.
///     </para>
/// </remarks>
[TestClass]
public class JournalConcurrencyTests
{
    private const int Steps = 2;

    // A fresh journal per test, so concurrent workers of one test never see another test's records.
    private string _path = null!;

    /// <summary>Picks a unique journal path in the temp directory; the backend creates the file on first use.</summary>
    [TestInitialize]
    public void SetUp()
        => _path = Path.Combine(Path.GetTempPath(), $"optunasharp-journal-cc-{Guid.NewGuid():N}.log");

    /// <summary>
    ///     Deletes the journal and everything derived from its name (the lock, renamed locks, the symlink probe).
    ///     A file still held by a straggling handle is left for the OS temp cleanup instead of failing the test.
    /// </summary>
    [TestCleanup]
    public void TearDown()
    {
        foreach (var f in Directory.GetFiles(Path.GetTempPath(), Path.GetFileName(_path) + "*"))
        {
            try
            {
                File.Delete(f);
            }
            catch (IOException)
            {
            }
        }
    }

    // ------------------------------------------------------------------ the locks alone

    /// <summary>
    ///     Sixteen open-file locks on one journal, each appending forty records at once: every append waits for
    ///     its turn, none fails, and the file holds exactly the records written, with no lock left behind.
    /// </summary>
    [TestMethod]
    public void OpenLock_ContendedByManyInstances_NeverFailsAnAppend()
    {
        var errors = AppendConcurrently(16, 40, () => new JournalFileOpenLock(_path));

        errors.Should().BeEmpty("contention must be waited out, never raised");
        File.ReadAllLines(_path).Should().HaveCount(16 * 40).And.OnlyContain(l => l.StartsWith("{\"op_code\":2,", StringComparison.Ordinal));
        File.Exists(_path + ".lock").Should().BeFalse();
    }

    /// <summary>The same contention through Optuna's default symlink lock, where this machine can create symbolic links.</summary>
    [TestMethod]
    public void SymlinkLock_ContendedByManyInstances_NeverFailsAnAppend()
    {
        RequireSymlinks();
        var errors = AppendConcurrently(16, 40, () => new JournalFileSymlinkLock(_path));

        errors.Should().BeEmpty("contention must be waited out, never raised");
        File.ReadAllLines(_path).Should().HaveCount(16 * 40);
        File.Exists(_path + ".lock").Should().BeFalse();
    }

    /// <summary>
    ///     The backend's default lock must keep Optuna's symlink protocol under contention: it may switch to the
    ///     open-file protocol only when symbolic links cannot be created at all. Before the fix, contention alone
    ///     flipped up to 31 of 32 instances.
    /// </summary>
    [TestMethod]
    public void DefaultLock_UnderContention_StaysOnTheSymlinkProtocol()
    {
        RequireSymlinks();
        var locks = new ConcurrentBag<JournalFileDefaultLock>();
        var errors = AppendConcurrently(16, 40, () =>
        {
            var lockObj = new JournalFileDefaultLock(_path);
            locks.Add(lockObj);
            return lockObj;
        });

        errors.Should().BeEmpty();
        File.ReadAllLines(_path).Should().HaveCount(16 * 40);
        locks.Should().HaveCount(16).And.OnlyContain(l => !l.UsesOpenLock, "contention is not a reason to switch protocols");
    }

    /// <summary>
    ///     How a failed acquire is classified, the port of Python's <c>err.errno == errno.EEXIST</c>: the two Windows
    ///     "already exists" codes count as contention on every OS (the symlink lock's fast path throws one of them
    ///     itself), and EEXIST counts off Windows. Anything else, such as a missing directory, a denied path or a
    ///     generic I/O error, is a real failure and must propagate.
    /// </summary>
    [TestMethod]
    public void LockProtocol_ClassifiesOnlyAlreadyExistsAsContention()
    {
        static IOException Io(int hResult) => new("probe", hResult);

        JournalFileLockProtocol.IsAlreadyExists(Io(unchecked((int)0x80070050))).Should().BeTrue("ERROR_FILE_EXISTS");
        JournalFileLockProtocol.IsAlreadyExists(Io(unchecked((int)0x800700B7))).Should().BeTrue("ERROR_ALREADY_EXISTS");
        JournalFileLockProtocol.IsAlreadyExists(Io(17)).Should().Be(!OperatingSystem.IsWindows(),
            ".NET puts the raw errno in HResult only off Windows");
        JournalFileLockProtocol.IsAlreadyExists(new DirectoryNotFoundException("probe")).Should().BeFalse("a missing directory");
        JournalFileLockProtocol.IsAlreadyExists(Io(unchecked((int)0x80070005))).Should().BeFalse("access denied");
        JournalFileLockProtocol.IsAlreadyExists(new IOException("probe")).Should().BeFalse("a generic I/O error");
    }

    /// <summary>
    ///     A lock whose directory does not exist fails at once instead of waiting forever. The failure is not "already
    ///     exists", so the error-code classification must let it through, on both protocols. A watchdog turns a
    ///     regression into a failure rather than a hung run.
    /// </summary>
    [TestMethod]
    public void Locks_InAMissingDirectory_RaiseInsteadOfWaiting()
    {
        string missing = Path.Combine(Path.GetTempPath(), $"optunasharp-missing-{Guid.NewGuid():N}", "journal.log");
        foreach (var lockObj in new BaseJournalFileLock[] { new JournalFileOpenLock(missing), new JournalFileSymlinkLock(missing) })
        {
            var acquire = System.Threading.Tasks.Task.Run(() => lockObj.Acquire());

            ((IAsyncResult)acquire).AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(10)).Should().BeTrue(
                $"{lockObj.GetType().Name} must raise, not wait, when its directory is missing");
            acquire.IsFaulted.Should().BeTrue();
            Exception? thrown = acquire.Exception!.InnerException;
            thrown.Should().NotBeNull();
            (thrown is IOException io && JournalFileLockProtocol.IsAlreadyExists(io)).Should().BeFalse(
                "the failure is a real one, never contention");
        }
    }

    /// <summary>
    ///     A waiter takes the lock within milliseconds of its release, however long it waited: the retry sleep stops
    ///     doubling at 16 ms. With Python's 1 s cap, a waiter that had already waited a second slept up to another
    ///     second past the release, and the journal sat idle (16 contending writers managed about 43 appends a
    ///     second). After a 1.2 s hold, that waiter would take the lock about 0.8 s late.
    /// </summary>
    [TestMethod]
    public void Lock_ReleasedAfterALongHold_IsTakenWithinMilliseconds()
    {
        var holder = new JournalFileOpenLock(_path);
        holder.Acquire().Should().BeTrue();
        var waiter = new JournalFileOpenLock(_path);
        long acquiredAt = 0;
        var waiting = new Thread(() =>
        {
            waiter.Acquire();
            acquiredAt = Stopwatch.GetTimestamp();
        });
        waiting.Start();

        Thread.Sleep(1200);   // longer than the 1023 ms the doubling sleeps take to reach Python's 1 s cap
        long releasedAt = Stopwatch.GetTimestamp();
        holder.Release();

        waiting.Join(TimeSpan.FromSeconds(10)).Should().BeTrue();
        waiter.Release();
        double lagMs = (acquiredAt - releasedAt) * 1000.0 / Stopwatch.Frequency;
        lagMs.Should().BeLessThan(400, "a waiter retries at least every 16 ms");
    }

    /// <summary>
    ///     A lock left behind by a holder that died (here a lock nobody owns) is forcibly released once its modification
    ///     time has stood still for the grace period, and the waiter then takes it with its own protocol: Optuna's
    ///     recovery (Optuna 5.0: a left-behind <c>.lock</c> with <c>grace_period=1</c> was acquired after 1.03 s).
    ///     The shorter retry sleep must not shorten the grace period.
    /// </summary>
    /// <param name="protocol"><c>open</c> for a regular lock file, <c>symlink</c> for a symbolic link.</param>
    [TestMethod]
    [DataRow("open")]
    [DataRow("symlink")]
    public void Lock_LeftBehindByADeadHolder_IsReclaimedAfterTheGracePeriod(string protocol)
    {
        File.WriteAllText(_path, "");
        if (protocol == "symlink")
        {
            RequireSymlinks();
            File.CreateSymbolicLink(_path + ".lock", _path);
        }
        else
        {
            File.WriteAllText(_path + ".lock", "");
        }
        // One second keeps the test short; the constructor warns that it is small, as Python's does.
        BaseJournalFileLock lockObj = protocol == "symlink"
            ? new JournalFileSymlinkLock(_path, gracePeriod: 1)
            : new JournalFileOpenLock(_path, gracePeriod: 1);

        var watch = Stopwatch.StartNew();
        var acquire = Task.Run(() => lockObj.Acquire());

        ((IAsyncResult)acquire).AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(15)).Should().BeTrue("a dead holder's lock must be reclaimed");
        acquire.Result.Should().BeTrue();
        watch.Elapsed.TotalSeconds.Should().BeGreaterThanOrEqualTo(0.9, "the lock is forced only after the grace period");
        (new FileInfo(_path + ".lock").LinkTarget is not null).Should().Be(protocol == "symlink",
            "the waiter now holds the path with its own protocol");
        lockObj.Release();
        File.Exists(_path + ".lock").Should().BeFalse();
    }

    /// <summary>
    ///     Without a grace period a left-behind lock is never forced: the waiter is still waiting well past the point where
    ///     a one-second grace period would have reclaimed it, and takes the lock as soon as the file is removed. That is
    ///     the cost of <c>gracePeriod: null</c>: a crashed holder blocks every writer until someone deletes its lock.
    /// </summary>
    [TestMethod]
    public void Lock_WithoutAGracePeriod_WaitsForALeftBehindLockUntilItIsRemoved()
    {
        File.WriteAllText(_path + ".lock", "");
        var lockObj = new JournalFileOpenLock(_path, gracePeriod: null);

        var acquire = Task.Run(() => lockObj.Acquire());
        ((IAsyncResult)acquire).AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(1.5)).Should().BeFalse("no grace period means no forced release");

        // Removed the way a release removes a lock, renamed away and then deleted, so the lock path is free at once.
        // Deleting it in place leaves the name "delete pending" for a moment on Windows, and the waiter's create used to
        // land in that window now and then (2 of about 8 full runs on a loaded machine); that case has its own test,
        // Lock_DeletedInPlace_IsWaitedOutOnWindows.
        string removed = _path + ".lock." + Guid.NewGuid().ToString("N") + ".removed";
        File.Move(_path + ".lock", removed);
        File.Delete(removed);
        ((IAsyncResult)acquire).AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(10)).Should().BeTrue();
        acquire.Result.Should().BeTrue();
        lockObj.Release();
    }

    /// <summary>
    ///     A lock path held "delete pending" (deleted in place, its deleting handle still open: the window an in-place
    ///     delete opens on Windows) is waited out. While the name is pending a plain exclusive create is denied, the
    ///     waiter keeps waiting instead of failing, and it takes the lock once the name is gone. Optuna's waiter raises
    ///     <c>PermissionError</c> in that window, hence <c>[Misaligned]</c> (owner's choice, 2026-11-12).
    /// </summary>
    /// <exception cref="System.ComponentModel.Win32Exception">The test cannot put the lock path in the delete-pending state.</exception>
    [TestMethod]
    [WindowsOnly]
    [Misaligned]
    public void Lock_DeletedInPlace_IsWaitedOutOnWindows()
    {
        if (!OperatingSystem.IsWindows())
        {
            Assert.Inconclusive("The delete-pending state is Windows'.");
            return;
        }

        string lockPath = _path + ".lock";
        File.WriteAllText(lockPath, "");
        var lockObj = new JournalFileOpenLock(_path, gracePeriod: null);
        Task<bool> acquire;
        using (DeletePending.Hold(lockPath))
        {
            // The window is real: an exclusive create is denied while the name is pending (else this test proves nothing).
            Action create = () => new FileStream(lockPath, FileMode.CreateNew, FileAccess.Write, FileShare.None).Dispose();
            create.Should().Throw<UnauthorizedAccessException>();

            acquire = Task.Run(() => lockObj.Acquire());
            // A waiter that failed on the denial would have completed (faulted) by now.
            ((IAsyncResult)acquire).AsyncWaitHandle.WaitOne(TimeSpan.FromMilliseconds(300)).Should()
                .BeFalse("the waiter waits out a pending name instead of failing");
        }

        ((IAsyncResult)acquire).AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(10)).Should().BeTrue();
        acquire.Result.Should().BeTrue();
        lockObj.Release();
    }

    /// <summary>
    ///     A denial that does not pass (a directory standing at the lock path denies every create) is still reported, as
    ///     the <see cref="UnauthorizedAccessException" /> it is, once the run of denials outlasts
    ///     <see cref="JournalFileLockProtocol.MaxDeniedStreakMilliseconds" />: the wait for a pending name is bounded.
    /// </summary>
    /// <exception cref="UnauthorizedAccessException">Never escapes: the assertion expects it.</exception>
    [TestMethod]
    [WindowsOnly]
    public void Lock_PersistentlyDenied_IsReportedAfterTheBound()
    {
        if (!OperatingSystem.IsWindows())
        {
            Assert.Inconclusive("The waited-out denial is Windows-only; elsewhere a denial is reported at once.");
            return;
        }

        string lockPath = _path + ".lock";
        Directory.CreateDirectory(lockPath);
        try
        {
            var lockObj = new JournalFileOpenLock(_path, gracePeriod: null);
            var watch = Stopwatch.StartNew();
            Action acquire = () => lockObj.Acquire();
            acquire.Should().Throw<UnauthorizedAccessException>();
            watch.Elapsed.Should().BeGreaterThanOrEqualTo(TimeSpan.FromMilliseconds(JournalFileLockProtocol.MaxDeniedStreakMilliseconds));
            watch.Elapsed.Should().BeLessThan(TimeSpan.FromSeconds(10));
        }
        finally
        {
            Directory.Delete(lockPath);
        }
    }

    /// <summary>Holds a file "delete pending" on Windows: the classic delete disposition, set on a handle kept open.</summary>
    /// <remarks>
    ///     <c>File.Delete</c> cannot do this on current Windows: it deletes with POSIX semantics, so the name is gone as soon
    ///     as the call returns, and the window it opens lasts only while <c>DeleteFile</c> holds its own handle (measured
    ///     2026-11-12). The classic disposition keeps the name until the handle that set it is closed.
    /// </remarks>
    private static class DeletePending
    {
        private const uint DeleteAccess = 0x00010000;
        private const uint ShareReadWriteDelete = 0x7;
        private const uint OpenExisting = 3;
        private const int FileDispositionInfo = 4;

        /// <summary>Marks <paramref name="path" /> for deletion and keeps its name pending until the returned handle is disposed.</summary>
        /// <param name="path">An existing file.</param>
        /// <returns>The handle that holds the pending deletion; disposing it removes the name.</returns>
        /// <exception cref="System.ComponentModel.Win32Exception">The file cannot be opened for deletion, or the disposition cannot be set.</exception>
        internal static Microsoft.Win32.SafeHandles.SafeFileHandle Hold(string path)
        {
            var handle = CreateFileW(path, DeleteAccess, ShareReadWriteDelete, IntPtr.Zero, OpenExisting, 0, IntPtr.Zero);
            if (handle.IsInvalid)
                throw new System.ComponentModel.Win32Exception(System.Runtime.InteropServices.Marshal.GetLastWin32Error());
            byte deleteFile = 1;
            if (!SetFileInformationByHandle(handle, FileDispositionInfo, ref deleteFile, 1))
            {
                int error = System.Runtime.InteropServices.Marshal.GetLastWin32Error();
                handle.Dispose();
                throw new System.ComponentModel.Win32Exception(error);
            }
            return handle;
        }

        /// <summary>Win32 <c>CreateFileW</c>, here to open a file with DELETE access.</summary>
        /// <param name="name">The file path.</param>
        /// <param name="access">The requested access rights.</param>
        /// <param name="share">The share mode.</param>
        /// <param name="security">Security attributes; none.</param>
        /// <param name="disposition">The creation disposition.</param>
        /// <param name="flags">Flags and attributes; none.</param>
        /// <param name="template">A template file; none.</param>
        /// <returns>The handle; invalid on failure (see the last Win32 error).</returns>
        [System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true, CharSet = System.Runtime.InteropServices.CharSet.Unicode)]
        private static extern Microsoft.Win32.SafeHandles.SafeFileHandle CreateFileW(
            string name, uint access, uint share, IntPtr security, uint disposition, uint flags, IntPtr template);

        /// <summary>Win32 <c>SetFileInformationByHandle</c>, here with the one-byte <c>FILE_DISPOSITION_INFO</c>.</summary>
        /// <param name="handle">A handle opened with DELETE access.</param>
        /// <param name="infoClass">The information class (<c>FileDispositionInfo</c>).</param>
        /// <param name="info">The structure: one byte, nonzero to delete.</param>
        /// <param name="size">The structure's size in bytes.</param>
        /// <returns>True on success (see the last Win32 error otherwise).</returns>
        [System.Runtime.InteropServices.DllImport("kernel32.dll", SetLastError = true)]
        private static extern bool SetFileInformationByHandle(
            Microsoft.Win32.SafeHandles.SafeFileHandle handle, int infoClass, ref byte info, uint size);
    }

    /// <summary>
    ///     A left-behind lock blocks writers, never readers: journal readers take no lock, as in Optuna, so a study can
    ///     still be loaded and inspected while a dead holder's lock waits out its grace period.
    /// </summary>
    [TestMethod]
    public void Lock_LeftBehindByADeadHolder_DoesNotBlockReaders()
    {
        var study = Optuna.CreateStudy(storage: NewStorage(), studyName: "read", sampler: new RandomSampler(60));
        study.Optimize(t => Objective(t, "w"), nTrials: 3);
        File.WriteAllText(_path + ".lock", "");

        var read = Task.Run(() => Optuna.LoadStudy("read", new JournalStorage(new JournalFileBackend(_path,
            new JournalFileOpenLock(_path, gracePeriod: null)))).Trials);

        ((IAsyncResult)read).AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(10)).Should().BeTrue("loading a study takes no lock");
        ShouldBePerfect(read.Result, 3);
        File.Delete(_path + ".lock");
    }

    /// <summary>
    ///     The two protocols share one lock path, so a symbolic-link holder and an open-file holder exclude each other in
    ///     both directions. This is what keeps a process that fell back to the open-file protocol (Windows without
    ///     Developer Mode) safe beside processes on Optuna's default symlink lock, Python's included.
    /// </summary>
    [TestMethod]
    public void SymlinkAndOpenFileLocks_OnOneJournal_ExcludeEachOther()
    {
        RequireSymlinks();
        File.WriteAllText(_path, "");
        foreach (var (holder, waiter) in new (BaseJournalFileLock Holder, BaseJournalFileLock Waiter)[]
        {
            (new JournalFileSymlinkLock(_path), new JournalFileOpenLock(_path)),
            (new JournalFileOpenLock(_path), new JournalFileSymlinkLock(_path)),
        })
        {
            holder.Acquire().Should().BeTrue();
            var acquire = Task.Run(() => waiter.Acquire());

            ((IAsyncResult)acquire).AsyncWaitHandle.WaitOne(TimeSpan.FromMilliseconds(500)).Should().BeFalse(
                $"{waiter.GetType().Name} must wait while {holder.GetType().Name} holds the path");
            holder.Release();
            ((IAsyncResult)acquire).AsyncWaitHandle.WaitOne(TimeSpan.FromSeconds(10)).Should().BeTrue();
            waiter.Release();
            File.Exists(_path + ".lock").Should().BeFalse();
        }
    }

    /// <summary>
    ///     Releasing a symbolic-link lock nobody holds raises Python's error (Optuna 5.0, verbatim: "Error: did not possess
    ///     lock"); the open-file lock's twin is pinned in <c>JournalStorageTests</c>.
    /// </summary>
    [TestMethod]
    public void SymlinkLock_ReleasedWithoutBeingHeld_RaisesPythonsError()
    {
        var release = () => new JournalFileSymlinkLock(_path).Release();

        release.Should().Throw<NumSharp.RuntimeError>().WithMessage("Error: did not possess lock");
    }

    // ------------------------------------------------------------------ several storages, one study

    /// <summary>
    ///     Sixteen workers, each with its own storage on the default lock, optimize one study at once: nobody dies
    ///     and the replayed study is perfect (contiguous numbers, every trial complete with all its steps).
    /// </summary>
    [TestMethod]
    public void SeparateStorages_OptimizingOneStudy_ProduceAPerfectStudy()
    {
        Optuna.CreateStudy(storage: NewStorage(), studyName: "s");

        var errors = RunConcurrently(16, w =>
        {
            var study = Optuna.LoadStudy("s", NewStorage(), sampler: new RandomSampler(100 + w), pruner: new NopPruner());
            study.Optimize(t => Objective(t, "w" + w), nTrials: 6);
        });

        errors.Should().BeEmpty("no worker may die of lock contention");
        ShouldBePerfect(LoadTrials("s"), 16 * 6);
    }

    /// <summary>
    ///     The offset cache under maximum pressure: a fresh backend per round, read in full by eight threads at once,
    ///     so every reader inserts every offset concurrently. A plain Dictionary lost entries here and threw
    ///     <c>KeyNotFoundException</c> (or detected the concurrent writes); every read must instead return every record.
    /// </summary>
    [TestMethod]
    public void SharedBackend_ReadInFullByManyThreads_KeepsItsOffsetCacheWhole()
    {
        const int records = 3000;
        File.WriteAllLines(_path, Enumerable.Range(0, records).Select(i =>
            $"{{\"op_code\":2,\"worker_id\":\"w\",\"study_id\":0,\"user_attr\":{{\"a{i}\":{i}}}}}"));

        var counts = new ConcurrentBag<int>();
        var errors = new List<Exception>();
        for (int round = 0; round < 10; round++)
        {
            var backend = new JournalFileBackend(_path, new JournalFileOpenLock(_path));
            errors.AddRange(RunConcurrently(8, _ => counts.Add(backend.ReadLogs(0).Count())));
            // A second pass seeks through the offsets the first pass cached, so a wrong entry would show here.
            errors.AddRange(RunConcurrently(8, r => counts.Add(backend.ReadLogs(r * 300).Count() + r * 300)));
        }

        errors.Should().BeEmpty();
        counts.Should().HaveCount(160).And.OnlyContain(c => c == records);
    }

    /// <summary>
    ///     Several <see cref="JournalStorage" />s over ONE backend instance (what a singleton backend under scoped or
    ///     transient storages produces) read while another writes, end to end: every reader converges on the
    ///     writer's study, and the study is perfect.
    /// </summary>
    [TestMethod]
    public void SharedBackend_ReadByManyStoragesWhileWritten_KeepsItsOffsetCacheWhole()
    {
        var backend = new JournalFileBackend(_path, new JournalFileOpenLock(_path));
        var writer = Optuna.CreateStudy(storage: new JournalStorage(backend), studyName: "s", sampler: new RandomSampler(7), pruner: new NopPruner());
        var readers = Enumerable.Range(0, 8).Select(_ => new JournalStorage(backend)).ToList();
        var errors = new ConcurrentQueue<Exception>();
        int done = 0;
        var threads = readers.Select(r => new Thread(() =>
        {
            while (Volatile.Read(ref done) == 0)
            {
                try
                {
                    _ = r.GetAllTrials(writer.StudyId, deepcopy: false);
                }
                catch (Exception e)
                {
                    errors.Enqueue(e);
                }
            }
        })).ToList();
        threads.ForEach(t => t.Start());

        try
        {
            writer.Optimize(t => Objective(t, "writer"), nTrials: 30);
        }
        finally
        {
            Volatile.Write(ref done, 1);
            threads.ForEach(t => t.Join());
        }

        errors.Should().BeEmpty();
        readers.Should().OnlyContain(r => r.GetAllTrials(writer.StudyId, deepcopy: false).Count == 30);
        ShouldBePerfect(LoadTrials("s"), 30);
    }

    // ------------------------------------------------------------------ several studies, one journal

    /// <summary>
    ///     Eight studies in one journal, created and optimized at once: half share one storage instance, half have
    ///     their own, and two of them run with <c>nJobs: 2</c>. Every study is complete and numbered from 0.
    /// </summary>
    [TestMethod]
    public void OneJournal_ManyStudies_OnSharedAndSeparateStorages_AreEachPerfect()
    {
        var shared = NewStorage();

        var errors = RunConcurrently(8, i =>
        {
            var storage = i % 2 == 0 ? shared : NewStorage();
            var study = Optuna.CreateStudy(storage: storage, studyName: "study-" + i, sampler: new RandomSampler(i), pruner: new NopPruner());
            study.Optimize(t => Objective(t, "s" + i), nTrials: 8, nJobs: i % 4 == 0 ? 2 : 1);
        });

        errors.Should().BeEmpty();
        NewStorage().GetAllStudies().Select(s => s.StudyName)
            .Should().BeEquivalentTo(Enumerable.Range(0, 8).Select(i => "study-" + i));
        for (int i = 0; i < 8; i++)
            ShouldBePerfect(LoadTrials("study-" + i), 8);
    }

    /// <summary>
    ///     Enqueued trials popped by eight separate storages at once: each runs exactly once, with its fixed
    ///     parameters, and none is left waiting. Popping is a WAITING→RUNNING claim decided by journal order.
    /// </summary>
    [TestMethod]
    public void EnqueuedTrials_PoppedBySeparateStorages_RunExactlyOnce()
    {
        var study0 = Optuna.CreateStudy(storage: NewStorage(), studyName: "s");
        for (int i = 0; i < 48; i++)
            study0.EnqueueTrial(new OrderedDict<string, object?> { ["x"] = i / 10.0 - 2.0, ["k"] = (long)i });

        var errors = RunConcurrently(8, w =>
            Optuna.LoadStudy("s", NewStorage(), pruner: new NopPruner()).Optimize(t => Objective(t, "w" + w), nTrials: 6));

        errors.Should().BeEmpty();
        var trials = LoadTrials("s");
        ShouldBePerfect(trials, 48);
        trials.Select(t => Convert.ToInt64(t.Params["k"])).Should().BeEquivalentTo(Enumerable.Range(0, 48).Select(i => (long)i),
            "every enqueued trial runs once, and its fixed parameters are kept");
    }

    // ------------------------------------------------------------------ several threads, one trial

    /// <summary>
    ///     Eight threads reporting the same steps of one trial: each step is written to the journal once, and the
    ///     stored value is the one written. Optuna's check-then-write let racing reporters write a step twice, so the
    ///     LAST value won although Optuna documents that the first one does (63% of contended steps in Python 5.0).
    /// </summary>
    [TestMethod]
    public void OneTrial_ConcurrentReportsOfTheSameSteps_WriteEachStepOnce()
    {
        var study = Optuna.CreateStudy(storage: NewStorage(), studyName: "s", pruner: new NopPruner());
        var trial = study.Ask();

        var errors = RunConcurrently(8, w =>
        {
            for (int step = 0; step < 20; step++)
            {
                trial.Report(w, step);
                _ = trial.ShouldPrune();
            }
        });

        errors.Should().BeEmpty();
        var records = File.ReadLines(_path)
            .Where(l => l.StartsWith("{\"op_code\":7,", StringComparison.Ordinal))
            .Select(l => (OrderedDict<string, object?>)OptunaSharp.Internal.PyJson.Loads(l)!)
            .ToList();
        records.Should().HaveCount(20, "the first report of a step wins; later ones are ignored with a warning");
        records.Select(r => Convert.ToInt64(r["step"])).Should().OnlyHaveUniqueItems();
        var stored = NewStorage().GetTrial(trial.TrialId).IntermediateValues;
        foreach (var r in records)
            stored[Convert.ToInt64(r["step"])].Should().Be(Convert.ToDouble(r["intermediate_value"]));
    }

    /// <summary>Eight threads suggesting the same parameter of one trial all get one value, sampled and written once.</summary>
    [TestMethod]
    public void OneTrial_ConcurrentSuggestionsOfOneName_SampleItOnce()
    {
        var study = Optuna.CreateStudy(storage: NewStorage(), studyName: "s", sampler: new RandomSampler(11));
        var trial = study.Ask();
        var values = new ConcurrentBag<double>();

        var errors = RunConcurrently(8, _ => values.Add(trial.SuggestFloat("x", -1, 1)));

        errors.Should().BeEmpty();
        values.Distinct().Should().ContainSingle();
        File.ReadLines(_path).Count(l => l.StartsWith("{\"op_code\":5,", StringComparison.Ordinal)).Should().Be(1);
    }

    /// <summary>
    ///     Eight threads setting one constraint key of one trial: it is written once, with the value of the call that
    ///     got there first, and the others are ignored with a warning, as Optuna documents. The "already set?" check
    ///     and the write used to interleave, so several calls could each write it.
    /// </summary>
    [TestMethod]
    public void OneTrial_ConcurrentConstraintsOfOneKey_WriteItOnce()
    {
        var study = Optuna.CreateStudy(storage: NewStorage(), studyName: "s");
        var trial = study.Ask();

        var errors = RunConcurrently(8, w => trial.SetConstraint("c", w));

        errors.Should().BeEmpty();
        var writes = File.ReadLines(_path)
            .Where(l => l.StartsWith("{\"op_code\":9,", StringComparison.Ordinal))
            .Select(l => (OrderedDict<string, object?>)OptunaSharp.Internal.PyJson.Loads(l)!)
            .Where(r => ((OrderedDict<string, object?>)r["system_attr"]!).ContainsKey("constraints:c"))
            .ToList();
        writes.Should().ContainSingle("the first call sets the constraint; later ones are ignored");
        double written = Convert.ToDouble(((OrderedDict<string, object?>)writes[0]["system_attr"]!)["constraints:c"]);
        Convert.ToDouble(NewStorage().GetTrial(trial.TrialId).SystemAttrs["constraints:c"]).Should().Be(written);
    }

    // ------------------------------------------------------------------ creating one study at once

    /// <summary>
    ///     Sixteen separate storages create one study name at once. With <c>loadIfExists</c>, every caller gets the one
    ///     study; without it, exactly one call creates it and the other fifteen get <see cref="DuplicatedStudyError" />.
    ///     The journal holds each study once. Creation is decided by journal order.
    /// </summary>
    [TestMethod]
    public void CreateStudy_RacingOnOneName_CreatesItOnce()
    {
        var ids = new ConcurrentBag<int>();
        var loadErrors = RunConcurrently(16, _ =>
            ids.Add(Optuna.CreateStudy(storage: NewStorage(), studyName: "shared", loadIfExists: true).StudyId));

        loadErrors.Should().BeEmpty();
        ids.Should().HaveCount(16).And.Subject.Distinct().Should().ContainSingle("every caller loads the one study");

        int created = 0, duplicates = 0;
        var createErrors = RunConcurrently(16, _ =>
        {
            try
            {
                Optuna.CreateStudy(storage: NewStorage(), studyName: "exclusive");
                Interlocked.Increment(ref created);
            }
            catch (DuplicatedStudyError)
            {
                Interlocked.Increment(ref duplicates);
            }
        });

        createErrors.Should().BeEmpty();
        created.Should().Be(1);
        duplicates.Should().Be(15);
        NewStorage().GetAllStudies().Select(s => s.StudyName).Should().BeEquivalentTo(new[] { "shared", "exclusive" });
    }

    // ------------------------------------------------------------------ helpers

    /// <summary>Opens a storage over the shared journal with the backend's default lock, as a user would.</summary>
    /// <returns>A new storage instance; each one replays the journal on construction.</returns>
    private JournalStorage NewStorage() => new(new JournalFileBackend(_path));

    /// <summary>Replays the journal from scratch and returns the study's trials.</summary>
    /// <param name="studyName">The study to load.</param>
    /// <returns>Deep copies of the trials in number order.</returns>
    private List<FrozenTrial> LoadTrials(string studyName) => Optuna.LoadStudy(studyName, NewStorage()).Trials;

    /// <summary>Three storage-backed parameters, <see cref="Steps" /> reports and one user attribute per trial.</summary>
    /// <param name="t">The running trial.</param>
    /// <param name="worker">Recorded as the <c>worker</c> user attribute, proving the trial ran to its end.</param>
    /// <returns>A value depending on every parameter.</returns>
    private static double Objective(Trial t, string worker)
    {
        double x = t.SuggestFloat("x", -10, 10);
        long k = t.SuggestInt("k", 0, 1000);
        for (int step = 0; step < Steps; step++)
            t.Report(x * x + step, step);
        t.SetUserAttr("worker", worker);
        return x * x + k * 1e-3;
    }

    /// <summary>The "perfect study" invariant: contiguous numbers from 0, all complete, every step and attribute present.</summary>
    /// <param name="trials">Trials from a fresh replay.</param>
    /// <param name="expected">The number of trials the workers ran.</param>
    private static void ShouldBePerfect(List<FrozenTrial> trials, int expected)
    {
        trials.Should().HaveCount(expected);
        trials.Select(t => t.Number).Should().Equal(Enumerable.Range(0, expected));
        trials.Should().OnlyContain(t => t.State == TrialState.Complete, "no trial may be left RUNNING or FAIL");
        trials.Should().OnlyContain(t => t.IntermediateValues.Count == Steps);
        trials.Should().OnlyContain(t => t.UserAttrs.ContainsKey("worker"));
    }

    /// <summary>Starts every worker on its own thread, releases them together, and collects what they threw.</summary>
    /// <param name="workers">The number of worker threads.</param>
    /// <param name="body">The work of one worker, given its index.</param>
    /// <returns>Every exception a worker let escape.</returns>
    private static List<Exception> RunConcurrently(int workers, Action<int> body)
    {
        var errors = new ConcurrentQueue<Exception>();
        using var barrier = new Barrier(workers);
        var threads = Enumerable.Range(0, workers).Select(w => new Thread(() =>
        {
            barrier.SignalAndWait();
            try
            {
                body(w);
            }
            catch (Exception e)
            {
                errors.Enqueue(e);
            }
        })).ToList();
        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join());
        return errors.ToList();
    }

    /// <summary>Each worker appends through its own backend and lock instance, as separate processes would.</summary>
    /// <param name="workers">The number of concurrent writers.</param>
    /// <param name="appendsPerWorker">Records each writer appends, one per call.</param>
    /// <param name="lockFactory">Builds one writer's lock.</param>
    /// <returns>Every append failure.</returns>
    private List<Exception> AppendConcurrently(int workers, int appendsPerWorker, Func<BaseJournalFileLock> lockFactory)
    {
        File.WriteAllText(_path, "");
        var failures = new ConcurrentQueue<Exception>();
        var errors = RunConcurrently(workers, w =>
        {
            var backend = new JournalFileBackend(_path, lockFactory());
            for (int i = 0; i < appendsPerWorker; i++)
            {
                var log = new OrderedDict<string, object?>
                {
                    ["op_code"] = 2,
                    ["worker_id"] = "w" + w,
                    ["study_id"] = 0,
                    ["user_attr"] = new OrderedDict<string, object?> { ["a" + i] = i },
                };
                try
                {
                    backend.AppendLogs(new[] { log });
                }
                catch (Exception e)
                {
                    failures.Enqueue(e);
                }
            }
        });
        return errors.Concat(failures).ToList();
    }

    /// <summary>Marks the test inconclusive where symbolic links cannot be created (Windows without Developer Mode).</summary>
    private void RequireSymlinks()
    {
        string probe = _path + ".symlink-probe";
        try
        {
            File.CreateSymbolicLink(probe, _path);
            File.Delete(probe);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Assert.Inconclusive("This machine cannot create symbolic links: " + e.Message);
        }
    }
}
