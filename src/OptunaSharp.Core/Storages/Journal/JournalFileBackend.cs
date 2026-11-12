using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Internal;
using OptunaSharp.Logging;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     File storage class for Journal log backend.
///     Port of <c>optuna/storages/journal/_file.py: JournalFileBackend</c>.
///
///     The on-disk format is JSON-lines (one compact-separator <c>json.dumps</c> per line,
///     <c>\n</c>-terminated, UTF-8) — byte-compatible with Python Optuna, so a journal file
///     written by either side can be continued by the other with no external dependency on
///     either side (the plan's Tier N cross-language persistence claim).
/// </summary>
public class JournalFileBackend : BaseJournalBackend
{
    /// <summary>Suffix used for the journal's exclusivity lock file.</summary>
    internal const string LockFileSuffix = ".lock";
    /// <summary>Suffix used when atomically renaming a lock before deleting it.</summary>
    internal const string RenameFileSuffix = ".rename";

    private readonly string _filePath;
    private readonly BaseJournalFileLock _lock;

    /// <summary>
    ///     How many log lines apart the kept start offsets lie (plan 14; Rustuna 82b13d8). Python keeps one offset per line
    ///     (<c>_log_number_offset</c>); a 100,000-trial, 40-parameter journal has about 4.5 million lines, 120-250 MB of
    ///     dictionary. Here only the furthest offset reached and one offset per this many lines are kept.
    /// </summary>
    internal const int OffsetCheckpointInterval = 4096;

    // Where a read can start without scanning from the file's beginning. Python's dict is safe to share between
    // readers under the GIL; these fields are not, and one backend instance can be read by several JournalStorage
    // instances at once (a singleton backend under scoped or transient storages). Every access therefore takes
    // _offsetLock, and each reader carries its own byte position (see ReadLogs).
    // The furthest log number whose start offset a read has established, and that offset. A sync asks for exactly the
    // next unread log, which is this one, so it seeks straight there as Python's per-line dictionary lets it.
    private int _latestLogNumber;
    private long _latestLogOffset;
    // The start offset of log number k * OffsetCheckpointInterval at index k (log 0 starts at byte 0). A read for an
    // earlier log number (a new storage over a shared backend, a replay from the start) seeks to the nearest checkpoint
    // at or before it and skips forward, never more than OffsetCheckpointInterval - 1 lines.
    private readonly List<long> _offsetCheckpoints = new() { 0 };
    private readonly object _offsetLock = new();

    /// <summary>How many offset checkpoints are kept, for the tests that check the bookkeeping stays sparse.</summary>
    internal int OffsetCheckpointCount
    {
        get
        {
            lock (_offsetLock)
                return _offsetCheckpoints.Count;
        }
    }

    /// <summary>
    ///     The lock this backend acquires around every append: the caller's, or the
    ///     <see cref="JournalFileDefaultLock" />. Exposed so tests can check which one a constructor chose.
    /// </summary>
    internal BaseJournalFileLock LockObj => _lock;

    /// <summary>Opens or creates a UTF-8 JSON-lines journal using the supplied interprocess lock.</summary>
    /// <param name="filePath">Path of file to persist the log to.</param>
    /// <param name="lockObj">
    ///     Lock object for process exclusivity. An instance of
    ///     <see cref="JournalFileSymlinkLock" /> or <see cref="JournalFileOpenLock" /> can be
    ///     passed. When <c>null</c>, the backend uses Optuna's default, the symbolic-link protocol, and
    ///     switches to the <see cref="JournalFileOpenLock" /> protocol by itself if this process cannot
    ///     create symbolic links (see the remarks).
    /// </param>
    /// <exception cref="System.IO.IOException">The journal file cannot be opened or created.</exception>
    /// <exception cref="System.UnauthorizedAccessException">The process lacks access to the journal path.</exception>
    /// <remarks>
    ///     Python's backend defaults to <see cref="JournalFileSymlinkLock" />, which needs Developer Mode or
    ///     the symbolic-link privilege on Windows and a filesystem with symbolic links everywhere. Without
    ///     them, Python's first write fails with "A required privilege is not held by the client". The
    ///     default lock here keeps the symlink protocol wherever it works and moves to the open-file
    ///     protocol instead of failing; both take the same <c>&lt;filePath&gt;.lock</c> path, so this process
    ///     still excludes Python workers on the same journal. A lock passed explicitly is used as is: an
    ///     explicit <see cref="JournalFileSymlinkLock" /> still fails where Python's does.
    /// </remarks>
    public JournalFileBackend(string filePath, BaseJournalFileLock? lockObj = null)
    {
        _filePath = filePath;
        _lock = lockObj ?? new JournalFileDefaultLock(_filePath);
        if (!File.Exists(_filePath))
        {
            // Create a file if it does not exist.
            using var _ = new FileStream(_filePath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
        }
    }

    /// <summary>Reads complete journal records within the file-size snapshot taken when enumeration begins.</summary>
    /// <param name="logNumberFrom">Zero-based inclusive log number to read.</param>
    /// <returns>Lazy decoded log records; dispose the enumerator to close its file stream.</returns>
    /// <exception cref="System.IO.IOException">The journal cannot be read.</exception>
    /// <exception cref="System.UnauthorizedAccessException">The journal cannot be opened for reading.</exception>
    /// <exception cref="NumSharp.ValueError">A malformed record is followed by another record.</exception>
    /// <exception cref="System.InvalidCastException">A decoded record is not a JSON object.</exception>
    /// <remarks>
    ///     Safe to call from several threads, and from several <see cref="JournalStorage" /> instances that share
    ///     this backend: the offset cache is locked per access, and each enumeration tracks its own byte position.
    ///     Readers never take the journal lock, as in Optuna; a record that a writer is still appending is left
    ///     for the next read (see the size snapshot below).
    /// </remarks>
    public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
    {
        using var f = new FileStream(_filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        // Maintain remaining_log_size to allow writing by another process while reading the log.
        long remainingLogSize = new FileInfo(_filePath).Length;
        // Byte offset of the line being read. Python reads `_log_number_offset[log_number]` for it, but that entry
        // can be missing here while another reader of this backend is still recording it, so the position is
        // carried by this enumeration instead; it equals the recorded offset whenever both exist.
        // Python seeks to `log_number_from` when it has that line's offset and reads from byte 0 otherwise; this starts
        // at the nearest known line start at or before it instead (plan 14). Either way the lines before
        // `log_number_from` are skipped exactly as below, so the records yielded and the size accounting are the same.
        int logNumberStart;
        long position;
        lock (_offsetLock)
            (logNumberStart, position) = ReadStart(logNumberFrom);
        if (position > 0)
        {
            f.Seek(position, SeekOrigin.Begin);
            remainingLogSize -= position;
        }

        Exception? lastDecodeError = null;
        int logNumber = logNumberStart;
        foreach (var (lineBytes, endsWithNewline) in ReadRawLines(f))
        {
            long byteLen = lineBytes.Length;
            remainingLogSize -= byteLen;
            if (remainingLogSize < 0)
                break;
            if (lastDecodeError is not null)
                throw lastDecodeError;

            // Ensure that each line ends with line separators (\n, \r\n).
            if (!endsWithNewline)
            {
                lastDecodeError = new ValueError("Invalid log format.");
                logNumber++;
                continue;
            }
            long nextOffset = position + byteLen;
            position = nextOffset;
            if (logNumber < logNumberFrom)
            {
                // A line before the requested range is skipped undecoded, as Python skips it, and its successor's start
                // is recorded as Python records it.
                lock (_offsetLock)
                    RecordLineStart(logNumber + 1, nextOffset);
                logNumber++;
                continue;
            }

            OrderedDict<string, object?>? parsed = null;
            try
            {
                // The UTF-8 reader gives PyJson's value or error for the line's bytes (plan 10, item 2.7), without
                // decoding the line to a string first.
                parsed = (OrderedDict<string, object?>)PyJsonUtf8.Loads(lineBytes)!;
            }
            catch (ValueError err)
            {
                // A record that does not decode must not become a seek target, or the next sync would skip it: its
                // successor's start is recorded only after it decodes (Python records it first and deletes it here).
                lastDecodeError = err;
            }
            if (parsed is not null)
            {
                lock (_offsetLock)
                    RecordLineStart(logNumber + 1, nextOffset);
                yield return parsed;
            }
            logNumber++;
        }
    }

    /// <summary>The known line start a read for <paramref name="logNumberFrom" /> begins at; the caller holds <c>_offsetLock</c>.</summary>
    /// <param name="logNumberFrom">The first log number the read must yield.</param>
    /// <returns>
    ///     The log number of the line the read starts at and that line's byte offset: the furthest line start established
    ///     when <paramref name="logNumberFrom" /> is at or past it, otherwise the checkpoint at or before it (log 0 at byte 0
    ///     when nothing nearer is known).
    /// </returns>
    private (int LogNumber, long Offset) ReadStart(int logNumberFrom)
    {
        if (logNumberFrom >= _latestLogNumber)
            return (_latestLogNumber, _latestLogOffset);
        int index = logNumberFrom / OffsetCheckpointInterval;
        // Every checkpoint below the furthest line start has been recorded: the read that established it passed each.
        return index < _offsetCheckpoints.Count
            ? (index * OffsetCheckpointInterval, _offsetCheckpoints[index])
            : (0, 0L);
    }

    /// <summary>
    ///     Records that log number <paramref name="logNumber" /> starts at byte <paramref name="offset" />, once the line
    ///     before it has been read whole and decoded; the caller holds <c>_offsetLock</c>.
    /// </summary>
    /// <param name="logNumber">The log number whose line starts at <paramref name="offset" />.</param>
    /// <param name="offset">The byte offset of that line's first byte.</param>
    /// <remarks>
    ///     Readers sharing the backend record the same offsets for the same lines (the file only grows), so the order
    ///     they record in does not matter: the furthest start only moves forward, and a checkpoint is appended by
    ///     whichever reader passes it first. A reader that starts past a checkpoint starts at a line some reader reached by
    ///     reading through every earlier checkpoint, so checkpoints are appended in order.
    /// </remarks>
    private void RecordLineStart(int logNumber, long offset)
    {
        if (logNumber > _latestLogNumber)
        {
            _latestLogNumber = logNumber;
            _latestLogOffset = offset;
        }
        if (logNumber % OffsetCheckpointInterval == 0 && logNumber / OffsetCheckpointInterval == _offsetCheckpoints.Count)
            _offsetCheckpoints.Add(offset);
    }

    /// <summary>
    ///     Streams raw lines (including the <c>\n</c> terminator, like Python's binary-file line
    ///     iteration) so byte lengths match the offsets Python would compute.
    /// </summary>
    /// <param name="stream">Caller-owned readable stream; this iterator does not close it.</param>
    /// <returns>Raw line bytes plus whether each line ends with newline.</returns>
    /// <exception cref="System.IO.IOException">Reading from the input stream fails.</exception>
    /// <exception cref="System.ObjectDisposedException">The stream is closed.</exception>
    /// <remarks>
    ///     Reads <see cref="LineChunkSize" /> bytes at a time and cuts each chunk at its <c>\n</c> bytes, which yields the
    ///     lines reading byte by byte did, at a fraction of the cost (plan 10, item 2.7: a 17 MB journal's cold start).
    ///     Like the byte-by-byte read, it reads to the end of the file as the file is when each chunk is read; the
    ///     caller stops at the size it saw first, so a record another process is still appending is left for the next
    ///     read either way.
    /// </remarks>
    private static IEnumerable<(byte[] Line, bool EndsWithNewline)> ReadRawLines(Stream stream)
    {
        byte[] chunk = new byte[LineChunkSize];
        // A line that runs past the end of a chunk: its first part, until the chunk that holds its '\n' is read.
        byte[] pending = Array.Empty<byte>();
        int pendingLength = 0;
        int read;
        while ((read = stream.Read(chunk, 0, chunk.Length)) > 0)
        {
            int start = 0;
            while (start < read)
            {
                int newline = Array.IndexOf(chunk, (byte)'\n', start, read - start);
                if (newline < 0)
                {
                    // No terminator in the rest of the chunk: keep it, the line continues in the next chunk.
                    int tail = read - start;
                    if (pendingLength + tail > pending.Length)
                        Array.Resize(ref pending, Math.Max(pendingLength + tail, Math.Max(256, pending.Length * 2)));
                    Buffer.BlockCopy(chunk, start, pending, pendingLength, tail);
                    pendingLength += tail;
                    break;
                }

                int length = newline + 1 - start;
                var line = new byte[pendingLength + length];
                Buffer.BlockCopy(pending, 0, line, 0, pendingLength);
                Buffer.BlockCopy(chunk, start, line, pendingLength, length);
                pendingLength = 0;
                start = newline + 1;
                yield return (line, true);
            }
        }
        if (pendingLength > 0)
            yield return (pending.AsSpan(0, pendingLength).ToArray(), false);
    }

    /// <summary>
    ///     How many bytes <see cref="ReadRawLines" /> reads at a time: large enough to read a journal in few calls, small
    ///     enough to stay off the large object heap.
    /// </summary>
    private const int LineChunkSize = 64 * 1024;

    /// <summary>Appends compact JSON records while holding the interprocess lock, then flushes data to disk.</summary>
    /// <param name="logs">Ordered JSON-serializable records to append as one locked batch.</param>
    /// <exception cref="System.IO.IOException">Writing or flushing the journal fails.</exception>
    /// <exception cref="System.UnauthorizedAccessException">Journal or lock-file access is denied.</exception>
    /// <exception cref="NumSharp.RuntimeError">Releasing the acquired lock fails.</exception>
    public override void AppendLogs(IReadOnlyList<OrderedDict<string, object?>> logs)
    {
        using (JournalFileLockScope.Acquire(_lock))
        {
            var sb = new StringBuilder();
            for (int i = 0; i < logs.Count; i++)
            {
                if (i > 0)
                    sb.Append('\n');
                sb.Append(PyJson.DumpsCompact(logs[i]));
            }
            sb.Append('\n');

            using var f = new FileStream(_filePath, FileMode.Append, FileAccess.Write, FileShare.ReadWrite);
            var bytes = Encoding.UTF8.GetBytes(sb.ToString());
            f.Write(bytes, 0, bytes.Length);
            f.Flush(flushToDisk: true);  // Python: f.flush() + os.fsync(f.fileno()).
        }
    }
}

/// <summary>Port of <c>BaseJournalFileLock</c>.</summary>
public abstract class BaseJournalFileLock
{
    /// <summary>Blocks until the journal lock is acquired.</summary>
    /// <returns>True after acquiring exclusive ownership.</returns>
    /// <exception cref="System.IO.IOException">Lock creation fails for a reason other than an existing lock.</exception>
    /// <exception cref="System.UnauthorizedAccessException">Lock-file or symbolic-link creation is denied.</exception>
    public abstract bool Acquire();

    /// <summary>Releases the currently owned journal lock.</summary>
    /// <exception cref="NumSharp.RuntimeError">The lock cannot be renamed and deleted, including when ownership was lost.</exception>
    public abstract void Release();
}

/// <summary>Python's <c>get_lock_file</c> context manager as a disposable scope.</summary>
internal readonly struct JournalFileLockScope : IDisposable
{
    private readonly BaseJournalFileLock _lockObj;

    /// <summary>Records an already-acquired lock for release on scope exit.</summary>
    /// <param name="lockObj">Acquired lock whose release is owned by this scope.</param>
    private JournalFileLockScope(BaseJournalFileLock lockObj)
    {
        _lockObj = lockObj;
    }

    /// <summary>Acquires an interprocess lock and returns a scope that releases it.</summary>
    /// <param name="lockObj">Shared lock implementation to acquire.</param>
    /// <returns>A disposable scope owning one Release call.</returns>
    /// <exception cref="System.Exception">The lock implementation fails to acquire ownership.</exception>
    internal static JournalFileLockScope Acquire(BaseJournalFileLock lockObj)
    {
        lockObj.Acquire();
        return new JournalFileLockScope(lockObj);
    }

    /// <summary>Releases the acquired lock when the protected operation ends.</summary>
    /// <exception cref="NumSharp.RuntimeError">The file lock cannot be released.</exception>
    public void Dispose() => _lockObj.Release();
}

/// <summary>
///     Lock class for synchronizing processes for NFSv2 or later.
///     Port of <c>JournalFileSymlinkLock</c>.
///
///     On acquiring the lock, a symbolic link is created for an exclusive file. The file is
///     deleted when the lock is released. In NFS environments prior to NFSv3, use this instead
///     of <see cref="JournalFileOpenLock" />. On Windows without symlink privileges, use
///     <see cref="JournalFileOpenLock" />. Passed explicitly, this lock fails there, as Python's does; a
///     <see cref="JournalFileBackend" /> built without a lock switches to the open-file protocol by itself.
/// </summary>
public class JournalFileSymlinkLock : BaseJournalFileLock
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.storages.journal._file");

    private readonly string _lockTargetFile;
    private readonly string _lockFile;

    /// <summary>Grace period before an existing lock is forcibly released.</summary>
    public int? GracePeriod { get; }

    /// <param name="filepath">The path of the file whose race condition must be protected.</param>
    /// <param name="gracePeriod">Grace period before an existing lock is forcibly released.</param>
    /// <summary>Creates a journal lock policy with optional stale-lock recovery.</summary>
    /// <exception cref="NumSharp.ValueError">A non-null grace period is zero or negative.</exception>
    public JournalFileSymlinkLock(string filepath, int? gracePeriod = 30)
    {
        _lockTargetFile = filepath;
        _lockFile = filepath + JournalFileBackend.LockFileSuffix;
        if (gracePeriod is not null)
        {
            if (gracePeriod <= 0)
                throw new ValueError("The value of `grace_period` should be a positive integer.");
            if (gracePeriod < 3)
                OptunaWarnings.Warn("The value of `grace_period` might be too small. ");
        }
        GracePeriod = gracePeriod;
    }

    /// <summary>
    ///     Acquire a lock in a blocking way by creating a symbolic link of a file.
    /// </summary>
    /// <returns>True after acquiring exclusive ownership.</returns>
    /// <exception cref="System.IO.IOException">Lock creation fails for a reason other than an existing lock.</exception>
    /// <exception cref="System.UnauthorizedAccessException">Lock-file or symbolic-link creation is denied.</exception>
    public override bool Acquire()
    {
        double sleepSecs = 0.001;
        const double warningInterval = 10.0;
        long lastUpdateTimestamp = Environment.TickCount64;
        long lastWarningTimestamp = Environment.TickCount64;
        DateTime? mtime = null;
        while (true)
        {
            try
            {
                // A cheap fast path for a held lock. It carries Windows' ERROR_FILE_EXISTS code on every OS, which
                // JournalFileLockProtocol.IsAlreadyExists accepts everywhere for exactly this reason; the symlink call
                // below stays the atomic test when the path appears in between.
                if (File.Exists(_lockFile) || Directory.Exists(_lockFile))
                    throw new IOException("File exists", JournalFileLockProtocol.ErrorFileExistsHResult);
                File.CreateSymbolicLink(_lockFile, _lockTargetFile);
                return true;
            }
            catch (IOException e) when (JournalFileLockProtocol.IsAlreadyExists(e))
            {
                // Contention is read from the failed call's own error code, as Python tests
                // `err.errno == errno.EEXIST`. Re-checking the path here instead raced with the holder's release:
                // a lock released in between turned plain contention into a fatal error. Any other failure (no
                // symlink privilege on Windows, a missing directory) is not caught and propagates, as in Python.
                if ((Environment.TickCount64 - lastWarningTimestamp) / 1000.0 > warningInterval)
                {
                    _logger.Warning(
                        $"It is taking longer than {warningInterval} seconds to acquire " +
                        $"the lock file: {_lockFile} Retrying...");
                    lastWarningTimestamp = Environment.TickCount64;
                }

                if (GracePeriod is not null)
                {
                    DateTime currentMtime;
                    try
                    {
                        currentMtime = File.GetLastWriteTimeUtc(_lockFile);
                    }
                    catch (IOException)
                    {
                        continue;
                    }
                    if (currentMtime != mtime)
                    {
                        mtime = currentMtime;
                        lastUpdateTimestamp = Environment.TickCount64;
                    }

                    if ((Environment.TickCount64 - lastUpdateTimestamp) / 1000.0 > GracePeriod.Value)
                    {
                        _logger.Warning(
                            "The existing lock file has not been released " +
                            "for an extended period. Forcibly releasing the lock file.");
                        lastWarningTimestamp = Environment.TickCount64;
                        try
                        {
                            Release();
                            sleepSecs = 0.001;
                        }
                        catch (RuntimeError)
                        {
                            continue;
                        }
                    }
                }

                Thread.Sleep(TimeSpan.FromSeconds(sleepSecs));
                sleepSecs = Math.Min(sleepSecs * 2, JournalFileLockProtocol.MaxRetrySleepSeconds);
            }
        }
    }

    /// <summary>Release a lock by removing the symbolic link.</summary>
    /// <exception cref="NumSharp.RuntimeError">The lock cannot be renamed and deleted, including when ownership was lost.</exception>
    public override void Release()
    {
        string lockRenameFile = _lockFile + Guid.NewGuid() + JournalFileBackend.RenameFileSuffix;
        try
        {
            File.Move(_lockFile, lockRenameFile);
            File.Delete(lockRenameFile);
        }
        catch (IOException)
        {
            throw new RuntimeError("Error: did not possess lock");
        }
        catch (UnauthorizedAccessException)
        {
            throw new RuntimeError("Error: did not possess lock");
        }
    }
}

/// <summary>
///     Lock class for synchronizing processes for NFSv3 or later.
///     Port of <c>JournalFileOpenLock</c>.
///
///     On acquiring the lock, the lock file is created with exclusive-create semantics
///     (<c>O_CREAT | O_EXCL</c>). The file is deleted when the lock is released. This is the
///     recommended lock on Windows.
/// </summary>
public class JournalFileOpenLock : BaseJournalFileLock
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.storages.journal._file");

    private readonly string _lockFile;

    /// <summary>Grace period before an existing lock is forcibly released.</summary>
    public int? GracePeriod { get; }

    /// <param name="filepath">The path of the file whose race condition must be protected.</param>
    /// <param name="gracePeriod">Grace period before an existing lock is forcibly released.</param>
    /// <summary>Creates a journal lock policy with optional stale-lock recovery.</summary>
    /// <exception cref="NumSharp.ValueError">A non-null grace period is zero or negative.</exception>
    public JournalFileOpenLock(string filepath, int? gracePeriod = 30)
    {
        _lockFile = filepath + JournalFileBackend.LockFileSuffix;
        if (gracePeriod is not null)
        {
            if (gracePeriod <= 0)
                throw new ValueError("The value of `grace_period` should be a positive integer.");
            if (gracePeriod < 3)
                OptunaWarnings.Warn("The value of `grace_period` might be too small. ");
        }
        GracePeriod = gracePeriod;
    }

    /// <summary>Acquire a lock in a blocking way by creating a lock file.</summary>
    /// <returns>True after acquiring exclusive ownership.</returns>
    /// <exception cref="System.IO.IOException">Lock creation fails for a reason other than an existing lock.</exception>
    /// <exception cref="System.UnauthorizedAccessException">
    ///     Lock-file creation is denied: off Windows on the first denial, as in Optuna; on Windows once denials have
    ///     run unbroken for longer than <see cref="JournalFileLockProtocol.MaxDeniedStreakMilliseconds" /> (a lock file
    ///     being deleted in place is waited out, not reported).
    /// </exception>
    /// <remarks>
    ///     Waits out a lock path that is "delete pending" on Windows (see
    ///     <see cref="JournalFileLockProtocol.MaxDeniedStreakMilliseconds" />): a stale lock someone deletes by hand while
    ///     this waits no longer kills the waiter. This departs from Optuna, whose waiter raises <c>PermissionError</c> there.
    /// </remarks>
    public override bool Acquire()
    {
        double sleepSecs = 0.001;
        const double warningInterval = 10.0;
        long lastUpdateTimestamp = Environment.TickCount64;
        long lastWarningTimestamp = Environment.TickCount64;
        DateTime? mtime = null;
        // When the current unbroken run of access-denied failures began, or -1 when the last attempt was not denied.
        long deniedSince = -1;
        while (true)
        {
            try
            {
                // os.O_CREAT | os.O_EXCL | os.O_WRONLY.
                new FileStream(_lockFile, FileMode.CreateNew, FileAccess.Write, FileShare.None).Dispose();
                return true;
            }
            catch (UnauthorizedAccessException) when (OperatingSystem.IsWindows())
            {
                // ERROR_ACCESS_DENIED from the exclusive create: the lock path is "delete pending" (deleted in place, its
                // deleting handle not yet closed), or the path cannot be created at all. The first passes within
                // milliseconds, so the waiter waits as for a held lock; a run longer than the bound is the second, and is
                // reported as before. Releases never cause it (they rename the lock away), so Optuna's protocol, which
                // fails here, is unchanged for every lock a worker takes and releases.
                long now = Environment.TickCount64;
                if (deniedSince < 0)
                    deniedSince = now;
                else if (now - deniedSince > JournalFileLockProtocol.MaxDeniedStreakMilliseconds)
                    throw;

                Thread.Sleep(TimeSpan.FromSeconds(sleepSecs));
                sleepSecs = Math.Min(sleepSecs * 2, JournalFileLockProtocol.MaxRetrySleepSeconds);
            }
            catch (IOException e) when (JournalFileLockProtocol.IsAlreadyExists(e))
            {
                // The path exists normally again, so any run of denials is over.
                deniedSince = -1;

                // EEXIST from the exclusive create itself means another holder; see JournalFileSymlinkLock.Acquire
                // for why the path is not re-checked. Other failures are not caught and propagate, as in Python.
                if ((Environment.TickCount64 - lastWarningTimestamp) / 1000.0 > warningInterval)
                {
                    _logger.Warning(
                        $"It is taking longer than {warningInterval} seconds to acquire " +
                        $"the lock file: {_lockFile} Retrying...");
                    lastWarningTimestamp = Environment.TickCount64;
                }

                if (GracePeriod is not null)
                {
                    DateTime currentMtime;
                    try
                    {
                        currentMtime = File.GetLastWriteTimeUtc(_lockFile);
                    }
                    catch (IOException)
                    {
                        continue;
                    }
                    if (currentMtime != mtime)
                    {
                        mtime = currentMtime;
                        lastUpdateTimestamp = Environment.TickCount64;
                    }

                    if ((Environment.TickCount64 - lastUpdateTimestamp) / 1000.0 > GracePeriod.Value)
                    {
                        _logger.Warning(
                            "The existing lock file has not been released " +
                            "for an extended period. Forcibly releasing the lock file.");
                        lastWarningTimestamp = Environment.TickCount64;
                        try
                        {
                            Release();
                            sleepSecs = 0.001;
                        }
                        catch (RuntimeError)
                        {
                            continue;
                        }
                    }
                }

                Thread.Sleep(TimeSpan.FromSeconds(sleepSecs));
                sleepSecs = Math.Min(sleepSecs * 2, JournalFileLockProtocol.MaxRetrySleepSeconds);
            }
        }
    }

    /// <summary>Release a lock by removing the created file.</summary>
    /// <exception cref="NumSharp.RuntimeError">The lock cannot be renamed and deleted, including when ownership was lost.</exception>
    public override void Release()
    {
        string lockRenameFile = _lockFile + Guid.NewGuid() + JournalFileBackend.RenameFileSuffix;
        try
        {
            File.Move(_lockFile, lockRenameFile);
            File.Delete(lockRenameFile);
        }
        catch (IOException)
        {
            throw new RuntimeError("Error: did not possess lock");
        }
        catch (UnauthorizedAccessException)
        {
            throw new RuntimeError("Error: did not possess lock");
        }
    }
}

/// <summary>
///     The lock a <see cref="JournalFileBackend" /> uses when the caller passes none: Optuna's
///     <see cref="JournalFileSymlinkLock" /> protocol, until this process fails to create a symbolic link
///     beside the journal; from then on it uses the <see cref="JournalFileOpenLock" /> protocol.
/// </summary>
/// <remarks>
///     <para>
///         <b>Why.</b> Python's <c>JournalFileBackend</c> always defaults to the symlink lock. Creating a
///         symbolic link on Windows needs Developer Mode or <c>SeCreateSymbolicLinkPrivilege</c>, and FAT/exFAT
///         volumes and some network shares have no symbolic links. There the first journal write fails, in
///         Python as well as here before this class existed. This lock keeps Optuna's protocol wherever it
///         works and replaces that failure with the open-file protocol.
///     </para>
///     <para>
///         <b>Exclusion still holds across the switch.</b> Both protocols take the same
///         <c>&lt;journal&gt;.lock</c> path, and both creation calls fail while that path exists. A process
///         holding the lock as a symbolic link and one holding it as a regular file therefore exclude each
///         other, including Python workers. Both protocols release the same way, by renaming the lock path
///         and deleting the renamed file.
///     </para>
///     <para>
///         <b>When it switches.</b> <see cref="JournalFileSymlinkLock.Acquire" /> waits while the link creation
///         reports that the lock path exists (EEXIST, recognized by <see cref="JournalFileLockProtocol" />) and
///         throws for any other failure. Contention therefore never causes a switch, and a switch is permanent
///         for this instance. If the open-file protocol cannot create its file either (a missing directory, a
///         denied path), its exception propagates, so a journal that cannot be locked at all still fails loudly.
///     </para>
///     <para>
///         <b>A lock deleted in place.</b> On Windows, a lock path someone deletes by hand is "delete pending" for a
///         moment, and creating the symbolic link there is denied. That counts as a symbolic-link failure, so this
///         instance switches to the open-file protocol, which waits such a window out
///         (<see cref="JournalFileLockProtocol.MaxDeniedStreakMilliseconds" />). Both protocols use the one lock path,
///         so exclusion holds across the switch. Releases never cause it: they rename the lock away first.
///     </para>
///     <para>
///         <b>Footgun fixed in 2026-10.</b> Both protocols used to decide "contention or real error" by
///         re-checking the path after a failed create. A holder that released in between turned contention into
///         an exception, which killed 7–10 of 16 concurrent workers on one journal and made this lock switch
///         protocols under load (up to 31 of 32 instances). Python decides from the error code, as both
///         protocols now do.
///     </para>
/// </remarks>
internal sealed class JournalFileDefaultLock : BaseJournalFileLock
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.storages.journal._file");

    private readonly BaseJournalFileLock _symlinkLock;
    private readonly Func<BaseJournalFileLock> _openLockFactory;

    // Null until the first failed symbolic-link creation, then the open-file lock for good. Published
    // with Volatile so a thread that starts acquiring after the switch never retries the symlink.
    private BaseJournalFileLock? _openLock;

    /// <summary>Creates the default lock for one journal file.</summary>
    /// <param name="filepath">The journal path; the lock lives at <c>filepath + ".lock"</c>.</param>
    internal JournalFileDefaultLock(string filepath)
        : this(new JournalFileSymlinkLock(filepath), () => new JournalFileOpenLock(filepath))
    {
    }

    /// <summary>
    ///     Creates the default lock from its two protocols. Tests use it to simulate a process that cannot
    ///     create symbolic links; production code uses the path constructor.
    /// </summary>
    /// <param name="symlinkLock">The protocol tried first, normally a <see cref="JournalFileSymlinkLock" />.</param>
    /// <param name="openLockFactory">
    ///     Builds the fallback protocol, normally a <see cref="JournalFileOpenLock" /> on the same path. It is
    ///     called at most once, on the first failed symbolic-link creation.
    /// </param>
    /// <exception cref="ArgumentNullException"><paramref name="symlinkLock" /> or <paramref name="openLockFactory" /> is <c>null</c>.</exception>
    internal JournalFileDefaultLock(BaseJournalFileLock symlinkLock, Func<BaseJournalFileLock> openLockFactory)
    {
        _symlinkLock = symlinkLock ?? throw new ArgumentNullException(nameof(symlinkLock));
        _openLockFactory = openLockFactory ?? throw new ArgumentNullException(nameof(openLockFactory));
    }

    /// <summary>Whether this lock has switched to the open-file protocol.</summary>
    internal bool UsesOpenLock => Volatile.Read(ref _openLock) is not null;

    /// <summary>
    ///     Acquires the lock with the symbolic-link protocol, or with the open-file protocol once symbolic
    ///     links have failed here. Blocks while another process or thread holds the lock.
    /// </summary>
    /// <returns>True after acquiring exclusive ownership.</returns>
    /// <exception cref="System.IO.IOException">The open-file protocol cannot create the lock file for a reason other than an existing lock.</exception>
    /// <exception cref="System.UnauthorizedAccessException">The open-file protocol is denied access to the lock path.</exception>
    public override bool Acquire()
    {
        var openLock = Volatile.Read(ref _openLock);
        if (openLock is null)
        {
            try
            {
                return _symlinkLock.Acquire();
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                // The symlink protocol throws only for a failure other than "the lock path exists": this
                // process cannot make symbolic links here. Contention is recognized from the error code and
                // waits inside Acquire, so it never reaches this catch.
                openLock = SwitchToOpenLock(e);
            }
        }
        return openLock.Acquire();
    }

    /// <summary>Releases the lock through the protocol currently in use.</summary>
    /// <exception cref="NumSharp.RuntimeError">The lock cannot be renamed and deleted, including when ownership was lost.</exception>
    /// <remarks>
    ///     The switch only happens inside <see cref="Acquire" /> while this instance holds nothing, so the
    ///     protocol in use is the one that acquired. Both protocols release identically anyway.
    /// </remarks>
    public override void Release() => (Volatile.Read(ref _openLock) ?? _symlinkLock).Release();

    /// <summary>Installs the open-file protocol once, keeping the instance another thread may have installed first.</summary>
    /// <param name="cause">The symbolic-link failure, recorded in the debug log.</param>
    /// <returns>The open-file lock every later acquisition uses.</returns>
    private BaseJournalFileLock SwitchToOpenLock(Exception cause)
    {
        var created = _openLockFactory();
        var installed = Interlocked.CompareExchange(ref _openLock, created, null) ?? created;
        if (ReferenceEquals(installed, created))
            _logger.Debug(
                "Cannot create a symbolic link for the journal lock (" + cause.Message + "). " +
                "Using JournalFileOpenLock for this journal instead.");
        return installed;
    }
}

/// <summary>
///     Rules both journal file locks share: how a failed acquire is classified (the port of Python's
///     <c>err.errno == errno.EEXIST</c>, which tells "another holder owns the lock path" apart from every other way
///     an exclusive create or a symbolic-link creation can fail), and how long a waiter sleeps between attempts.
/// </summary>
/// <remarks>
///     The classification must come from the failed call's own exception. Looking at the path afterwards is a
///     race: the holder can release between the failure and the look, and the waiter then reports a fatal error
///     for plain contention. .NET carries the OS error in <see cref="Exception.HResult" />: Windows wraps Win32
///     codes as <c>0x8007xxxx</c>, while Unix builds put the raw <c>errno</c> there.
/// </remarks>
internal static class JournalFileLockProtocol
{
    /// <summary><c>HRESULT_FROM_WIN32(ERROR_FILE_EXISTS)</c>: <c>CreateFile(CREATE_NEW)</c> on an existing path.</summary>
    internal const int ErrorFileExistsHResult = unchecked((int)0x80070050);

    /// <summary><c>HRESULT_FROM_WIN32(ERROR_ALREADY_EXISTS)</c>: <c>CreateSymbolicLink</c> on an existing path.</summary>
    internal const int ErrorAlreadyExistsHResult = unchecked((int)0x800700B7);

    /// <summary><c>EEXIST</c> on Linux, macOS and the BSDs, which .NET stores as the HResult of a Unix IOException.</summary>
    internal const int EExist = 17;

    /// <summary>
    ///     Longest sleep between two attempts to take a held lock, in seconds; the sleep starts at 1 ms and doubles
    ///     up to this. Python doubles up to 1 s.
    /// </summary>
    /// <remarks>
    ///     Timing only: exclusion, the lock path and the grace-period recovery are unchanged, so Python and C#
    ///     workers still share a journal. With Python's 1 s cap, a waiter that had lost a few rounds slept up to a
    ///     second after the lock was freed, and the journal sat idle: 16 contending writers managed about 43
    ///     appends per second here, and this cap gave 2.5–4× that. The cost is at most about 60 failed attempts per
    ///     second per waiter (one existence check and one create), negligible next to an append's <c>fsync</c>.
    /// </remarks>
    internal const double MaxRetrySleepSeconds = 0.016;

    /// <summary>
    ///     Longest unbroken run of access-denied failures, in milliseconds, that <see cref="JournalFileOpenLock.Acquire" />
    ///     waits out on Windows before it treats the denial as a real permission problem and rethrows.
    /// </summary>
    /// <remarks>
    ///     <para>
    ///         <b>Why a denial can be transient.</b> On Windows, creating a file whose name is "delete pending" fails with
    ///         ERROR_ACCESS_DENIED (.NET's <see cref="UnauthorizedAccessException" />). A lock path becomes delete pending
    ///         only when something deletes it in place: <c>DeleteFile</c> marks the file and the name stays until the
    ///         deleting handle closes (until every handle closes where POSIX delete semantics are unavailable). Releases
    ///         never do that: C# and Python release by renaming the lock away first. A person or tool deleting a stale
    ///         lock by hand does, and so did a journal test (measured 2026-11-12: 5,070 denials in 40,000 contended
    ///         acquisitions with in-place deletes on an idle machine, 14,277 under full CPU load, 0 with rename-then-delete).
    ///     </para>
    ///     <para>
    ///         <b>Why bounded.</b> A directory the process may not write into, or a directory standing at the lock path,
    ///         denies every attempt for good; after this long the denial is rethrown, so such a journal still fails, one
    ///         second later than before. Off Windows a denial is always rethrown at once.
    ///     </para>
    ///     <para>
    ///         <b>Departure from Optuna.</b> Optuna's <c>os.open(O_CREAT | O_EXCL)</c> raises <c>PermissionError</c> in the
    ///         same window and catches only <c>FileExistsError</c>, so a Python waiter fails where this one waits.
    ///     </para>
    /// </remarks>
    internal const long MaxDeniedStreakMilliseconds = 1000;

    /// <summary>True when a lock-acquire failure means the lock path already exists, so the caller should wait.</summary>
    /// <param name="e">The exception the exclusive create or the symbolic-link creation threw.</param>
    /// <returns>
    ///     True for the two Windows "already exists" codes on every OS, and for <c>EEXIST</c> off Windows. The
    ///     Windows codes are accepted everywhere because <see cref="JournalFileSymlinkLock.Acquire" />'s fast path
    ///     throws a synthetic <see cref="ErrorFileExistsHResult" />; a raw <c>errno</c> is small and can never
    ///     equal them.
    /// </returns>
    internal static bool IsAlreadyExists(IOException e)
    {
        int hr = e.HResult;
        return hr == ErrorFileExistsHResult || hr == ErrorAlreadyExistsHResult
            || (!OperatingSystem.IsWindows() && hr == EExist);
    }
}

/// <summary>
///     Deprecated alias of <see cref="JournalFileBackend" />.
/// </summary>
[Obsolete("Deprecated in v4.0.0. This feature will be removed in v6.0.0. Use JournalFileBackend instead.")]
public class JournalFileStorage : JournalFileBackend
{
    /// <summary>Opens or creates a UTF-8 JSON-lines journal using the supplied interprocess lock.</summary>
    /// <param name="filePath">Journal file path; its parent directory must already exist.</param>
    /// <param name="lockObj">
    ///     Lock implementation; null selects the backend's default lock, the symbolic-link protocol with a
    ///     switch to the open-file protocol where symbolic links cannot be created.
    /// </param>
    /// <exception cref="System.IO.IOException">The journal file cannot be opened or created.</exception>
    /// <exception cref="System.UnauthorizedAccessException">The process lacks access to the journal path.</exception>
    public JournalFileStorage(string filePath, BaseJournalFileLock? lockObj = null)
        : base(filePath, lockObj)
    {
    }
}
