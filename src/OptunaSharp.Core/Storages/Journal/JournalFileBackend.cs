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
    private readonly Dictionary<int, long> _logNumberOffset = new() { [0] = 0 };

    /// <param name="filePath">Path of file to persist the log to.</param>
    /// <param name="lockObj">
    ///     Lock object for process exclusivity. An instance of
    ///     <see cref="JournalFileSymlinkLock" /> or <see cref="JournalFileOpenLock" /> can be
    ///     passed. In a Windows environment symlink creation may need privileges — prefer
    ///     <see cref="JournalFileOpenLock" /> there (same guidance as Optuna's docs).
    /// </param>
    /// <summary>Opens or creates a UTF-8 JSON-lines journal using the supplied interprocess lock.</summary>
    /// <exception cref="System.IO.IOException">The journal file cannot be opened or created.</exception>
    /// <exception cref="System.UnauthorizedAccessException">The process lacks access to the journal path.</exception>
    public JournalFileBackend(string filePath, BaseJournalFileLock? lockObj = null)
    {
        _filePath = filePath;
        _lock = lockObj ?? new JournalFileSymlinkLock(_filePath);
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
    public override IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom)
    {
        using var f = new FileStream(_filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite);
        // Maintain remaining_log_size to allow writing by another process while reading the log.
        long remainingLogSize = new FileInfo(_filePath).Length;
        int logNumberStart = 0;
        if (_logNumberOffset.TryGetValue(logNumberFrom, out long offset))
        {
            f.Seek(offset, SeekOrigin.Begin);
            logNumberStart = logNumberFrom;
            remainingLogSize -= offset;
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
            if (!_logNumberOffset.ContainsKey(logNumber + 1))
                _logNumberOffset[logNumber + 1] = _logNumberOffset[logNumber] + byteLen;
            if (logNumber < logNumberFrom)
            {
                logNumber++;
                continue;
            }

            OrderedDict<string, object?>? parsed = null;
            try
            {
                parsed = (OrderedDict<string, object?>)PyJson.Loads(Encoding.UTF8.GetString(lineBytes))!;
            }
            catch (ValueError err)
            {
                lastDecodeError = err;
                _logNumberOffset.Remove(logNumber + 1);
            }
            if (parsed is not null)
                yield return parsed;
            logNumber++;
        }
    }

    /// <summary>
    ///     Streams raw lines (including the <c>\n</c> terminator, like Python's binary-file line
    ///     iteration) so byte lengths match the offsets Python would compute.
    /// </summary>
    /// <param name="stream">Caller-owned readable stream; this iterator does not close it.</param>
    /// <returns>Raw line bytes plus whether each line ends with newline.</returns>
    /// <exception cref="System.IO.IOException">Reading a byte from the input stream fails.</exception>
    /// <exception cref="System.ObjectDisposedException">The stream is closed.</exception>
    private static IEnumerable<(byte[] Line, bool EndsWithNewline)> ReadRawLines(Stream stream)
    {
        var buffer = new List<byte>(256);
        int b;
        while ((b = stream.ReadByte()) != -1)
        {
            buffer.Add((byte)b);
            if (b == (byte)'\n')
            {
                yield return (buffer.ToArray(), true);
                buffer.Clear();
            }
        }
        if (buffer.Count > 0)
            yield return (buffer.ToArray(), false);
    }

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
///     <see cref="JournalFileOpenLock" />.
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
                if (File.Exists(_lockFile) || Directory.Exists(_lockFile))
                    throw new IOException("File exists", unchecked((int)0x80070050));
                File.CreateSymbolicLink(_lockFile, _lockTargetFile);
                return true;
            }
            catch (IOException)
            {
                bool exists = File.Exists(_lockFile) || Directory.Exists(_lockFile);
                if (!exists)
                {
                    // Symlink creation failed for a non-EEXIST reason (e.g. missing privilege
                    // on Windows) — propagate, like Python re-raises non-EEXIST OSErrors.
                    throw;
                }

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
                sleepSecs = Math.Min(sleepSecs * 2, 1);
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
                // os.O_CREAT | os.O_EXCL | os.O_WRONLY.
                new FileStream(_lockFile, FileMode.CreateNew, FileAccess.Write, FileShare.None).Dispose();
                return true;
            }
            catch (IOException)
            {
                if (!File.Exists(_lockFile))
                {
                    // Non-EEXIST failure — propagate like Python.
                    throw;
                }

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
                sleepSecs = Math.Min(sleepSecs * 2, 1);
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
///     Deprecated alias of <see cref="JournalFileBackend" />.
/// </summary>
[Obsolete("Deprecated in v4.0.0. This feature will be removed in v6.0.0. Use JournalFileBackend instead.")]
public class JournalFileStorage : JournalFileBackend
{
    /// <summary>Opens or creates a UTF-8 JSON-lines journal using the supplied interprocess lock.</summary>
    /// <param name="filePath">Journal file path; its parent directory must already exist.</param>
    /// <param name="lockObj">Lock implementation; null selects a symbolic-link lock.</param>
    /// <exception cref="System.IO.IOException">The journal file cannot be opened or created.</exception>
    /// <exception cref="System.UnauthorizedAccessException">The process lacks access to the journal path.</exception>
    public JournalFileStorage(string filePath, BaseJournalFileLock? lockObj = null)
        : base(filePath, lockObj)
    {
    }
}
