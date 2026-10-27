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
///         <b>When it switches.</b> <see cref="JournalFileSymlinkLock.Acquire" /> waits while the lock path
///         exists and throws only when creating the link failed with no lock path present. Contention never
///         causes a switch, and a switch is permanent for this instance. If the open-file protocol cannot
///         create its file either (a missing directory, a denied path), its exception propagates, so a
///         journal that cannot be locked at all still fails loudly.
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
                // The symlink protocol throws only when the link could not be created while no lock path
                // existed: this process cannot make symbolic links here. Contention waits inside Acquire
                // and never reaches this catch.
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
