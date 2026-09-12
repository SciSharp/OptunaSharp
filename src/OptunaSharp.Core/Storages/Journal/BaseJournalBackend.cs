using System;
using System.Collections.Generic;
using OptunaSharp.Collections;

namespace OptunaSharp.Storages.Journal;

/// <summary>
///     Base class for Journal storages. Port of
///     <c>optuna/storages/journal/_base.py: BaseJournalBackend</c>.
///
///     Storage classes implementing this base class must guarantee process safety. This means,
///     multiple processes might concurrently call <see cref="ReadLogs" /> and
///     <see cref="AppendLogs" />. If the backend storage does not internally support mutual
///     exclusion mechanisms, such as locks, you might want to use
///     <see cref="JournalFileSymlinkLock" /> or <see cref="JournalFileOpenLock" /> for creating a
///     critical section.
/// </summary>
public abstract class BaseJournalBackend
{
    /// <summary>
    ///     Read logs with a log number greater than or equal to <paramref name="logNumberFrom" />.
    ///     If <paramref name="logNumberFrom" /> is 0, read all the logs.
    /// </summary>
    /// <param name="logNumberFrom">Zero-based inclusive log-record number; zero reads the complete journal.</param>
    /// <returns>A sequence of decoded records in append order, starting at the requested record number.</returns>
    /// <exception cref="System.Exception">The journal backend cannot read or decode its records; lazy backends can fail during enumeration.</exception>
    public abstract IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom);

    /// <summary>Append logs to the backend.</summary>
    /// <param name="logs">A list that contains json-serializable logs.</param>
    /// <exception cref="System.Exception">The backend cannot serialize or durably append the records.</exception>
    public abstract void AppendLogs(IReadOnlyList<OrderedDict<string, object?>> logs);
}

/// <summary>
///     Optional base interface for Journal storages. Port of <c>BaseJournalSnapshot</c>
///     (a mixin class in Python; an interface in C#).
///
///     Storage classes implementing this may work faster when constructing the internal state
///     from the large amount of logs.
/// </summary>
public interface IBaseJournalSnapshot
{
    /// <summary>Save snapshot to the backend.</summary>
    /// <param name="snapshot">Opaque serialized replay state to persist atomically according to the backend contract.</param>
    /// <exception cref="System.Exception">The snapshot backend cannot persist the bytes.</exception>
    void SaveSnapshot(byte[] snapshot);

    /// <summary>Load snapshot from the backend, or <c>null</c> if not found.</summary>
    /// <returns>Snapshot bytes, or null when no snapshot is available.</returns>
    /// <exception cref="System.Exception">The snapshot backend fails to read the saved state.</exception>
    byte[]? LoadSnapshot();
}

/// <summary>
///     Deprecated alias of <see cref="BaseJournalBackend" />.
/// </summary>
[Obsolete("Deprecated in v4.0.0. This feature will be removed in v6.0.0. Use BaseJournalBackend instead.")]
public abstract class BaseJournalLogStorage : BaseJournalBackend
{
}
