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
    public abstract IEnumerable<OrderedDict<string, object?>> ReadLogs(int logNumberFrom);

    /// <summary>Append logs to the backend.</summary>
    /// <param name="logs">A list that contains json-serializable logs.</param>
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
    void SaveSnapshot(byte[] snapshot);

    /// <summary>Load snapshot from the backend, or <c>null</c> if not found.</summary>
    byte[]? LoadSnapshot();
}

/// <summary>
///     Deprecated alias of <see cref="BaseJournalBackend" />.
/// </summary>
[Obsolete("Deprecated in v4.0.0. This feature will be removed in v6.0.0. Use BaseJournalBackend instead.")]
public abstract class BaseJournalLogStorage : BaseJournalBackend
{
}
