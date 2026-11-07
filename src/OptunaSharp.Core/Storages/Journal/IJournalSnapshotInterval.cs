namespace OptunaSharp.Storages.Journal;

/// <summary>
///     A snapshot-capable journal backend that tells <see cref="JournalStorage" /> how many study or trial ids apart to
///     save a replay snapshot when the caller does not choose. OptunaSharp only: Optuna snapshots every 100 ids
///     whatever the backend, and has no such interface.
/// </summary>
/// <remarks>
///     <para>
///         A snapshot pays off when replaying the records it replaces costs more than restoring it, and it costs the
///         worker that saves it a serialization of the whole study, under the storage's lock. Which side wins depends on
///         how the backend reads its records, which only the backend knows. Measured on 2026-11-07 at 20,000 trials on
///         a Redis journal (plan 10, item 2.2): reading the records in batches, a cold start took 1.30 s without any
///         snapshot and 1.19 s with one every 100 ids, while those snapshots made every trial 48% slower to write;
///         reading one record per round trip (cluster mode), the same cold start took 53.8 s without a snapshot and
///         1.17 s with one every 100 ids. The Redis backends therefore answer 1000 on a standalone server and 100 in
///         cluster mode.
///     </para>
///     <para>
///         <see cref="JournalStorage" /> reads <see cref="DefaultSnapshotInterval" /> once, in its constructor, and only
///         when that constructor's <c>snapshotInterval</c> argument is <c>null</c>. A backend that stores snapshots
///         (<see cref="IBaseJournalSnapshot" />) without implementing this interface gets Optuna's 100. Implementing it
///         on a backend without snapshots changes nothing.
///     </para>
/// </remarks>
public interface IJournalSnapshotInterval
{
    /// <summary>
    ///     How many study or trial ids apart a <see cref="JournalStorage" /> over this backend saves a snapshot when its
    ///     constructor is not given one: the worker that creates a study or trial whose id is a multiple of this value
    ///     saves the whole replay state. Must be at least 1.
    /// </summary>
    /// <remarks>
    ///     A value below 1 makes the <see cref="JournalStorage" /> constructor throw <see cref="System.ArgumentException" />
    ///     rather than snapshot on every id (1) or never; a backend that should never snapshot should not implement
    ///     <see cref="IBaseJournalSnapshot" />.
    /// </remarks>
    int DefaultSnapshotInterval { get; }
}
