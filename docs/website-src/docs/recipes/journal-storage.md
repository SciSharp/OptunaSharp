# Journal-file storage

`JournalStorage` records every change to a study as an appended JSON line in a plain file. With it you
can run a **distributed optimization over a shared filesystem (NFS)** without standing up a database
server or a Redis instance — every worker appends to, and replays, the same log. It lives in **Core**
(namespace `OptunaSharp.Storages.Journal`), so no extra package is needed.

Because the format is line-delimited JSON over ordinary file I/O, the log is **byte-compatible with
Python Optuna's `JournalFileBackend`**: the same file can be resumed and appended to from either
runtime.

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Journal;

var storage = new JournalStorage(new JournalFileBackend("./optuna_journal_storage.log"));
var study = Optuna.CreateStudy(storage: storage, studyName: "example-study");

study.Optimize(trial =>
{
    double x = trial.SuggestFloat("x", -10, 10);
    return (x - 2) * (x - 2);
}, nTrials: 3);
```

The optimization above is too short to bother parallelizing, but the same script — pointed at a log
on a shared mount and started from several processes — collaborates on one study. Add
`loadIfExists: true` so each worker joins the existing study instead of failing on the duplicate name.

## File locking on Windows

Concurrent writers coordinate through a lock file. The default `JournalFileSymlinkLock` uses a symlink
and may report *"A required privilege is not held by the client"* on Windows, where creating symlinks
needs elevated rights. Switch to `JournalFileOpenLock`, which uses exclusive file creation instead:

```csharp
var backend = new JournalFileBackend(
    "./optuna_journal_storage.log",
    new JournalFileOpenLock("./optuna_journal_storage.log"));
var storage = new JournalStorage(backend);
```

Both locks accept an optional `gracePeriod` (seconds; default `30`) after which a stale lock left by a
crashed process is reclaimed.

## Redis instead of a file

The same `JournalStorage` runs over a Redis backend when workers don't share a filesystem — install a
client adapter such as `OptunaSharp.Storage.StackExchange.Redis`. See the
[Redis storage guide](../../integrations/storage/redis/index.md).

## Relationship to Optuna

| Optuna (Python) | OptunaSharp |
|-----------------|-------------|
| `JournalStorage(JournalFileBackend("x.log"))` | `new JournalStorage(new JournalFileBackend("x.log"))` |
| `journal.JournalFileOpenLock(path)` | `new JournalFileOpenLock(path)` |
| `JournalStorage(JournalRedisBackend("redis://…"))` | `new JournalStorage(new StackExchangeJournalRedisBackend("redis://…"))` |

**Divergence.** Journal *snapshots* are serialized as JSON rather than Python `pickle` bytes, so a
snapshot cannot be resumed across languages — but the *log* itself is fully cross-language. This is a
deliberate, documented difference.

## See also

[Storage](../storage.md) · [Saving/Resuming with RDB](rdb.md) ·
[Ablation study with brute force](ablation-study.md) (a distributed use of the journal) ·
the [Redis integration guide](../../integrations/storage/redis/index.md).
