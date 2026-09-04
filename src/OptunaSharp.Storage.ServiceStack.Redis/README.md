# OptunaSharp.Storage.ServiceStack.Redis

Redis journal-log backend for [**OptunaSharp**](https://github.com/SciSharp/OptunaSharp) on
[**ServiceStack.Redis**](https://github.com/ServiceStack/ServiceStack.Redis). Part of the
[SciSharp STACK](https://github.com/SciSharp).

`ServiceStackJournalRedisBackend` persists an Optuna study's operation log to a Redis server. It is
byte-compatible with Python Optuna's `JournalRedisBackend` and with the sibling
`StackExchange`/`NRedisStack` backends — a study written by any of them can be read by the others over
the same Redis.

## ⚠️ License

**ServiceStack.Redis is commercially licensed.** Its free tier is **request-quota-limited** (and
requires accepting ServiceStack's license terms). Unlike the MIT
`OptunaSharp.Storage.StackExchange.Redis` and `OptunaSharp.Storage.NRedisStack` backends, real
production use of this package needs a [ServiceStack license](https://servicestack.net). Choose this
package only if your application already standardizes on ServiceStack.Redis.

## Usage

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Journal;

var backend = new ServiceStackJournalRedisBackend("redis://localhost:6379");
var study = Optuna.CreateStudy(storage: new JournalStorage(backend));
```

`redis://` and `rediss://` (TLS) URLs are accepted (unix sockets are not supported by
ServiceStack.Redis). An internal constructor accepts an existing `IRedisClientsManager`.

## License

OptunaSharp is MIT © SciSharp STACK Team — **but this package depends on the commercial
ServiceStack.Redis** (see above).
