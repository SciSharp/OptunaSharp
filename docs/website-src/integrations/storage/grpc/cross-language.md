# Python and C# over gRPC

OptunaSharp's gRPC client can use Python Optuna's server, and Python's client can use an OptunaSharp server. Both speak the `optuna.StorageService` protobuf contract. This lets each worker keep its native objective and ML stack while sharing stored studies.

## Python server, C# worker

The Python environment needs Optuna with its gRPC storage API and `grpcio`. This local example uses Python's SQLite storage, so it needs no separate database server:

```python
from optuna.storages import get_storage, run_grpc_proxy_server

run_grpc_proxy_server(
    get_storage("sqlite:///optuna.db"),
    host="localhost", port=13000)
```

Install `OptunaSharp.Storage.Grpc` in the C# worker, then connect:

```csharp
using OptunaSharp;
using OptunaSharp.Storages.Grpc;

using var storage = new GrpcStorageProxy("localhost", 13000);
storage.WaitServerReady(TimeSpan.FromSeconds(10));
var study = Optuna.CreateStudy(
    storage: storage, studyName: "distributed", loadIfExists: true);
study.Optimize(trial => Math.Pow(trial.SuggestFloat("x", -10, 10), 2), nTrials: 20);
```

For separate machines, bind the server to its worker-facing interface and use its reachable host name in the client.

## C# server, Python worker

Install `OptunaSharp.Storage.Grpc` and `OptunaSharp.Storage.Sqlite` in the C# server:

```csharp
using OptunaSharp.Storages;
using OptunaSharp.Storages.Grpc;

using var backend = new RDBStorage("sqlite:///optuna.db");
GrpcProxyServer.RunGrpcProxyServer(
    StorageFunctions.GetStorage(backend), "localhost", 13000);
```

Then use Python's client:

```python
import optuna
from optuna.storages import GrpcStorageProxy

storage = GrpcStorageProxy(host="localhost", port=13000)
try:
    storage.wait_server_ready(timeout=10)
    study = optuna.create_study(
        storage=storage, study_name="distributed", load_if_exists=True)
    study.optimize(lambda trial: trial.suggest_float("x", -10, 10) ** 2,
                   n_trials=20)
finally:
    storage.close()
```

Only the server needs the database provider or access to its database file. The clients need network access to the gRPC endpoint. Sampler/pruner settings and objective code remain local to each client.

## What the compatibility boundary contains

The vendored protocol keeps Optuna's protobuf package, RPC names, messages, and field numbers. Its extra `option csharp_namespace` is a C# code-generation setting, not a wire-format change. Describing the protocol file as literally unmodified would therefore be inaccurate.

Numeric parameters and values travel as protobuf doubles. Attribute JSON uses Optuna-compatible JSON encoding, distributions use the matching distribution representation, and datetimes use the upstream six-digit fractional-second string format. The proxy preserves multi-objective trial values and nonfinite intermediate values that its backing storage supports.

Expected storage exceptions map through gRPC statuses and back into typed exceptions at the matching RPCs:

| Condition | Status | Client exception |
|---|---|---|
| Missing study/trial | `NOT_FOUND` | `KeyError` |
| Duplicate named study | `ALREADY_EXISTS` | `DuplicatedStudyError` |
| Update of a finished trial | `FAILED_PRECONDITION` | `UpdateFinishedTrialError` |
| Invalid parameter/distribution operation | `INVALID_ARGUMENT` | `ValueError` |

The wire format does not transfer sampler random state, executing objectives, artifact file contents, or heartbeat lifecycle operations. Matching stored history is not a claim that separately configured Python and C# workers will suggest identical future trials.

### Dictionary order (a deliberate difference)

Optuna's Python client rebuilds every trial from protobuf maps, and Python protobuf (upb) iterates a map in its own hash-table order. So through Optuna's `GrpcStorageProxy`, `trial.params`, `trial.distributions` and the attribute dictionaries come back in that hash order, not in the order the parameters were suggested: nine floats suggested as `u0, y1, depth2, x3, lr4, z5, a, b, c` read back as `depth2, u0, x3, b, y1, z5, lr4, c, a` (Optuna 5.0.0, grpcio 1.78.0, protobuf 6.33.5), whichever server answers. The C# client keeps the order the server sends: insertion order from a C# server, as every other OptunaSharp storage keeps it.

Two samplers read that order. `QMCSampler` assigns Sobol dimensions in the order of the first trial's distributions, and `BruteForceSampler` builds its tree from it. So a seeded QMC or BruteForce study run through Optuna's proxy can differ from the same study run in memory, in Optuna itself; the same study through the C# client equals its in-memory run. Reproducing upb's internal layout was considered and declined (2026-11-12): it is an implementation detail of Python protobuf that can change with its version. The other samplers are unaffected.

## Transport and verification scope

The default constructors on both sides use cleartext HTTP/2. HTTPS is available through the .NET [channel](client.md) and [host](server.md) configuration paths, but it must match the other endpoint's transport and trust configuration. An HTTPS C# client cannot communicate with an unchanged insecure Python endpoint. The default examples do not configure authentication.

The [cross-language test suite](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/GrpcStorageProxyInteropTests.cs) exercises both directions with real TCP connections, mixed parameter types, multi-objective data, and nonfinite values. It uses in-memory backing storage, and its cross-runtime study snapshots compare timestamp nullness rather than asserting clock equality. Tests self-skip when Python, Optuna, or gRPC dependencies are unavailable.

The [localhost C# suite](https://github.com/SciSharp/OptunaSharp/blob/main/test/OptunaSharp.Tests.Interop/GrpcLocalhostInteropTests.cs) also verifies two independent clients observing the same study and subsequent updates. Those fixtures do not establish coverage for every TLS setup, authentication policy, backend, or combination of Python package versions.

Protocol evidence: [vendored api.proto](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Storage.Grpc/Protos/api.proto) and [trial conversion](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Storage.Grpc/Storages/Grpc/GrpcProto.cs).
