# Cross-language (Python ↔ C#)

The gRPC proxy is generated from **Optuna's own `api.proto`, vendored verbatim**, so an OptunaSharp
runtime and a Python Optuna runtime interoperate on the wire in **both directions**. This is verified
live in the interop suite (a C# server driven by a Python client, and a Python server driven by a C#
client, including multi-objective studies and NaN/±inf values).

## Both directions

**A Python server, OptunaSharp (C#) clients** — run the server in Python, connect from C#:

```python
# Python
from optuna.storages import run_grpc_proxy_server, get_storage
run_grpc_proxy_server(get_storage("sqlite:///optuna.db"), host="0.0.0.0", port=13000)
```

```csharp
// C#
using var storage = new GrpcStorageProxy(host: "python-host", port: 13000);
var study = Optuna.CreateStudy(storage: storage, studyName: "dist", loadIfExists: true);
```

**An OptunaSharp (C#) server, Python clients** — run the server in C#, connect from Python:

```csharp
// C#
GrpcProxyServer.RunGrpcProxyServer(StorageFunctions.GetStorage(backend), host: "0.0.0.0", port: 13000);
```

```python
# Python
from optuna.storages import GrpcStorageProxy
storage = GrpcStorageProxy(host="csharp-host", port=13000)
study = optuna.create_study(storage=storage, study_name="dist", load_if_exists=True)
```

## What makes it byte-compatible

- **The wire contract is identical.** `OptunaSharp.Storage.Grpc` compiles the same `api.proto` — same
  package, messages, field numbers, and `StorageService` RPCs — that Python Optuna's grpcio code does.
  The only addition is a wire-neutral `option csharp_namespace` (so the generated .NET types don't
  occupy the global `Optuna` namespace); it changes the generated C# namespace only, never the bytes on
  the wire.
- **The strings inside the messages match too.** Attribute values are encoded with the same `json.dumps`
  form, distributions with the same `distribution_to_json`, and datetimes with Optuna's
  `"%Y-%m-%d %H:%M:%S.%f"` — so a `FrozenTrial` reconstructed on either side is identical.

## Error mapping

Storage exceptions travel as gRPC status codes and are re-raised as the matching typed exception on the
other side — the same mapping Optuna uses, so behaviour is identical whether the client or server is
Python or C#:

| Storage condition | gRPC status | Client raises |
|---|---|---|
| No such study / trial | `NOT_FOUND` | `KeyError` |
| Duplicate study name | `ALREADY_EXISTS` | `DuplicatedStudyError` |
| Write to a finished trial | `FAILED_PRECONDITION` | `UpdateFinishedTrialError` |
| Invalid argument | `INVALID_ARGUMENT` | `ValueError` |

## Transport and security

The default transport is **insecure cleartext HTTP/2 (h2c)**, matching Optuna's `insecure_channel` /
`add_insecure_port` — this is what makes the default Python ↔ C# interop work out of the box. Keep it
behind your own network boundary.

For a secure deployment, configure TLS on both ends: on the C# client use an `https://` address with TLS
credentials (see [Client](client.md)), and on the C# server add a TLS endpoint via `configureBuilder`
(see [Server](server.md)). A TLS channel no longer interoperates with an *insecure* Python server, so
secure both sides together.

## See also

- [gRPC overview](index.md)
- [Client — `GrpcStorageProxy`](client.md)
- [Server — `GrpcProxyServer`](server.md)
