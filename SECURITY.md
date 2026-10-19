# Security Policy

## Supported versions

OptunaSharp is pre-1.0 and under active development. Security fixes go to the latest release on NuGet
and to `master`.

| Version | Supported |
|---------|-----------|
| latest  | ✅        |
| older   | ❌        |

## Reporting a vulnerability

Report vulnerabilities **privately** — do not open a public issue.

- Preferred: [open a private security advisory](../../security/advisories/new) for this repository
  (Security → Report a vulnerability).
- Alternatively, contact a maintainer directly.

Include a description, the affected package(s) and version(s), reproduction steps, and the impact. We
aim to acknowledge reports within a few business days and will coordinate a fix and a disclosure
timeline with you.

## Scope

**In scope:** the code in this repository — the `OptunaSharp` package and its satellite packages
(storages, artifact stores, the gRPC storage proxy, the `optunasharp` CLI tool, visualization, the
ML integrations and the Python bridges) — and its build and release workflows.

**Out of scope — report these upstream:** NumSharp and `NumSharp.Interop.OpenBLAS` (including the
OpenBLAS binaries it ships), pythonnet, CPython and every Python package a bridge imports (`optuna`,
`optunahub` and its registry, `botorch`/`torch`, `cma`, `pymoo`, …), database drivers and EF Core
providers, the Redis clients, the cloud SDKs, and ASP.NET Core / gRPC.

## Security model

OptunaSharp is a library: it runs inside your process with your process's privileges, and it does
not sandbox the code it is given. These are the trust boundaries to keep in mind.

- **Objectives, callbacks and custom samplers are your code.** `Study.Optimize` calls them in-process,
  on thread-pool threads when `nJobs > 1`.
- **The Python bridges embed CPython.** `OptunaSharp.Interop.pythonnet`, `.Interop.BoTorch`,
  `.Interop.PyCma` and `.Interop.Pymoo` load a CPython interpreter into the .NET process (the one named
  by `PYTHONNET_PYDLL`, otherwise the `python` found through `$PYTHON` or `PATH`) and import packages
  from its environment. Anything
  those packages execute runs inside your process — use a dedicated, trusted Python environment.
- **OptunaHub modules are remote code.** `OptunaHub.LoadModule` downloads a package from a GitHub
  repository (by default `optuna/optunahub-registry` at `main`) and executes it, exactly like Python's
  `optunahub.load_module`. Load only from repositories you trust, pin `gitRef` to a reviewed commit, or
  vendor the package and use `OptunaHub.LoadLocalModule`.
- **The gRPC storage proxy is unauthenticated cleartext by default.** `GrpcProxyServer.RunGrpcProxyServer`
  and `MakeServer` listen with cleartext HTTP/2 on `localhost:13000`, like Optuna's
  `run_grpc_proxy_server` (`add_insecure_port`). Any client that can reach the port can read, change and
  delete every study in the backing storage. Keep it on localhost or a private network, or add TLS and
  authentication through the `configureBuilder` hook (Kestrel HTTPS, ASP.NET Core authentication) and
  connect with `new GrpcStorageProxy("https://…", channelOptions)`.
- **Storages and artifact stores have no access control of their own.** Journal files, SQLite
  databases and `FileSystemArtifactStore` folders are protected only by filesystem permissions;
  PostgreSQL, MySQL, Redis and the cloud artifact stores only by their own credentials. Anyone who can
  write to a storage can change trial history — and with it what the samplers learn from.
- **Storage URLs and connection strings are secrets.** They usually carry credentials: keep them out of
  source control, shared command lines, logs and crash reports, and pass them through environment
  variables or a secret store.
- **Persisted data is JSON, never pickle.** Studies, trials, attributes and sampler state are stored as
  JSON (journal, RDB, Redis); OptunaSharp uses no pickle or type-embedding serializer, so opening an
  untrusted storage cannot run code. Its contents are still attacker-chosen data, such as oversized
  attributes.
- **Logs contain trial parameters and values.** At `INFO`, every finished trial is logged with its
  parameters and objective value, as in Optuna. If those are sensitive, lower the verbosity
  (`OptunaLogging.SetVerbosity(LogLevel.Warning)`) or route logging to a protected sink.
- **Strong-name signatures are identity, not authenticity.** Every assembly is signed with the
  publicly available `Open.snk` key that NumSharp and the .NET open-source projects share, so anyone can
  produce the same signature. Rely on nuget.org's repository signature and the release tags instead.
- **The interop test suite runs Python.** `OptunaSharp.Tests.Interop` embeds CPython and executes Optuna
  and the bridged packages; run it only against trusted Python environments.
