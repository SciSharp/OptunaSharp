# Command-line tool

**`OptunaSharp.Cli`** exposes the `optunasharp` command for creating and inspecting studies and
coordinating ask/tell optimization from scripts or another process. It is a .NET global tool,
not a Python wrapper. The tool targets .NET 8 and .NET 10 and bundles Core, the SQLite backend,
and the StackExchange.Redis journal adapter.

```bash
dotnet tool install --global OptunaSharp.Cli
optunasharp --help
```

From a source checkout, the equivalent development entry point is:

```bash
dotnet run --project src/OptunaSharp.Cli/OptunaSharp.Cli.csproj -f net8.0 -- --help
```

## Storage selection

Pass `--storage` to each invocation, or set `OPTUNA_STORAGE`. An explicit argument takes
precedence over the environment variable.

| Input | Resolved backend |
|---|---|
| URL beginning with `redis` (including `rediss`) | `JournalStorage` with `StackExchangeJournalRedisBackend` |
| Existing file path | `JournalStorage` with `JournalFileBackend` |
| Other input, such as `sqlite:///study.db` | SQLite `RDBStorage`; invalid input fails with a CLI usage error |

An existing SQLite file should be supplied as `sqlite:///study.db`, not just `study.db`, because
the plain existing-file path selects the journal backend. To create a new journal file, explicitly
choose `--storage-class JournalFileBackend`. The file backend's default uses symlink locking;
see [journal locking](../docs/storage.md#file-locking) for Windows permissions and the library's
alternative lock implementation.

`--storage-class` accepts `RDBStorage`, `JournalFileBackend`, `JournalRedisBackend`, and the
deprecated `JournalFileStorage` / `JournalRedisStorage` aliases. Redis class names select the
StackExchange adapter. The CLI does not resolve EF Core, gRPC, or the other Redis client adapters;
use their [.NET APIs](storage/index.md) for those integrations. Its URL resolution is also separate
from Core's `StorageFunctions.GetStorage(string)`.

## Ask/tell from an external training process

This Bash example creates a study and obtains a trial's sampled parameters as JSON:

```bash
optunasharp create-study --study-name quadratic --storage sqlite:///study.db

optunasharp ask --study-name quadratic --storage sqlite:///study.db \
  --sampler RandomSampler --sampler-kwargs '{"seed": 0}' \
  --search-space '{"x": {"name": "FloatDistribution", "attributes": {"low": -10.0, "high": 10.0, "log": false, "step": null}}}'
```

Read the returned trial `number` and `params`, evaluate those parameters in your training process,
then report that trial's result:

```bash
# Replace 0 with the number from ask, and 4.0 with the measured objective value.
optunasharp tell --study-name quadratic --storage sqlite:///study.db --trial-number 0 --values 4.0
optunasharp trials --study-name quadratic --storage sqlite:///study.db -f json --flatten
optunasharp best-trial --study-name quadratic --storage sqlite:///study.db
```

In PowerShell, put each command on one line or use PowerShell continuation syntax instead of Bash
backslashes. `ask` allocates a trial; it does not launch a training program. A worker that exits
before `tell` leaves its trial unfinished, so the orchestrating application must decide how to
handle failures. For multi-objective studies, use `--directions` at creation and supply the
corresponding objective values to `tell`; inspect the front with `best-trials`.

The sampler factory resolves public sampler types in **Core** by reflection. It matches keyword
names after removing underscores and ignoring case. It does not load BoTorch, PyCma, pymoo, or
OptunaHub samplers as command-line plugins. Reconstructing a seeded sampler per CLI invocation
also is not the same as keeping a stateful sampler alive in a .NET process.

Keyword binding supports scalar values; arbitrary nested objects or collection constructor
arguments, such as a `GridSampler` search-space dictionary, are not generally constructible
through `--sampler-kwargs`. This is separate from `ask`'s distribution-based `--search-space`.

## Commands and output

| Command | Purpose |
|---|---|
| `create-study` | Create a named or generated-name study; `--skip-if-exists` permits reuse |
| `delete-study` | Delete a study and its trial records |
| `study set-user-attr` | Set a study attribute using `--key` and `--value` |
| `study-names` | List study names |
| `studies` | List study summaries |
| `trials` | List trial details |
| `best-trial` | Inspect the single-objective best trial |
| `best-trials` | Inspect the multi-objective Pareto front |
| `ask` | Allocate a trial and suggest from an explicit search space |
| `tell` | Record trial values or a terminal state |
| `storage upgrade` | Run the SQLite schema migration chain |

Common options include `--storage`, `--storage-class`, `-v` / `--verbose`, `-q` / `--quiet`,
`--log-file`, and `--debug`. Use subcommand help for accepted state names and argument cardinality.

`--log-file` behaves exactly as Optuna's: it creates the file and writes nothing to it. Optuna attaches its file
handler to Python's root logger, which the `optuna` logger does not propagate to, so the log records still go only to
stderr; OptunaSharp copies that on purpose, for byte-for-byte CLI parity. Redirect stderr to keep a log
(`optunasharp ... 2>> run.log`).

`-f value|json|table|yaml` selects a format. `study-names` defaults to `value`, listing commands to
`table`, and `ask` to `json`. `--flatten` expands structured columns such as parameters, attributes,
and multi-objective values. The differential tests compare `value`, `json`, and `table` output
against Python Optuna. YAML is valid but not byte-identical to PyYAML; command help and argument
validation come from System.CommandLine rather than Python's argparse.

## SQLite upgrades and cross-language use

The CLI uses the same [SQLite schema](storage/sqlite.md) as the library. Python Optuna can open
compatible database files created through either entry point. `storage upgrade` invokes the
SQLite backend's migration chain; it is not an EF Core migration command. It changes the database,
so plan upgrades as part of your database lifecycle.

For Redis, [journal log compatibility and snapshot limitations](storage/redis/index.md) apply to
the CLI too. The command-line interface does not broaden a backend's compatibility contract.

## Implementation and verification

The [command definitions](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Cli/CliProgram.cs),
[storage resolver](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Cli/StorageResolver.cs),
and [sampler factory](https://github.com/SciSharp/OptunaSharp/blob/main/src/OptunaSharp.Cli/SamplerFactory.cs)
define the supported surface. The
[CLI tests](https://github.com/SciSharp/OptunaSharp/tree/main/test/OptunaSharp.Tests.Cli)
cover command-output parity, storage resolution, sampler argument binding, and SQLite upgrades.
