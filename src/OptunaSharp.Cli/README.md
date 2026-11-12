# OptunaSharp.Cli

The **`optunasharp` command-line tool** — a faithful port of the [Optuna](https://github.com/optuna/optuna)
CLI (`optuna/cli.py`) for [OptunaSharp](https://github.com/SciSharp/OptunaSharp), built on
[System.CommandLine](https://learn.microsoft.com/dotnet/standard/commandline/).

Create and inspect studies and drive ask/tell trials from the shell, against **journal-file**,
**SQLite** (`RDBStorage`) and **Redis** storages — with **byte-identical `value`/`json`/`table`/`yaml`
output** to the `optuna` command (verified live against Optuna 5.0.0; `yaml` is a port of PyYAML's `safe_dump`,
pinned by a 763-case corpus).

Part of the [SciSharp STACK](https://github.com/SciSharp).

## Install

```bash
dotnet tool install --global OptunaSharp.Cli
```

This exposes the `optunasharp` command (the name `optuna` belongs to the Python package; this is the
drop-in .NET port). It bundles the SQLite and StackExchange.Redis storage backends, so `sqlite:///`
and `redis://` URLs work out of the box; journal-file storage is built in.

## Commands

Every command mirrors its Optuna counterpart, including the shared options
`--storage` / `--storage-class` / `-v` / `-q` / `--log-file` / `--debug`. Like Optuna's, `--log-file` creates the
file but writes nothing to it (Optuna's handler sits on Python's root logger, which the `optuna` logger does not
propagate to); redirect stderr to keep a log.

| Command | What it does |
|---|---|
| `create-study` | Create a new study (`--direction` / `--directions` / `--skip-if-exists`); prints the name |
| `delete-study` | Delete a study by name |
| `study set-user-attr` | Set a study user attribute (`--key`, `--value`) |
| `study-names` | List all study names in a storage |
| `studies` | List studies (name, direction(s), trial count, start time, user attrs) |
| `trials` | List a study's trials *(experimental)* |
| `best-trial` | Show the best trial *(experimental)* |
| `best-trials` | Show the Pareto-front trials *(experimental)* |
| `storage upgrade` | Report an RDB storage's schema state (see the note below) |
| `ask` | Create a trial and suggest parameters *(experimental)* |
| `tell` | Finish a trial created by `ask` *(experimental)* |

The `--storage` URL is resolved exactly as Optuna resolves it: a `redis…` URL → a Redis journal
backend; an existing file → a journal-file backend; otherwise → `RDBStorage`, where only SQLAlchemy's
`ArgumentError` (a string that is not a URL, an unknown dialect) becomes Optuna's
`Failed to guess storage class from storage_url` usage error and every other error propagates.
The URL's dialect picks the RDB package: SQLite is bundled; a PostgreSQL or MySQL URL needs the
`OptunaSharp.Storage.EFCore` package, which the tool does not ship, and fails with Optuna's
`ImportError` text. `--storage-class` forces the class (`RDBStorage`, `JournalFileBackend`,
`JournalRedisBackend`, and the deprecated `JournalFileStorage` / `JournalRedisStorage` aliases). The
`OPTUNA_STORAGE` environment variable is honored as a fallback, like Optuna's.

## Examples

```bash
# Create a study in a SQLite DB (Python Optuna can open the very same file).
optunasharp create-study --study-name quadratic --storage sqlite:///study.db

# Ask a trial with a seeded RandomSampler over an explicit search space, then tell its value.
optunasharp ask  --study-name quadratic --storage sqlite:///study.db \
    --sampler RandomSampler --sampler-kwargs '{"seed": 0}' \
    --search-space '{"x": {"name": "FloatDistribution", "attributes": {"low": -10.0, "high": 10.0, "log": false, "step": null}}}'
optunasharp tell --study-name quadratic --storage sqlite:///study.db --trial-number 0 --values 4.0

# Inspect it in any format.
optunasharp studies    --storage sqlite:///study.db                # table (default)
optunasharp trials     --study-name quadratic --storage sqlite:///study.db -f json --flatten
optunasharp best-trial --study-name quadratic --storage sqlite:///study.db -f value
```

Because the `--sampler RandomSampler --sampler-kwargs '{"seed": 0}'` path runs the same MT19937 stream
as NumPy, the suggested parameters are **bit-identical** to what the `optuna` command would produce.

## Output formats

`-f value|json|table|yaml` (defaults per command follow Optuna: `study-names` → `value`, `studies`
/ `trials` / `best-trial` / `best-trials` → `table`, `ask` → `json`). `--flatten` flattens nested
columns (`params`, `user_attrs`, multi-objective `values`, `direction`).

The `value`, `json` and `table` formats are **byte-identical** to Optuna's — the same Python float
`repr`, dict/tuple `repr`, `json.dumps` spacing, timedelta strings and box-drawn table. `yaml` is
valid YAML but **not** byte-identical to PyYAML's `safe_dump` (which quotes timestamp/duration scalars
its implicit resolver would reinterpret); YAML byte-parity is not a goal.

## Notes / divergences

- **`storage upgrade`** replays Optuna's alembic migration chain to migrate an older SQLite database
  to the head schema — the same behavior as Python's `optuna storage upgrade`, producing a
  byte-for-byte identical file. (SQLite backend; the EF Core PostgreSQL/MySQL backend is still
  head-only.)
- **Redis** is wired to the `StackExchange.Redis` backend (the SciSharp-canonical .NET Redis client).
- Help text and validation-error wording are System.CommandLine's, not argparse's; command **output**
  (stdout) is what matches Optuna.

## License

MIT, as part of the SciSharp STACK.
