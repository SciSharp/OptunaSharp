# Command-Line Interface

OptunaSharp ships a command-line tool, **`optunasharp`**, that is a faithful port of the
[`optuna` command](https://optuna.readthedocs.io/en/stable/reference/cli.html) (`optuna/cli.py`). It
lets you create and inspect studies and drive ask/tell trials from the shell, against journal-file,
SQLite and Redis storages — with output that is **byte-identical** to Optuna's for the `value`,
`json` and `table` formats.

The tool lives in the **`OptunaSharp.Cli`** package (built on
[System.CommandLine](https://learn.microsoft.com/dotnet/standard/commandline/) and
[YamlDotNet](https://github.com/aaubry/YamlDotNet)) and is distributed as a
[.NET global tool](https://learn.microsoft.com/dotnet/core/tools/global-tools).

## Install

```bash
dotnet tool install --global OptunaSharp.Cli
```

This registers the `optunasharp` command. (The name `optuna` belongs to the Python package; this is
the drop-in .NET port.) It bundles the SQLite (`RDBStorage`) and StackExchange.Redis backends, so
`sqlite:///` and `redis://` URLs work immediately; journal-file storage is part of Core.

## Storage resolution

Every command takes `--storage <url>` (or reads the `OPTUNA_STORAGE` environment variable). The URL is
resolved exactly as Optuna resolves it:

| URL / hint | Backend |
|---|---|
| `redis…` / `rediss…` | Redis journal backend (StackExchange.Redis) |
| an existing file path | Journal-file backend (Core) |
| anything else (e.g. `sqlite:///study.db`) | SQLite `RDBStorage` |

`--storage-class` overrides the guess: `RDBStorage`, `JournalFileBackend`, `JournalRedisBackend`, and
the deprecated `JournalFileStorage` / `JournalRedisStorage` aliases.

Because OptunaSharp's SQLite schema is [Optuna-compatible](storage.md), a database created by
`optunasharp` opens unchanged in Python's `optuna` command, and vice versa.

## Commands

```text
create-study            create a study (--direction / --directions / --skip-if-exists); prints the name
delete-study            delete a study by name
study set-user-attr     set a study user attribute (--key / --value)
study-names             list all study names
studies                 list studies (name, direction, n_trials, datetime_start, user_attrs)
trials                  list a study's trials                     (experimental)
best-trial              show the best trial                        (experimental)
best-trials             show the Pareto-front trials               (experimental)
storage upgrade         report an RDB storage's schema state
ask                     create a trial and suggest parameters      (experimental)
tell                    finish a trial created by ask              (experimental)
```

The common options — `--storage`, `--storage-class`, `-v`/`--verbose`, `-q`/`--quiet`, `--log-file`,
`--debug` — may appear before or after the subcommand.

## A worked example

```bash
# Create a study in a SQLite DB.
optunasharp create-study --study-name quadratic --storage sqlite:///study.db

# Ask a trial with a seeded RandomSampler over an explicit search space, then tell its value.
optunasharp ask  --study-name quadratic --storage sqlite:///study.db \
    --sampler RandomSampler --sampler-kwargs '{"seed": 0}' \
    --search-space '{"x": {"name": "FloatDistribution", "attributes": {"low": -10.0, "high": 10.0, "log": false, "step": null}}}'
optunasharp tell --study-name quadratic --storage sqlite:///study.db --trial-number 0 --values 4.0

# Inspect it.
optunasharp trials --study-name quadratic --storage sqlite:///study.db
```

The `ask` above draws from the same MT19937 stream NumPy uses, so with a fixed seed the suggested
parameters are **bit-identical** to what the `optuna` command would produce — the same
[RNG discipline](samplers.md) that underpins OptunaSharp's differential parity everywhere.

## Output formats

`-f value|json|table|yaml` selects the format (the defaults follow Optuna: `value` for `study-names`,
`table` for the listing commands, `json` for `ask`). `--flatten` flattens nested columns — `params`,
`user_attrs`, the objective `values` of a multi-objective study, and `direction`.

The `value`, `json` and `table` formats are reproduced down to the byte: the same Python float `repr`,
tuple/dict `repr`, `json.dumps` spacing, `timedelta` strings, and box-drawn table. `yaml` is valid
YAML but is **not** byte-identical to PyYAML's `safe_dump` (which quotes the timestamp and duration
scalars its implicit resolver would otherwise reinterpret) — YAML byte-parity is not a goal.

## Notes and divergences

- **`storage upgrade`** validates the RDB URL and reports whether the schema is the version OptunaSharp
  supports. OptunaSharp's `RDBStorage` implements only the head alembic revision and does not port
  Optuna's migration chain, so it cannot migrate an older schema — do that with Python's
  `optuna storage upgrade`, then reopen the database with OptunaSharp.
- Help text and validation-error wording come from System.CommandLine, not argparse; it is the command
  **output** (stdout) that matches Optuna byte-for-byte.
