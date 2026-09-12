# Integrations

Connect OptunaSharp to model trainers, databases, Python optimizers, cloud storage, and plotting
tools. This section covers every integration package in the current source tree, plus the
training integrations built into Core. Redis and gRPC are part of the storage family.

## Choose an integration

| Task | Start here |
|---|---|
| Tune a model with cross-validation | [Native scikit-learn-style search](training/scikit-learn.md) or [ML.NET](training/ml-net.md) |
| Tune LightGBM in stages | [LightGBM](training/lightgbm.md) |
| Stop unpromising training runs early | [Training-loop pruning](training/pruning.md) |
| Persist studies locally | [SQLite](storage/sqlite.md), or the Core [journal-file guide](../docs/storage.md#journal-file-storage-persistent-shareable) |
| Use a PostgreSQL or MySQL database | [EF Core](storage/ef-core.md), including its current concurrency limits |
| Share a Redis journal | [Redis and its three client adapters](storage/redis/index.md) |
| Expose a study backend over the network | [gRPC client, server, and cross-language use](storage/grpc/index.md) |
| Run Python optimization algorithms from C# | [Python runtime](python/pythonnet.md), [OptunaHub](python/optunahub.md), [BoTorch](python/botorch.md), [PyCma](python/pycma.md), or [pymoo](python/pymoo.md) |
| Store model files and other large outputs | [Cloud artifacts](artifacts/index.md): S3, GCS, Azure Blob, or R2 |
| Render interactive optimization charts | [Plotly.NET](visualization/plotly.md) |
| Drive studies from scripts or another process | [Command-line tool](cli.md) |

## Package map

Install the package for the integration you use. Package IDs below come from the projects in this
repository; runtime prerequisites, constructors, examples, and limitations are documented on each
integration page. The base `OptunaSharp` package is referenced by the satellites.

| Package | Integration |
|---|---|
| `OptunaSharp.Integration.ML.NET` | [ML.NET pipelines and forest importance](training/ml-net.md) |
| `OptunaSharp.Integration.LightGbm` | [LightGBM tuners, callbacks, and importance](training/lightgbm.md) |
| `OptunaSharp.Storage.Sqlite` | [Optuna-compatible SQLite RDB storage](storage/sqlite.md) |
| `OptunaSharp.Storage.EFCore` | [PostgreSQL and MySQL RDB storage](storage/ef-core.md) |
| `OptunaSharp.Storage.Redis` | [Shared Redis journal contract and protocol](storage/redis/index.md) |
| `OptunaSharp.Storage.StackExchange.Redis` | [StackExchange.Redis journal client](storage/redis/stackexchange-redis.md) |
| `OptunaSharp.Storage.NRedisStack` | [NRedisStack journal client](storage/redis/nredisstack.md) |
| `OptunaSharp.Storage.ServiceStack.Redis` | [ServiceStack.Redis journal client](storage/redis/servicestack-redis.md) |
| `OptunaSharp.Storage.Grpc` | [gRPC storage proxy](storage/grpc/index.md) |
| `OptunaSharp.Interop.pythonnet` | [CPython runtime](python/pythonnet.md) and [OptunaHub](python/optunahub.md) |
| `OptunaSharp.Interop.BoTorch` | [BoTorch Bayesian optimization](python/botorch.md) |
| `OptunaSharp.Interop.PyCma` | [Python `cma` sampler](python/pycma.md) |
| `OptunaSharp.Interop.Pymoo` | [pymoo population algorithms](python/pymoo.md) |
| `OptunaSharp.Artifacts.S3` | [Amazon S3](artifacts/s3.md) |
| `OptunaSharp.Artifacts.Gcs` | [Google Cloud Storage](artifacts/gcs.md) |
| `OptunaSharp.Artifacts.AzureBlob` | [Azure Blob Storage](artifacts/azure-blob.md) |
| `OptunaSharp.Artifacts.R2` | [Cloudflare R2](artifacts/r2.md) |
| `OptunaSharp.Visualization.Plotly` | [Plotly.NET charts](visualization/plotly.md) |
| `OptunaSharp.Cli` | [`optunasharp` .NET tool](cli.md) |

`OptunaSearchCV`, `ICrossValidationEstimator`, the managed scikit-learn-style regressor and fold
utilities, and `PruningCallback` are in **Core**. They do not require Python or a separate pruning
package. Core also contains file artifacts, journal files, in-memory storage, and visualization
data builders; their conceptual guides remain in [Docs](../docs/intro.md).

## How the pieces fit

A sampler proposes parameters; a training integration evaluates them; a storage integration
records the study and trials. Artifact stores hold larger files referenced by trial metadata.
Plotly renders the results, while the CLI can coordinate an ask/tell workflow around an external
training process. These integrations can be combined without using every package.

For example, an ML.NET objective can record its trials in SQLite and upload its model to S3.
Putting gRPC in front of a database changes how workers reach that study; it does not transport
the model files or distribute the training code. A Python sampler bridge executes Python within
the .NET process; it is different from gRPC interoperability between separate processes.

## Compatibility and implementation boundaries

The guides distinguish a compatible format or protocol from an identical training result. SQLite,
Redis journal logs, and gRPC have specific interoperability contracts. Redis snapshots have a
different serialization format across runtimes. ML.NET and LightGBM use their own trainer engines;
sharing a search algorithm does not make their fitted models identical to Python's.

Each guide links the relevant implementation and tests so its public API, ownership rules, and
support limits can be checked. In particular, installing an RDB package does not teach Core's
`StorageFunctions.GetStorage(string)` to load it: construct the backend explicitly. Python bridges
also have different concurrency and state-synchronization limits; read those before choosing an
algorithm or sharing a bridge instance.
