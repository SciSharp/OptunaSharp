using System;
using System.Linq;
using Amazon;
using Amazon.Runtime;
using Amazon.S3;
using AwesomeAssertions;
using Google.Cloud.Storage.V1;
using Microsoft.Extensions.DependencyInjection;

// OptunaSharp core + the DI extension namespaces of EVERY DI-enabled package. Importing all of them in a
// single file is itself the primary "no clash" proof: several packages deliberately share a namespace
// (Artifacts x4 -> OptunaSharp.Artifacts.Extensions; the three Redis backends ->
// OptunaSharp.Storages.Journal.Extensions), so if any two `ServiceCollectionExtensions` classes had the
// same name this file would not compile (CS0433 ambiguous reference).
using OptunaSharp;
using OptunaSharp.Artifacts;                        // IArtifactStore + the concrete stores
using OptunaSharp.Artifacts.Extensions;             // AddS3/Gcs/AzureBlob/R2 ArtifactStore
using OptunaSharp.Extensions;                        // AddOptunaSampler/Pruner/Storage/StudyFactory
using OptunaSharp.Importance;                        // BaseImportanceEvaluator
using OptunaSharp.Integration;                       // ICrossValidationEstimator
using OptunaSharp.Integration.LightGbm;              // ILightGbmTrainer, ILightGbmBoosterStore
using OptunaSharp.Integration.LightGbm.Extensions;   // AddLightGbm*
using OptunaSharp.Integration.MLNet.Extensions;      // AddFastForest*/FastTree*/MLNetEstimator/MLNetImportanceEvaluator
using OptunaSharp.Interop.BoTorch.Extensions;        // AddBoTorchSampler
using OptunaSharp.Interop.PyCma.Extensions;          // AddPyCmaSampler
using OptunaSharp.Interop.Pymoo.Extensions;          // AddPymooSampler
using OptunaSharp.Interop.PythonNet.Extensions;      // AddOptunaHubSampler/Pruner
using OptunaSharp.Pruners;                           // BasePruner, MedianPruner
using OptunaSharp.Samplers;                           // BaseSampler, RandomSampler
using OptunaSharp.Storages;                          // BaseStorage, RDBStorage (SQLite)
using OptunaSharp.Storages.EFCore.Extensions;        // AddEFCoreStorage
using OptunaSharp.Storages.Extensions;               // AddSqliteStorage
using OptunaSharp.Storages.Grpc;                     // GrpcStorageProxy
using OptunaSharp.Storages.Grpc.Extensions;          // AddOptunaGrpcStorageProxyClient
using OptunaSharp.Storages.Journal;                  // JournalStorage
using OptunaSharp.Storages.Journal.Extensions;       // Add{StackExchange,NRedisStack,ServiceStack}RedisJournalStorage
using OptunaSharp.Visualization.Plotly;              // PlotlyChartCustomization
using OptunaSharp.Visualization.Plotly.Extensions;   // AddOptunaPlotlyVisualization, IOptunaPlotlyVisualizer

namespace OptunaSharp.Tests.Hosting;

/// <summary>
///     A DI "host" that composes EVERY DI-enabled OptunaSharp package into one
///     <see cref="IServiceCollection" />, proving the whole <c>*.Extensions</c> surface (1) compiles
///     together with no namespace / type-name clash, (2) registers without one package overwriting another,
///     and (3) resolves working instances for the offline-constructible services. Services that require an
///     external resource (a live DB / Redis server, cloud credentials) or the embedded CPython runtime
///     (the Interop samplers) are proven at the REGISTRATION level here — their construction is exercised by
///     their own storage / interop / integration test suites — because this host initializes no external
///     dependency and no Python.
/// </summary>
[TestClass]
public class AllPackagesDiHostTests
{
    /// <summary>
    ///     Register the DI extension of every package into <paramref name="services" />. Compiling this
    ///     method is the core "no clash" proof (all shared-namespace extension classes coexist); calling it
    ///     is the "no overwrite" proof (every registration is additive). The convenience overloads for the
    ///     Interop samplers register LAZY factories, so no Python is touched here.
    ///     <para><b>Order matters for one thing only:</b> the Core registrations come LAST so the study
    ///     factory's last-wins <c>GetService</c> resolves the offline in-memory storage / random sampler /
    ///     median pruner (not one of the satellite storages/samplers that would need an external resource).</para>
    /// </summary>
    /// <param name="services">The collection every package's <c>Add*</c> extension is invoked on.</param>
    private static void ConfigureEverything(IServiceCollection services)
    {
        // ---- Artifacts (all four share OptunaSharp.Artifacts.Extensions) ----
        services.AddS3ArtifactStore("bucket");
        services.AddGcsArtifactStore("bucket");
        services.AddAzureBlobArtifactStore("UseDevelopmentStorage=true", "container");
        services.AddR2ArtifactStore("account", "accessKeyId", "secretAccessKey", "bucket");

        // ---- Storage (SQLite / EF Core / gRPC client / three Redis backends) ----
        services.AddSqliteStorage("sqlite:///:memory:");
        services.AddEFCoreStorage("postgresql://user:pw@localhost:5432/optuna");
        services.AddOptunaGrpcStorageProxyClient("localhost", 13000);
        services.AddStackExchangeRedisJournalStorage("redis://localhost:6379");
        services.AddNRedisStackJournalStorage("redis://localhost:6379");
        services.AddServiceStackRedisJournalStorage("redis://localhost:6379");

        // ---- Interop samplers / pruner (lazy factories — no CPython at registration) ----
        services.AddBoTorchSampler();
        services.AddPyCmaSampler();
        services.AddPymooSampler();
        services.AddOptunaHubSampler("samplers/auto_sampler", "AutoSampler");
        services.AddOptunaHubPruner("pruners/example", "ExamplePruner");

        // ---- Integration (LightGBM tuner engine + off-label evaluators; ML.NET evaluators + estimator) ----
        services.AddLightGbmTrainer();
        services.AddLightGbmFanovaImportanceEvaluator();
        services.AddLightGbmMeanDecreaseImpurityImportanceEvaluator();
        services.AddFastForestFanovaImportanceEvaluator();
        services.AddFastForestMeanDecreaseImpurityImportanceEvaluator();
        services.AddFastTreeFanovaImportanceEvaluator();
        services.AddFastTreeMeanDecreaseImpurityImportanceEvaluator();
        services.AddFastTreeTweedieFanovaImportanceEvaluator();
        services.AddFastTreeTweedieMeanDecreaseImpurityImportanceEvaluator();
        // MLNetEstimator needs a live MLContext + IDataView + pipeline + scorer, so it is registration-only
        // here (its resolution is covered by OptunaSharp.Tests.Integration.ML.NET). The factory is never
        // invoked in these tests — ICrossValidationEstimator is not resolved.
        services.AddMLNetEstimator(_ => throw new NotSupportedException(
            "This host proves registration coexistence only; resolve MLNetEstimator with a real ML.NET pipeline."));

        // ---- Visualization facade ----
        services.AddOptunaPlotlyVisualization(new PlotlyChartCustomization());

        // ---- Core LAST, so the study factory's last-wins resolution picks the offline pieces ----
        services.AddOptunaSampler(new RandomSampler(seed: 0));
        services.AddOptunaPruner(new MedianPruner());
        services.AddOptunaStorage(new InMemoryStorage());
        services.AddOptunaStudyFactory(o => o.DefaultDirection = StudyDirection.Minimize);
    }

    /// <summary>Counts the registrations for a service type without constructing anything.</summary>
    /// <param name="services">The populated collection.</param>
    /// <typeparam name="T">The service type to count descriptors for.</typeparam>
    /// <returns>The number of <see cref="ServiceDescriptor" />s whose <c>ServiceType</c> is <typeparamref name="T" />.</returns>
    private static int Count<T>(IServiceCollection services)
        => services.Count(d => d.ServiceType == typeof(T));

    [TestMethod]
    public void EveryDiExtension_FromAllPackages_CoexistsWithoutClashOrOverwrite()
    {
        var services = new ServiceCollection();
        ConfigureEverything(services);

        // Every package's registrations are present side by side — a later package never overwrote an
        // earlier one's abstraction. These exact counts ARE the coexistence proof.
        Count<IArtifactStore>(services).Should().Be(4, "S3 + GCS + AzureBlob + R2 each register IArtifactStore");
        Count<BaseStorage>(services).Should().Be(7, "SQLite + EFCore + gRPC + 3 Redis + Core in-memory");
        Count<JournalStorage>(services).Should().Be(3, "the three Redis backends each wrap a JournalStorage");
        Count<BaseSampler>(services).Should().Be(5, "BoTorch + PyCma + Pymoo + OptunaHub + Core random");
        Count<BasePruner>(services).Should().Be(2, "OptunaHub + Core median");
        Count<BaseImportanceEvaluator>(services).Should().Be(8, "2 LightGBM + 6 ML.NET evaluators");
        Count<ILightGbmTrainer>(services).Should().Be(1);
        Count<ILightGbmBoosterStore>(services).Should().Be(1);
        Count<ICrossValidationEstimator>(services).Should().Be(1);
        Count<IOptunaPlotlyVisualizer>(services).Should().Be(1);
        Count<IOptunaStudyFactory>(services).Should().Be(1);

        // The container itself builds cleanly with the full registration set (validates scope rules; does
        // not construct any service).
        using ServiceProvider provider = services.BuildServiceProvider(validateScopes: true);
        provider.Should().NotBeNull();
    }

    [TestMethod]
    public void AllPackagesRegistered_OfflineServices_ResolveAndWork()
    {
        var services = new ServiceCollection();
        ConfigureEverything(services);
        using ServiceProvider provider = services.BuildServiceProvider(validateScopes: true);

        // Pure-managed / offline-constructible services resolve to working instances even with every other
        // package registered alongside them.
        provider.GetRequiredService<IOptunaPlotlyVisualizer>().Should().NotBeNull();
        provider.GetRequiredService<ILightGbmTrainer>().Should().NotBeNull();
        provider.GetRequiredService<ILightGbmBoosterStore>().Should().NotBeNull();

        // All eight importance evaluators construct (they are configuration holders until Evaluate is called).
        provider.GetServices<BaseImportanceEvaluator>().Should().HaveCount(8).And.OnlyContain(e => e != null);

        // Offline storage backends: SQLite in-memory opens + creates its schema; the gRPC client builds its
        // channel lazily (no connection at construction).
        provider.GetRequiredService<RDBStorage>().Should().NotBeNull();          // OptunaSharp.Storages.RDBStorage (SQLite :memory:)
        provider.GetRequiredService<GrpcStorageProxy>().Should().NotBeNull();

        // Artifact stores that construct without cloud credentials (Azure connection-string, R2 explicit keys).
        provider.GetRequiredService<AzureBlobArtifactStore>().Should().NotBeNull();
        provider.GetRequiredService<R2ArtifactStore>().Should().NotBeNull();

        // The composition root: the factory resolves the (last-registered, offline) Core sampler/pruner/
        // storage and produces a runnable study.
        var factory = provider.GetRequiredService<IOptunaStudyFactory>();
        Study study = factory.CreateStudy();
        study.Direction.Should().Be(StudyDirection.Minimize);
        study.Optimize(t => t.SuggestFloat("x", -5, 5), nTrials: 4);
        study.Trials.Should().HaveCount(4);
    }

    [TestMethod]
    public void AllFourArtifactStores_ResolveOffline_ViaContainerProvidedClients()
    {
        var services = new ServiceCollection();

        // Providing the cloud clients through DI lets every artifact store construct offline: an S3 client
        // with explicit (non-resolving) credentials, and an unauthenticated GCS client — so the stores
        // never reach out for ambient credentials at construction.
        services.AddSingleton<IAmazonS3>(new AmazonS3Client(
            new BasicAWSCredentials("test", "test"), RegionEndpoint.USEast1));
        services.AddSingleton(StorageClient.CreateUnauthenticated());

        services.AddS3ArtifactStore("bucket");                                             // resolves the DI IAmazonS3
        services.AddGcsArtifactStore("bucket");                                            // resolves the DI StorageClient
        services.AddAzureBlobArtifactStore("UseDevelopmentStorage=true", "container");
        services.AddR2ArtifactStore("account", "accessKeyId", "secretAccessKey", "bucket");

        using ServiceProvider provider = services.BuildServiceProvider(validateScopes: true);

        // All four distinct backends resolve through the shared IArtifactStore abstraction, side by side.
        var stores = provider.GetServices<IArtifactStore>().ToList();
        stores.Should().HaveCount(4).And.OnlyContain(s => s != null);
        stores.Select(s => s.GetType()).Should().BeEquivalentTo(new[]
        {
            typeof(Boto3ArtifactStore), typeof(GCSArtifactStore),
            typeof(AzureBlobArtifactStore), typeof(R2ArtifactStore),
        });
    }

    [TestMethod]
    public void InteropSamplerExtensions_RegisterLazily_WithoutInitializingPython()
    {
        // The convenience overloads register factories only — constructing a BoTorch/PyCma/Pymoo/OptunaHub
        // sampler (which initializes embedded CPython and imports the Python library) is deferred to first
        // resolution. Registering them and building the provider must therefore succeed with no Python
        // present; the tests intentionally do NOT resolve BaseSampler.
        var services = new ServiceCollection();
        services.AddBoTorchSampler();
        services.AddPyCmaSampler();
        services.AddPymooSampler();
        services.AddOptunaHubSampler("samplers/auto_sampler", "AutoSampler");

        Count<BaseSampler>(services).Should().Be(4);

        using ServiceProvider provider = services.BuildServiceProvider(validateScopes: true);
        provider.Should().NotBeNull();
    }
}
