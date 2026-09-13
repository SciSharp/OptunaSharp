using System;
using System.Reflection;
using AwesomeAssertions;

namespace OptunaSharp.Tests.PackageParity;

/// <summary>
///     The "every shipped LIBRARY package restores, loads, and is ours" gate. One case per NuGet library
///     we produce (19 of the 20; the 20th, <c>OptunaSharp.Cli</c>, is a <c>dotnet tool</c> with no lib
///     assembly and is covered by <see cref="PackagedCliToolSmokeTests" />). Because the project consumes
///     each package from the local feed with NO path to <c>src/</c>, a case passing means the shipped
///     <c>.nupkg</c> restored against its real declared dependency graph (its third-party deps from
///     nuget.org) and its managed assembly deployed and loaded — the isolation the whole project exists to
///     enforce. The strong-name-token assertion pins that we bound to OUR signed artifact, not a same-named
///     shadow.
/// </summary>
/// <remarks>
///     This is deliberately a LOAD + IDENTITY smoke, not a behavioral test: loading a managed assembly does
///     not force its foreign runtime dependencies (Python, a database, native LightGBM, cloud SDKs) to
///     resolve, so a green case here does not claim those work — restore success already proves the deps
///     EXIST on the feeds (the NU1101 check), and the behavioral gates (<see cref="PackageSmokeTests" /> for
///     Core, <see cref="PackagedRedisJournalParityTests" /> for Redis, the oracle replays for the samplers)
///     prove the ones that can run credential-free actually run. No Python and no oracle are needed here.
/// </remarks>
[TestClass]
public class PackagedAssembliesSmokeTests
{
    /// <summary>The shared SciSharp public-key token every OptunaSharp assembly must carry (Open.snk).</summary>
    private const string SciSharpPublicKeyToken = "cc7b13ffcd2ddd51";

    /// <summary>
    ///     Asserts a shipped library package's assembly restored+deployed and carries the SciSharp identity.
    ///     Driven by ASSEMBLY name (Core's assembly is <c>OptunaSharp</c>; every satellite's assembly equals
    ///     its PackageId), so this list is the set of library packages minus the Cli tool.
    /// </summary>
    /// <param name="assemblyName">
    ///     The simple assembly name to load. It must resolve through the app's dependency context — a failure
    ///     to load is the signal that the shipped package did not restore/deploy as a downstream consumer
    ///     would receive it.
    /// </param>
    [TestMethod]
    [DataRow("OptunaSharp")]
    [DataRow("OptunaSharp.Storage.Sqlite")]
    [DataRow("OptunaSharp.Storage.EFCore")]
    [DataRow("OptunaSharp.Storage.Grpc")]
    [DataRow("OptunaSharp.Storage.Redis")]
    [DataRow("OptunaSharp.Storage.StackExchange.Redis")]
    [DataRow("OptunaSharp.Storage.NRedisStack")]
    [DataRow("OptunaSharp.Storage.ServiceStack.Redis")]
    [DataRow("OptunaSharp.Artifacts.S3")]
    [DataRow("OptunaSharp.Artifacts.Gcs")]
    [DataRow("OptunaSharp.Artifacts.AzureBlob")]
    [DataRow("OptunaSharp.Artifacts.R2")]
    [DataRow("OptunaSharp.Integration.LightGbm")]
    [DataRow("OptunaSharp.Integration.ML.NET")]
    [DataRow("OptunaSharp.Interop.pythonnet")]
    [DataRow("OptunaSharp.Interop.BoTorch")]
    [DataRow("OptunaSharp.Interop.PyCma")]
    [DataRow("OptunaSharp.Interop.Pymoo")]
    [DataRow("OptunaSharp.Visualization.Plotly")]
    public void PackagedLibrary_Loads_AndCarriesSciSharpStrongNameToken(string assemblyName)
    {
        // Load through the standard resolver (deps.json + probing) — this is what a consumer's runtime does.
        // If the package failed to restore/deploy its lib assembly, this throws and the case fails.
        Assembly asm = Assembly.Load(new AssemblyName(assemblyName));
        asm.Should().NotBeNull($"{assemblyName} must restore from the feed and load standalone");

        // Identity, not authenticity: prove we bound to the OptunaSharp we ship (shared SciSharp key),
        // never a same-named assembly from elsewhere on the probing path.
        byte[]? token = asm.GetName().GetPublicKeyToken();
        token.Should().NotBeNullOrEmpty($"{assemblyName} must be strong-named");
        Convert.ToHexString(token!).ToLowerInvariant()
            .Should().Be(SciSharpPublicKeyToken, $"{assemblyName} must carry the shared SciSharp key");
    }
}
