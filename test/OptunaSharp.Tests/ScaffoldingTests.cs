using System;
using System.Linq;
using System.Reflection;
using AwesomeAssertions;
using NumSharp;

namespace OptunaSharp.Tests;

/// <summary>
///     Smoke tests for the project stack itself. No Optuna algorithms are ported yet, so these
///     assert the scaffolding is wired correctly: the OptunaSharp assembly loads and is strong-named
///     with the shared SciSharp key, the version metadata is present, and the NumSharp dependency
///     resolves and runs through OptunaSharp's reference. They are the seed the real port grows from.
/// </summary>
[TestClass]
public class ScaffoldingTests
{
    /// <summary>The public key token every SciSharp STACK assembly shares (Microsoft's open key).</summary>
    private const string ExpectedPublicKeyToken = "cc7b13ffcd2ddd51";

    [TestMethod]
    public void Version_Metadata_IsPresent()
    {
        OptunaSharpVersion.Version.Should().NotBeNullOrWhiteSpace();
        OptunaSharpVersion.OptunaCompatibilityTarget.Should().NotBeNullOrWhiteSpace();
    }

    [TestMethod]
    public void OptunaSharpAssembly_IsStrongNamed_WithSciSharpKey()
    {
        AssemblyName name = typeof(OptunaSharpVersion).Assembly.GetName();

        name.Name.Should().Be("OptunaSharp");

        byte[]? token = name.GetPublicKeyToken();
        token.Should().NotBeNull();
        token!.Should().NotBeEmpty("the assembly must be strong-named for the SciSharp STACK identity");

        string tokenHex = string.Concat(token.Select(b => b.ToString("x2")));
        tokenHex.Should().Be(ExpectedPublicKeyToken);
    }

    [TestMethod]
    public void NumSharp_Dependency_Resolves_AndRuns()
    {
        // Proves OptunaSharp's NumSharp reference (csproj sibling or NuGet fallback) resolves at
        // runtime and NumSharp's np.* surface is usable — the numerical foundation the port builds on.
        using NDArray a = np.arange(6);
        ((int)a.size).Should().Be(6);

        using NDArray b = np.arange(6).reshape(2, 3);
        b.shape.Should().Equal(2, 3);
    }
}
