using System.Collections.Generic;

namespace OptunaSharp.Tests;

/// <summary>
///     Test categories used for filtering in CI and local development. The CI pipeline
///     (<c>.github/workflows/build-and-release.yml</c>) uses MSTest's <c>--filter</c> to exclude
///     some of these, e.g. <c>--filter "TestCategory!=OpenBugs&amp;TestCategory!=HighMemory"</c>.
/// </summary>
public static class TestCategory
{
    /// <summary>Known-failing reproductions (a real OptunaSharp bug, or a not-yet-ported feature).
    ///     Excluded from CI. Remove the attribute when the test starts passing.</summary>
    public const string OpenBugs = "OpenBugs";

    /// <summary>Tests documenting an intentional/unavoidable difference from Optuna's behavior.
    ///     These PASS and run in CI — Optuna is the source of truth, this records where we diverge.</summary>
    public const string Misaligned = "Misaligned";

    /// <summary>Tests requiring a discoverable Python + Optuna install (live parity). Self-skip
    ///     (Inconclusive) when absent, so they never turn a machine without Python red.</summary>
    public const string RequiresPython = "RequiresPython";

    /// <summary>Tests requiring a lot of RAM. Excluded from CI runners.</summary>
    public const string HighMemory = "HighMemory";

    /// <summary>Tests requiring Windows-only APIs. Excluded on non-Windows CI runners.</summary>
    public const string WindowsOnly = "WindowsOnly";
}

/// <summary>Shorthand for <c>[TestCategory("OpenBugs")]</c>. See <see cref="TestCategory.OpenBugs" />.</summary>
[TestClass]
public class OpenBugsAttribute : TestCategoryBaseAttribute
{
    /// <summary>Optional URL to the tracking issue (clickable in most IDEs).</summary>
    public string? IssueUrl { get; set; }

    public override IList<string> TestCategories => [TestCategory.OpenBugs];
}

/// <summary>Shorthand for <c>[TestCategory("Misaligned")]</c>. See <see cref="TestCategory.Misaligned" />.</summary>
[TestClass]
public class MisalignedAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.Misaligned];
}

/// <summary>Shorthand for <c>[TestCategory("RequiresPython")]</c>. See <see cref="TestCategory.RequiresPython" />.</summary>
[TestClass]
public class RequiresPythonAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.RequiresPython];
}

/// <summary>Shorthand for <c>[TestCategory("HighMemory")]</c>. See <see cref="TestCategory.HighMemory" />.</summary>
[TestClass]
public class HighMemoryAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.HighMemory];
}

/// <summary>Shorthand for <c>[TestCategory("WindowsOnly")]</c>. See <see cref="TestCategory.WindowsOnly" />.</summary>
[TestClass]
public class WindowsOnlyAttribute : TestCategoryBaseAttribute
{
    public override IList<string> TestCategories => [TestCategory.WindowsOnly];
}
