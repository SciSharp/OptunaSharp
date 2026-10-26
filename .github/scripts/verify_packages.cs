#:property PublishAot=false
// -----------------------------------------------------------------------------------------------
// Release gate for the PACKED OptunaSharp family: checks that the feed is exactly what a release
// should push to nuget.org.
//
// WHY THIS EXISTS. Every check below catches a defect that builds, tests and even passes the
// strong-name gate cleanly, then ships to users:
//   - a package is missing from the feed (the release job once packed only Core, 1 of 21);
//   - a version was not stamped (a satellite depending on an older OptunaSharp; assemblies whose
//     informational version, and so OptunaSharpVersion.Version, still read "0.1.0");
//   - nuget.org metadata is broken or stale (no license/icon/readme, "early scaffolding" text);
//   - SourceLink is absent or points at another repository or commit, so "step into source" fails.
//
// It also decides whether the feed is PUBLISHABLE. Each nuspec <repository url> names the
// canonical public repository (Directory.Build.props). The PDBs' SourceLink names the repository
// the build actually ran in. They agree only when the release runs in the canonical repository,
// and only then can a user step from the published package into its sources. A private mirror
// builds valid packages that must never be pushed.
//
//   dotnet run .github/scripts/verify_packages.cs -- <feed-dir>
//       --version <package version>          e.g. 1.2.3 or 1.2.3-rc.1
//       --commit <40-hex sha>                the commit the packages must be built from
//       --source-repository <url>            https://github.com/<owner>/<repo> the build ran in
//       [--projects <dir>]                   default: src — every packable <dir>/*/*.csproj must ship
//       [--require-publishable]              fail (instead of report) when the feed is not publishable
//
// Writes `publishable`, `canonical_repository` and `package_count` to $GITHUB_OUTPUT when set.
// Exit 0 = every check passed; exit 1 = anything else, including an empty feed (no vacuous pass).
// -----------------------------------------------------------------------------------------------
using System.IO.Compression;
using System.Reflection.Metadata;
using System.Reflection.PortableExecutable;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Xml.Linq;

var options = GateOptions.Parse(args);
if (options is null) return 1;

var failures = new List<string>();
var expectedIds = PackageChecks.ExpectedPackageIds(options.ProjectsDir, failures);
var packagePaths = Directory.Exists(options.FeedDir)
    ? Directory.GetFiles(options.FeedDir, "*.nupkg", SearchOption.TopDirectoryOnly)
        .Where(p => !p.EndsWith(".snupkg", StringComparison.OrdinalIgnoreCase)).OrderBy(p => p).ToList()
    : new List<string>();

if (packagePaths.Count == 0)
{
    Console.Error.WriteLine($"::error::verify_packages: no .nupkg in '{options.FeedDir}'. Refusing to pass " +
                            "vacuously: if the pack step produced nothing, that IS the failure.");
    return 1;
}

Console.WriteLine($"Expecting {expectedIds.Count} packages at version {options.Version}, built from " +
                  $"{options.SourceRepository} @ {options.Commit}\n");

var seenIds = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
var canonicalRepositories = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
foreach (string path in packagePaths)
{
    Console.WriteLine(Path.GetFileName(path));
    var report = PackageChecks.VerifyPackage(path, options, expectedIds);
    foreach (string line in report.Notes) Console.WriteLine($"  OK   {line}");
    foreach (string problem in report.Problems)
    {
        Console.Error.WriteLine($"  ::error::{Path.GetFileName(path)}: {problem}");
        failures.Add($"{Path.GetFileName(path)}: {problem}");
    }

    if (report.Id is null) continue;
    if (seenIds.TryGetValue(report.Id, out string? earlier))
        failures.Add($"package id {report.Id} appears twice ({Path.GetFileName(earlier)} and {Path.GetFileName(path)})");
    else seenIds[report.Id] = path;
    if (report.RepositoryUrl is not null) canonicalRepositories.Add(PackageChecks.NormalizeRepositoryUrl(report.RepositoryUrl));
}

// The feed must be the WHOLE family: a missing package is the defect this gate was written for, and
// an unexpected one means a project slipped in that the release did not intend to publish.
foreach (string id in expectedIds.Keys.Where(id => !seenIds.ContainsKey(id)))
    failures.Add($"no package for {id} ({expectedIds[id]}) — every packable project must ship in the release");
foreach (string id in seenIds.Keys.Where(id => !expectedIds.ContainsKey(id)))
    failures.Add($"unexpected package {id} — no packable project under '{options.ProjectsDir}' declares it");

// One family, one canonical home: packages that disagree on <repository url> cannot all be right.
if (canonicalRepositories.Count > 1)
    failures.Add($"packages disagree on the canonical repository: {string.Join(", ", canonicalRepositories)}");

string canonical = canonicalRepositories.FirstOrDefault() ?? "";
bool publishable = canonicalRepositories.Count == 1
                   && string.Equals(canonical, PackageChecks.NormalizeRepositoryUrl(options.SourceRepository), StringComparison.OrdinalIgnoreCase);

Console.WriteLine();
if (publishable)
    Console.WriteLine($"publishable: yes. SourceLink and <repository> both point at {canonical}");
else
{
    string message = $"not publishable: the packages name {canonical} as their repository, but their SourceLink " +
                     $"points at {options.SourceRepository}, where this build ran. Users of a published package " +
                     "could not step into its sources. Release from the canonical repository.";
    if (options.RequirePublishable) failures.Add(message);
    else Console.WriteLine($"::notice::{message}");
}

string? githubOutput = Environment.GetEnvironmentVariable("GITHUB_OUTPUT");
if (!string.IsNullOrEmpty(githubOutput))
{
    // Step outputs let the publish job refuse a non-publishable feed without re-reading the packages.
    File.AppendAllText(githubOutput,
        $"publishable={(publishable ? "true" : "false")}\ncanonical_repository={canonical}\npackage_count={seenIds.Count}\n");
}

Console.WriteLine($"\npackages checked: {packagePaths.Count}   expected: {expectedIds.Count}   failures: {failures.Count}");
if (failures.Count > 0)
{
    foreach (string failure in failures) Console.Error.WriteLine($"::error::verify_packages: {failure}");
    return 1;
}
Console.WriteLine("verify_packages: OK");
return 0;

/// <summary>
///     The parsed command line of the gate. Every option except <see cref="ProjectsDir" /> and
///     <see cref="RequirePublishable" /> is mandatory: a gate that guesses the version or the
///     commit would pass a feed built from the wrong one.
/// </summary>
/// <param name="FeedDir">Directory holding the <c>.nupkg</c> files to check (top level only).</param>
/// <param name="Version">The exact package version every package, dependency and assembly must carry.</param>
/// <param name="Commit">The full 40-hex commit the packages must name, in their nuspec and their SourceLink.</param>
/// <param name="SourceRepository">The <c>https://github.com/&lt;owner&gt;/&lt;repo&gt;</c> URL the build ran in.</param>
/// <param name="ProjectsDir">Directory whose packable <c>*/*.csproj</c> define the expected package set.</param>
/// <param name="RequirePublishable">When true, a non-publishable feed is a failure instead of a notice.</param>
internal sealed record GateOptions(
    string FeedDir, string Version, string Commit, string SourceRepository, string ProjectsDir, bool RequirePublishable)
{
    /// <summary>
    ///     Parses the gate's arguments. Usage problems are printed as workflow errors so that CI shows
    ///     them as annotations.
    /// </summary>
    /// <param name="args">The raw command-line arguments after <c>--</c>.</param>
    /// <returns>The options, or <see langword="null" /> when the command line is incomplete or malformed.</returns>
    public static GateOptions? Parse(string[] args)
    {
        string? feed = null, version = null, commit = null, repository = null, projects = "src";
        bool requirePublishable = false;
        for (int i = 0; i < args.Length; i++)
        {
            // Every value option consumes the next argument; a trailing flag with no value is a usage error.
            string? Next() => i + 1 < args.Length ? args[++i] : null;
            switch (args[i])
            {
                case "--version": version = Next(); break;
                case "--commit": commit = Next(); break;
                case "--source-repository": repository = Next(); break;
                case "--projects": projects = Next(); break;
                case "--require-publishable": requirePublishable = true; break;
                default:
                    if (args[i].StartsWith("--", StringComparison.Ordinal) || feed is not null)
                    {
                        Console.Error.WriteLine($"::error::verify_packages: unexpected argument '{args[i]}'");
                        return null;
                    }
                    feed = args[i];
                    break;
            }
        }

        if (feed is null || version is null || commit is null || repository is null || projects is null)
        {
            Console.Error.WriteLine("::error::verify_packages: usage: <feed-dir> --version <v> --commit <sha> " +
                                    "--source-repository <url> [--projects <dir>] [--require-publishable]");
            return null;
        }
        // A short or symbolic revision cannot be compared with the full SHA SourceLink records.
        if (!Regex.IsMatch(commit, "^[0-9a-f]{40}$"))
        {
            Console.Error.WriteLine($"::error::verify_packages: --commit must be a full 40-hex SHA, got '{commit}'");
            return null;
        }
        return new GateOptions(feed, version, commit, repository, projects, requirePublishable);
    }
}

/// <summary>What <see cref="PackageChecks.VerifyPackage" /> found in one package.</summary>
/// <param name="Id">The nuspec <c>&lt;id&gt;</c>, or <see langword="null" /> when the nuspec could not be read.</param>
/// <param name="RepositoryUrl">The nuspec <c>&lt;repository url&gt;</c>, or <see langword="null" /> when absent.</param>
/// <param name="Notes">Human-readable descriptions of the checks that passed.</param>
/// <param name="Problems">Every failed check; an empty list means the package is releasable.</param>
internal sealed record PackageReport(string? Id, string? RepositoryUrl, List<string> Notes, List<string> Problems);

/// <summary>The individual checks the gate runs on the feed and on each package.</summary>
internal static class PackageChecks
{
    /// <summary>Kind GUID of the portable-PDB custom debug information that holds the SourceLink JSON.</summary>
    private static readonly Guid SourceLinkKind = new("CC110556-A091-4D38-9FEC-25AB9A351A6A");

    /// <summary>
    ///     Stale-status phrases that must never reach nuget.org. The Core package shipped "This is early
    ///     scaffolding: the algorithm surface is not yet ported" long after the port was complete.
    /// </summary>
    private static readonly string[] StalePhrases = { "early scaffolding", "not yet ported" };

    /// <summary>The target frameworks every OptunaSharp package must carry (both are supported).</summary>
    private static readonly string[] RequiredFrameworks = { "net8.0", "net10.0" };

    /// <summary>
    ///     Reads the package set the release must contain: one package per packable project directly
    ///     under <paramref name="projectsDir" />.
    /// </summary>
    /// <param name="projectsDir">The directory whose <c>*/*.csproj</c> are the shipping projects (normally <c>src</c>).</param>
    /// <param name="failures">Receives a failure for each project whose package id cannot be read statically.</param>
    /// <returns>Package id → project path, compared case-insensitively as NuGet does.</returns>
    /// <remarks>
    ///     The id is read from the project XML (<c>PackageId</c>, else <c>AssemblyName</c>, else the file
    ///     name) instead of running MSBuild evaluation, which costs seconds per project. An id built from an
    ///     MSBuild expression is reported as a failure, so a project that outgrows this shortcut is noticed
    ///     rather than silently mis-matched.
    /// </remarks>
    public static Dictionary<string, string> ExpectedPackageIds(string projectsDir, List<string> failures)
    {
        var ids = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
        if (!Directory.Exists(projectsDir))
        {
            failures.Add($"projects directory '{projectsDir}' not found");
            return ids;
        }

        foreach (string csproj in Directory.GetDirectories(projectsDir)
                     .SelectMany(d => Directory.GetFiles(d, "*.csproj")).OrderBy(p => p))
        {
            XDocument doc = XDocument.Load(csproj);
            string? Property(string name) =>
                doc.Descendants().FirstOrDefault(e => e.Name.LocalName == name)?.Value.Trim();

            if (string.Equals(Property("IsPackable"), "false", StringComparison.OrdinalIgnoreCase)) continue;
            string id = Property("PackageId") ?? Property("AssemblyName") ?? Path.GetFileNameWithoutExtension(csproj);
            if (id.Contains("$(", StringComparison.Ordinal))
            {
                failures.Add($"{csproj}: PackageId '{id}' is an MSBuild expression; verify_packages reads it statically");
                continue;
            }
            ids[id] = csproj;
        }
        return ids;
    }

    /// <summary>
    ///     Runs every per-package check: the nuspec metadata, the dependency versions, the framework
    ///     folders, and each OptunaSharp assembly's version stamp and SourceLink.
    /// </summary>
    /// <param name="path">The <c>.nupkg</c> to inspect.</param>
    /// <param name="options">The expected version, commit and source repository.</param>
    /// <param name="expectedIds">The package family; a dependency on any of these must be pinned to <see cref="GateOptions.Version" />.</param>
    /// <returns>The package's id, repository URL, and passed and failed checks.</returns>
    /// <remarks>
    ///     A corrupt package (an unreadable zip, a missing nuspec) becomes a problem in the report and does
    ///     not abort the run, so one broken package cannot hide the state of the others.
    /// </remarks>
    public static PackageReport VerifyPackage(string path, GateOptions options, Dictionary<string, string> expectedIds)
    {
        var notes = new List<string>();
        var problems = new List<string>();
        string? id = null, repositoryUrl = null;
        try
        {
            using ZipArchive zip = ZipFile.OpenRead(path);
            ZipArchiveEntry? nuspecEntry = zip.Entries.FirstOrDefault(e =>
                !e.FullName.Contains('/') && e.Name.EndsWith(".nuspec", StringComparison.OrdinalIgnoreCase));
            if (nuspecEntry is null)
            {
                problems.Add("no .nuspec at the package root");
                return new PackageReport(null, null, notes, problems);
            }

            XElement metadata;
            using (Stream s = nuspecEntry.Open())
                metadata = XDocument.Load(s).Root?.Elements().FirstOrDefault(e => e.Name.LocalName == "metadata")
                           ?? throw new InvalidDataException("nuspec has no <metadata>");
            string? Meta(string name) => metadata.Elements().FirstOrDefault(e => e.Name.LocalName == name)?.Value.Trim();
            XElement? MetaElement(string name) => metadata.Elements().FirstOrDefault(e => e.Name.LocalName == name);
            bool HasEntry(string? name) =>
                name is not null && zip.Entries.Any(e => e.FullName.Equals(name.Replace('\\', '/'), StringComparison.OrdinalIgnoreCase));

            id = Meta("id");
            if (string.IsNullOrEmpty(id)) problems.Add("nuspec has no <id>");

            string? version = Meta("version");
            if (!string.Equals(version, options.Version, StringComparison.OrdinalIgnoreCase))
                problems.Add($"version is '{version}', expected '{options.Version}' (was -p:Version passed to pack?)");
            else notes.Add($"version {version}");

            CheckMetadataText(Meta, MetaElement, HasEntry, zip, notes, problems);

            XElement? repository = MetaElement("repository");
            repositoryUrl = repository?.Attribute("url")?.Value;
            string? commit = repository?.Attribute("commit")?.Value;
            if (string.IsNullOrEmpty(repositoryUrl)) problems.Add("nuspec has no <repository url>");
            if (!string.Equals(commit, options.Commit, StringComparison.OrdinalIgnoreCase))
                problems.Add($"<repository commit> is '{commit}', expected {options.Commit}");
            else notes.Add($"repository {repositoryUrl} @ {commit![..7]}");

            CheckDependencies(MetaElement("dependencies"), options.Version, expectedIds, notes, problems);
            CheckAssemblies(zip, options, notes, problems);
        }
        catch (Exception ex) when (ex is InvalidDataException or IOException or System.Xml.XmlException or BadImageFormatException)
        {
            problems.Add($"unreadable package: {ex.Message}");
        }
        return new PackageReport(id, repositoryUrl, notes, problems);
    }

    /// <summary>
    ///     Strips the forms under which the same GitHub repository is written (trailing <c>/</c>,
    ///     <c>.git</c>) so two URLs can be compared.
    /// </summary>
    /// <param name="url">A repository URL such as <c>https://github.com/SciSharp/OptunaSharp.git</c>.</param>
    /// <returns>The URL without a trailing slash or <c>.git</c> suffix.</returns>
    public static string NormalizeRepositoryUrl(string url)
    {
        string trimmed = url.Trim().TrimEnd('/');
        return trimmed.EndsWith(".git", StringComparison.OrdinalIgnoreCase) ? trimmed[..^4] : trimmed;
    }

    /// <summary>
    ///     Checks the nuget.org-facing metadata: authors, license, icon, readme, project URL, and that
    ///     neither the description nor the packed readme carries a stale status phrase.
    /// </summary>
    /// <param name="meta">Reads a nuspec metadata element's text by local name.</param>
    /// <param name="metaElement">Reads a nuspec metadata element by local name.</param>
    /// <param name="hasEntry">Tells whether the package contains a file at a nuspec-relative path.</param>
    /// <param name="zip">The package, to read the packed readme's text.</param>
    /// <param name="notes">Receives the checks that passed.</param>
    /// <param name="problems">Receives the checks that failed.</param>
    private static void CheckMetadataText(
        Func<string, string?> meta, Func<string, XElement?> metaElement, Func<string?, bool> hasEntry,
        ZipArchive zip, List<string> notes, List<string> problems)
    {
        // The list is shared with the other checks; only problems added here decide this check's note.
        int problemsBefore = problems.Count;
        if (string.IsNullOrEmpty(meta("authors"))) problems.Add("nuspec has no <authors>");
        if (string.IsNullOrEmpty(meta("projectUrl"))) problems.Add("nuspec has no <projectUrl>");

        XElement? license = metaElement("license");
        if (license?.Attribute("type")?.Value != "expression" || license.Value.Trim() != "MIT")
            problems.Add($"license is '{license?.Value}', expected the MIT license expression");

        string? icon = meta("icon");
        if (!hasEntry(icon)) problems.Add($"<icon> '{icon}' is missing or not inside the package");

        string? readme = meta("readme");
        string readmeText = "";
        if (!hasEntry(readme)) problems.Add($"<readme> '{readme}' is missing or not inside the package");
        else
        {
            using var reader = new StreamReader(zip.Entries.First(e =>
                e.FullName.Equals(readme!.Replace('\\', '/'), StringComparison.OrdinalIgnoreCase)).Open());
            readmeText = reader.ReadToEnd();
        }

        string description = meta("description") ?? "";
        if (description.Length == 0) problems.Add("nuspec has no <description>");
        foreach (string phrase in StalePhrases)
        {
            if (description.Contains(phrase, StringComparison.OrdinalIgnoreCase))
                problems.Add($"<description> still says \"{phrase}\"");
            if (readmeText.Contains(phrase, StringComparison.OrdinalIgnoreCase))
                problems.Add($"packed readme still says \"{phrase}\"");
        }
        if (problems.Count == problemsBefore) notes.Add("metadata: authors, MIT license, icon, readme, project URL, no stale text");
    }

    /// <summary>
    ///     Requires every dependency on another package of the family to be pinned to this release's
    ///     version, in every target-framework group.
    /// </summary>
    /// <param name="dependencies">The nuspec <c>&lt;dependencies&gt;</c> element, if any.</param>
    /// <param name="version">The release version.</param>
    /// <param name="familyIds">The OptunaSharp package ids.</param>
    /// <param name="notes">Receives the checks that passed.</param>
    /// <param name="problems">Receives the checks that failed.</param>
    /// <remarks>
    ///     A satellite packed without the release's <c>-p:Version</c> depends on <c>OptunaSharp 0.1.0</c>.
    ///     NuGet then resolves it against whatever Core the consumer has, which is a version the satellite
    ///     was never built or tested with.
    /// </remarks>
    private static void CheckDependencies(
        XElement? dependencies, string version, Dictionary<string, string> familyIds, List<string> notes, List<string> problems)
    {
        int pinned = 0;
        foreach (XElement dependency in dependencies?.Descendants().Where(e => e.Name.LocalName == "dependency")
                                         ?? Enumerable.Empty<XElement>())
        {
            string? depId = dependency.Attribute("id")?.Value;
            if (depId is null || !familyIds.ContainsKey(depId)) continue;

            // NuGet writes a plain minimum ("1.2.3") or an interval ("[1.2.3, )"); the lower bound is
            // what decides which Core a consumer gets.
            string range = dependency.Attribute("version")?.Value ?? "";
            string lower = range.Trim('[', '(', ' ').Split(',')[0].Trim();
            string framework = dependency.Parent?.Attribute("targetFramework")?.Value ?? "(any)";
            if (!string.Equals(lower, version, StringComparison.OrdinalIgnoreCase))
                problems.Add($"depends on {depId} {range} for {framework}; family dependencies must be pinned to {version}");
            else pinned++;
        }
        if (pinned > 0) notes.Add($"{pinned} OptunaSharp dependencies pinned to {version}");
    }

    /// <summary>
    ///     Checks the assembly layout and, for every OptunaSharp assembly, its version stamp and the
    ///     SourceLink of its PDB.
    /// </summary>
    /// <param name="zip">The package.</param>
    /// <param name="options">The expected version, commit and source repository.</param>
    /// <param name="notes">Receives the checks that passed.</param>
    /// <param name="problems">Receives the checks that failed.</param>
    /// <remarks>
    ///     A library ships <c>lib/&lt;tfm&gt;/</c> with an XML doc file and a PDB next to each of its own
    ///     assemblies. A <c>dotnet tool</c> (marked by <c>DotnetToolSettings.xml</c>) ships
    ///     <c>tools/&lt;tfm&gt;/any/</c> with its dependencies bundled. Third-party assemblies inside a tool
    ///     carry their publishers' versions, so only <c>OptunaSharp*.dll</c> is checked.
    /// </remarks>
    private static void CheckAssemblies(ZipArchive zip, GateOptions options, List<string> notes, List<string> problems)
    {
        bool isTool = zip.Entries.Any(e => e.Name.Equals("DotnetToolSettings.xml", StringComparison.OrdinalIgnoreCase));
        string root = isTool ? "tools" : "lib";
        foreach (string tfm in RequiredFrameworks)
        {
            string folder = isTool ? $"tools/{tfm}/any/" : $"lib/{tfm}/";
            if (!zip.Entries.Any(e => e.FullName.StartsWith(folder, StringComparison.OrdinalIgnoreCase)
                                      && e.FullName.EndsWith(".dll", StringComparison.OrdinalIgnoreCase)))
                problems.Add($"no assemblies under {folder} — every package targets {string.Join(" and ", RequiredFrameworks)}");
        }

        var ours = zip.Entries.Where(e => e.FullName.StartsWith(root + "/", StringComparison.OrdinalIgnoreCase)
                                          && e.Name.StartsWith("OptunaSharp", StringComparison.OrdinalIgnoreCase)
                                          && e.Name.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
                              .OrderBy(e => e.FullName).ToList();
        if (ours.Count == 0)
        {
            problems.Add($"no OptunaSharp*.dll under {root}/");
            return;
        }

        string numericVersion = options.Version.Split('-', '+')[0];
        int sourceLinked = 0;
        foreach (ZipArchiveEntry dll in ours)
        {
            string stem = dll.FullName[..^4];
            ZipArchiveEntry? pdb = zip.GetEntry(stem + ".pdb");
            // A library's own assemblies must come with docs and symbols. In a tool package only the PDBs
            // that were packed can be checked; the tool's own entry assembly must have one.
            if (!isTool && zip.GetEntry(stem + ".xml") is null) problems.Add($"{dll.FullName} has no XML doc file next to it");
            if (pdb is null)
            {
                if (!isTool) problems.Add($"{dll.FullName} has no PDB next to it");
                continue;
            }

            CheckAssemblyVersion(ReadAll(dll), dll.FullName, options.Version, numericVersion, problems);
            if (CheckSourceLink(ReadAll(pdb), pdb.FullName, options, problems)) sourceLinked++;
        }
        if (isTool && sourceLinked == 0) problems.Add("the tool ships no PDB for any OptunaSharp assembly");
        if (sourceLinked > 0)
            notes.Add($"{ours.Count} OptunaSharp assemblies stamped {options.Version}; {sourceLinked} SourceLinked to the commit");
    }

    /// <summary>
    ///     Requires the assembly's informational version (the source of <c>OptunaSharpVersion.Version</c>)
    ///     to be the release version, and its assembly version to be the release's numeric part.
    /// </summary>
    /// <param name="image">The assembly bytes.</param>
    /// <param name="name">The assembly's path in the package, for messages.</param>
    /// <param name="version">The release version, e.g. <c>1.2.3-rc.1</c>.</param>
    /// <param name="numericVersion">The release version's numeric part, e.g. <c>1.2.3</c>.</param>
    /// <param name="problems">Receives the checks that failed.</param>
    private static void CheckAssemblyVersion(byte[] image, string name, string version, string numericVersion, List<string> problems)
    {
        using var pe = new PEReader(new MemoryStream(image));
        if (!pe.HasMetadata)
        {
            problems.Add($"{name} is not a managed assembly");
            return;
        }
        MetadataReader md = pe.GetMetadataReader();
        AssemblyDefinition assembly = md.GetAssemblyDefinition();

        // The SDK appends "+<commit>" build metadata; only the part before it is the version.
        string? informational = ReadStringAttribute(md, assembly.GetCustomAttributes(), "System.Reflection.AssemblyInformationalVersionAttribute");
        string stamped = informational?.Split('+')[0] ?? "<none>";
        if (!string.Equals(stamped, version, StringComparison.OrdinalIgnoreCase))
            problems.Add($"{name} informational version is '{informational}', expected '{version}[+commit]'");

        // System.Version pads to four parts, so "1.2.3" is compared as 1.2.3.0.
        if (!System.Version.TryParse(numericVersion, out System.Version? numeric)
            || assembly.Version != new System.Version(numeric.Major, numeric.Minor, Math.Max(numeric.Build, 0), 0))
            problems.Add($"{name} assembly version is {assembly.Version}, expected {numericVersion}.0");
    }

    /// <summary>
    ///     Requires a portable PDB to carry SourceLink that maps its documents to
    ///     <c>https://raw.githubusercontent.com/&lt;owner&gt;/&lt;repo&gt;/&lt;commit&gt;/</c> of the repository and
    ///     commit this build ran from, and to use the deterministic <c>/_/</c> source root.
    /// </summary>
    /// <param name="pdbBytes">The PDB file's bytes.</param>
    /// <param name="name">The PDB's path in the package, for messages.</param>
    /// <param name="options">The expected commit and source repository.</param>
    /// <param name="problems">Receives the checks that failed.</param>
    /// <returns><see langword="true" /> when the PDB passed every SourceLink check.</returns>
    /// <remarks>
    ///     The <c>/_/</c> root is what <c>ContinuousIntegrationBuild=true</c> produces. Without it the PDB
    ///     embeds the build machine's checkout path, and SourceLink still works only because of the
    ///     mapping, so the check catches a pack that skipped the CI property.
    /// </remarks>
    private static bool CheckSourceLink(byte[] pdbBytes, string name, GateOptions options, List<string> problems)
    {
        // "BSJB" is the metadata signature; Windows (MSF) PDBs carry no SourceLink this check can read.
        if (pdbBytes.Length < 4 || pdbBytes[0] != (byte)'B' || pdbBytes[1] != (byte)'S' || pdbBytes[2] != (byte)'J' || pdbBytes[3] != (byte)'B')
        {
            problems.Add($"{name} is not a portable PDB");
            return false;
        }

        using MetadataReaderProvider provider = MetadataReaderProvider.FromPortablePdbStream(
            new MemoryStream(pdbBytes), MetadataStreamOptions.PrefetchMetadata);
        MetadataReader pdb = provider.GetMetadataReader();

        string? json = null;
        foreach (CustomDebugInformationHandle handle in pdb.CustomDebugInformation)
        {
            CustomDebugInformation info = pdb.GetCustomDebugInformation(handle);
            if (pdb.GetGuid(info.Kind) == SourceLinkKind) json = Encoding.UTF8.GetString(pdb.GetBlobBytes(info.Value));
        }
        if (json is null)
        {
            problems.Add($"{name} has no SourceLink (is Microsoft.SourceLink active, and was the build inside a git checkout?)");
            return false;
        }

        // Expected target: the raw-content URL of THIS repository at THIS commit. The owner/repo part is
        // case-insensitive on GitHub; the SHA is not.
        Uri source = new(NormalizeRepositoryUrl(options.SourceRepository));
        string expectedPrefix = $"https://raw.githubusercontent.com{source.AbsolutePath.TrimEnd('/')}/{options.Commit}/";
        var mappings = JsonDocument.Parse(json).RootElement.GetProperty("documents").EnumerateObject()
            .Select(p => (Key: p.Name, Target: p.Value.GetString() ?? "")).ToList();
        if (!mappings.Any(m => m.Target.StartsWith(expectedPrefix, StringComparison.OrdinalIgnoreCase)
                               && m.Target.Contains(options.Commit, StringComparison.Ordinal)))
        {
            problems.Add($"{name} SourceLink maps to [{string.Join(", ", mappings.Select(m => m.Target))}], expected {expectedPrefix}*");
            return false;
        }

        var documents = pdb.Documents.Select(h => pdb.GetString(pdb.GetDocument(h).Name)).ToList();
        if (documents.Count == 0 || documents.Any(d => !d.StartsWith("/_/", StringComparison.Ordinal)))
        {
            string example = documents.FirstOrDefault(d => !d.StartsWith("/_/", StringComparison.Ordinal)) ?? "(no documents)";
            problems.Add($"{name} records non-deterministic source paths such as '{example}' (pack with -p:ContinuousIntegrationBuild=true)");
            return false;
        }
        return true;
    }

    /// <summary>Reads the single string argument of a custom attribute, such as an assembly-level version attribute.</summary>
    /// <param name="md">The assembly's metadata.</param>
    /// <param name="attributes">The attributes to search.</param>
    /// <param name="fullTypeName">The attribute type, namespace-qualified.</param>
    /// <returns>The attribute's first constructor argument, or <see langword="null" /> when the attribute is absent.</returns>
    private static string? ReadStringAttribute(MetadataReader md, CustomAttributeHandleCollection attributes, string fullTypeName)
    {
        foreach (CustomAttributeHandle handle in attributes)
        {
            CustomAttribute attribute = md.GetCustomAttribute(handle);
            if (AttributeTypeName(md, attribute) != fullTypeName) continue;

            // ECMA-335 §II.23.3: a 0x0001 prolog, then the fixed arguments; a string is a SerString.
            BlobReader blob = md.GetBlobReader(attribute.Value);
            return blob.ReadUInt16() == 1 ? blob.ReadSerializedString() : null;
        }
        return null;
    }

    /// <summary>Resolves the namespace-qualified type name of a custom attribute from its constructor reference.</summary>
    /// <param name="md">The assembly's metadata.</param>
    /// <param name="attribute">The attribute.</param>
    /// <returns>The type name, or <see langword="null" /> for a constructor shape this gate does not decode.</returns>
    private static string? AttributeTypeName(MetadataReader md, CustomAttribute attribute)
    {
        // An attribute defined in another assembly (every BCL attribute) is referenced through a
        // MemberReference whose parent is a TypeReference; one defined locally is a MethodDefinition.
        switch (attribute.Constructor.Kind)
        {
            case HandleKind.MemberReference:
                MemberReference member = md.GetMemberReference((MemberReferenceHandle)attribute.Constructor);
                if (member.Parent.Kind != HandleKind.TypeReference) return null;
                TypeReference type = md.GetTypeReference((TypeReferenceHandle)member.Parent);
                return $"{md.GetString(type.Namespace)}.{md.GetString(type.Name)}";
            case HandleKind.MethodDefinition:
                TypeDefinition definition = md.GetTypeDefinition(md.GetMethodDefinition((MethodDefinitionHandle)attribute.Constructor).GetDeclaringType());
                return $"{md.GetString(definition.Namespace)}.{md.GetString(definition.Name)}";
            default:
                return null;
        }
    }

    /// <summary>Reads a zip entry fully into memory (PE and PDB readers need a seekable stream; a zip stream is not).</summary>
    /// <param name="entry">The entry.</param>
    /// <returns>The entry's decompressed bytes.</returns>
    private static byte[] ReadAll(ZipArchiveEntry entry)
    {
        using var buffer = new MemoryStream();
        using (Stream s = entry.Open()) s.CopyTo(buffer);
        return buffer.ToArray();
    }
}
