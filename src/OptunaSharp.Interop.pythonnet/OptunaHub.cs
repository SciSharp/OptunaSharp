using System;
using System.Collections.Generic;
using Python.Runtime;

namespace OptunaSharp.Interop.PythonNet;

/// <summary>
///     The .NET entry point to <b>OptunaHub</b> (https://hub.optuna.org/) — a faithful mirror of the
///     Python <c>optunahub.load_module</c> / <c>load_local_module</c> API. It imports a contributor
///     package from the OptunaHub registry (a GitHub repo; git sparse-checkout + a local cache, exactly
///     as the Python library does — this bridge just calls it) and returns an
///     <see cref="OptunaHubModule" /> whose samplers and benchmarks can be wrapped for use from an
///     OptunaSharp <see cref="Study" />.
///     <para>Requires a discoverable CPython with <c>pip install optuna optunahub</c>
///     (see <see cref="PythonRuntime" />).</para>
///     <example>
///     <code>
///     using var bbob = OptunaHub.LoadModule("benchmarks/bbob");
///     var problem = bbob.CreateProblem("Problem", new() { ["function_id"] = 1, ["dimension"] = 2 });
///     var study = Optuna.CreateStudy(directions: problem.Directions);
///     study.Optimize(problem.Objective, nTrials: 20);
///     </code>
///     </example>
/// </summary>
public static class OptunaHub
{
    /// <summary>
    ///     Import a package from the OptunaHub registry — mirrors
    ///     <c>optunahub.load_module(package, repo_owner=, repo_name=, ref=, force_reload=, base_url=)</c>.
    /// </summary>
    /// <param name="package">The package name, e.g. <c>"samplers/auto_sampler"</c> or <c>"benchmarks/bbob"</c>.</param>
    /// <param name="repoOwner">The registry repository owner (default <c>"optuna"</c>).</param>
    /// <param name="repoName">The registry repository name (default <c>"optunahub-registry"</c>).</param>
    /// <param name="gitRef">The git reference — branch, tag, or SHA (default <c>"main"</c>). Passed as Python <c>ref</c>.</param>
    /// <param name="forceReload">If <c>true</c>, re-download instead of using the local cache.</param>
    /// <param name="baseUrl">Optional base URL for a third-party registry (GitHub API/SSH/GHE/GitLab).</param>
    public static OptunaHubModule LoadModule(
        string package,
        string repoOwner = "optuna",
        string repoName = "optunahub-registry",
        string gitRef = "main",
        bool forceReload = false,
        string? baseUrl = null)
    {
        if (string.IsNullOrWhiteSpace(package))
            throw new ArgumentException("package must be non-empty, e.g. \"samplers/auto_sampler\".", nameof(package));

        return PythonRuntime.Run(() =>
        {
            using var loadModule = PythonRuntime.OptunaHubModuleRef.GetAttr("load_module");
            using var kwargs = new PyDict();
            kwargs["repo_owner"] = repoOwner.ToPython();
            kwargs["repo_name"] = repoName.ToPython();
            kwargs["ref"] = gitRef.ToPython();
            kwargs["force_reload"] = forceReload.ToPython();
            if (baseUrl is not null)
                kwargs["base_url"] = baseUrl.ToPython();

            using var pkg = package.ToPython();
            PyObject module;
            try
            {
                module = loadModule.Invoke(new[] { pkg }, kwargs);
            }
            catch (PythonException ex)
            {
                throw new OptunaHubInteropException(
                    $"optunahub.load_module(\"{package}\") failed. Check the package name against " +
                    "https://hub.optuna.org/, that `git` is on PATH (the default download path), and that " +
                    "the package's own requirements are installed.", ex);
            }

            return new OptunaHubModule(package, module);
        });
    }

    /// <summary>
    ///     Import a package from a LOCAL registry checkout — mirrors
    ///     <c>optunahub.load_local_module(package, registry_root=)</c>. Handy against the pinned
    ///     <c>refs/optunahub-registry</c> submodule (no network).
    /// </summary>
    /// <param name="package">The package name, e.g. <c>"samplers/simple"</c>.</param>
    /// <param name="registryRoot">The registry root directory that contains the <c>package/</c> tree.</param>
    public static OptunaHubModule LoadLocalModule(string package, string registryRoot)
    {
        if (string.IsNullOrWhiteSpace(package))
            throw new ArgumentException("package must be non-empty.", nameof(package));
        if (string.IsNullOrWhiteSpace(registryRoot))
            throw new ArgumentException("registryRoot must be non-empty.", nameof(registryRoot));

        return PythonRuntime.Run(() =>
        {
            using var loadLocal = PythonRuntime.OptunaHubModuleRef.GetAttr("load_local_module");
            using var kwargs = new PyDict();
            kwargs["registry_root"] = registryRoot.ToPython();
            using var pkg = package.ToPython();
            PyObject module;
            try
            {
                module = loadLocal.Invoke(new[] { pkg }, kwargs);
            }
            catch (PythonException ex)
            {
                throw new OptunaHubInteropException(
                    $"optunahub.load_local_module(\"{package}\", registry_root=\"{registryRoot}\") failed.", ex);
            }

            return new OptunaHubModule(package, module);
        });
    }
}
