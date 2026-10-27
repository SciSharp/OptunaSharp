using System;
using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;

namespace OptunaSharp.Interop.PythonNet.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for OptunaHub contributor packages: register a
///     sampler or pruner loaded by name from the hub (<c>OptunaHub.LoadModule(...).CreateSampler(...)</c>)
///     as the app's <see cref="BaseSampler" /> / <see cref="BasePruner" />, so it composes with Core's
///     <c>AddOptunaStudyFactory()</c>.
///     <para>
///     <b>Why this fits DI:</b> a hub sampler/pruner is an <see cref="IDisposable" /> that owns a live
///     Python object, and the container that resolves it <b>disposes it for you</b>. <b>Lazy by design:</b>
///     the convenience overloads register a <b>factory</b>, so the hub package is downloaded / imported and
///     the object constructed on first resolution, not when the <c>Add…</c> method is called.
///     </para>
///     <para>
///     <b>One wrapper per study:</b> a hub sampler/pruner wrapper drives a synchronized shadow study, so it
///     serves exactly one single-worker study. The convenience and factory overloads therefore default to
///     <see cref="ServiceLifetime.Transient" />, and the study factory builds a wrapper per study (the hub
///     module itself is loaded once and cached). A singleton would feed every study's trials into one
///     shadow study.
///     </para>
///     <para>
///     <b>Disposal follows the resolving container:</b> a transient wrapper resolved inside a request scope
///     is disposed when that scope ends, and one resolved from the root provider at application shutdown.
///     A study that keeps optimizing after its scope ends needs the <see cref="ServiceLifetime.Singleton" />
///     lifetime (with one study at a time) or a study factory resolved from a longer-lived scope.
///     </para>
/// </summary>
public static class OptunaHubServiceCollectionExtensions
{
    /// <summary>
    ///     Register a sampler loaded from an OptunaHub package (e.g. package <c>"samplers/auto_sampler"</c>,
    ///     class <c>"AutoSampler"</c>) as the app's <see cref="BaseSampler" />, built lazily. The module
    ///     wrapper is disposed once the sampler is created — the sampler keeps its own live Python instance,
    ///     so it stays valid.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="package">The hub package name, e.g. <c>"samplers/auto_sampler"</c>.</param>
    /// <param name="className">The sampler class name within the package, e.g. <c>"AutoSampler"</c>.</param>
    /// <param name="kwargs">Constructor keyword arguments for the Python sampler, or <c>null</c>.</param>
    /// <param name="repoOwner">The registry repository owner; defaults to <c>"optuna"</c>.</param>
    /// <param name="repoName">The registry repository name; defaults to <c>"optunahub-registry"</c>.</param>
    /// <param name="gitRef">The git reference (branch/tag/SHA); defaults to <c>"main"</c>.</param>
    /// <param name="forceReload">Whether to re-download instead of using the local cache.</param>
    /// <param name="baseUrl">Optional base URL for a third-party registry.</param>
    /// <param name="lifetime">
    ///     The registration lifetime; defaults to <see cref="ServiceLifetime.Transient" />, so every study the
    ///     study factory builds gets its own wrapper and shadow study. <see cref="ServiceLifetime.Singleton" />
    ///     shares one wrapper, and one shadow study, across all studies.
    /// </param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="package" /> or <paramref name="className" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaHubSampler(
        this IServiceCollection services,
        string package,
        string className,
        Dictionary<string, object?>? kwargs = null,
        string repoOwner = "optuna",
        string repoName = "optunahub-registry",
        string gitRef = "main",
        bool forceReload = false,
        string? baseUrl = null,
        ServiceLifetime lifetime = ServiceLifetime.Transient)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(className);
        return AddSampler(services, _ =>
        {
            // Dispose the module wrapper as soon as the sampler is built: it only releases our reference to
            // the module object (which stays in sys.modules), while the sampler owns its own instance — so
            // the sampler outlives the wrapper, and the container owns the sampler's disposal.
            using var module = OptunaHub.LoadModule(package, repoOwner, repoName, gitRef, forceReload, baseUrl);
            return module.CreateSampler(className, kwargs);
        }, lifetime);
    }

    /// <summary>Register an <see cref="OptunaHubSampler" /> produced by a factory (full control over module loading).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the hub sampler from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">
    ///     The registration lifetime; defaults to <see cref="ServiceLifetime.Transient" />, so every study the
    ///     study factory builds gets its own wrapper and shadow study. <see cref="ServiceLifetime.Singleton" />
    ///     shares one wrapper, and one shadow study, across all studies.
    /// </param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaHubSampler(
        this IServiceCollection services,
        Func<IServiceProvider, OptunaHubSampler> factory,
        ServiceLifetime lifetime = ServiceLifetime.Transient)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return AddSampler(services, factory, lifetime);
    }

    /// <summary>
    ///     Register a pruner loaded from an OptunaHub package as the app's <see cref="BasePruner" />, built
    ///     lazily. As with the sampler, the module wrapper is disposed once the pruner is created.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="package">The hub package name, e.g. <c>"pruners/…"</c>.</param>
    /// <param name="className">The pruner class name within the package.</param>
    /// <param name="kwargs">Constructor keyword arguments for the Python pruner, or <c>null</c>.</param>
    /// <param name="repoOwner">The registry repository owner; defaults to <c>"optuna"</c>.</param>
    /// <param name="repoName">The registry repository name; defaults to <c>"optunahub-registry"</c>.</param>
    /// <param name="gitRef">The git reference (branch/tag/SHA); defaults to <c>"main"</c>.</param>
    /// <param name="forceReload">Whether to re-download instead of using the local cache.</param>
    /// <param name="baseUrl">Optional base URL for a third-party registry.</param>
    /// <param name="lifetime">
    ///     The registration lifetime; defaults to <see cref="ServiceLifetime.Transient" />, so every study the
    ///     study factory builds gets its own wrapper and shadow study. <see cref="ServiceLifetime.Singleton" />
    ///     shares one wrapper, and one shadow study, across all studies.
    /// </param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" />, <paramref name="package" /> or <paramref name="className" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaHubPruner(
        this IServiceCollection services,
        string package,
        string className,
        Dictionary<string, object?>? kwargs = null,
        string repoOwner = "optuna",
        string repoName = "optunahub-registry",
        string gitRef = "main",
        bool forceReload = false,
        string? baseUrl = null,
        ServiceLifetime lifetime = ServiceLifetime.Transient)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(package);
        ArgumentNullException.ThrowIfNull(className);
        return AddPruner(services, _ =>
        {
            using var module = OptunaHub.LoadModule(package, repoOwner, repoName, gitRef, forceReload, baseUrl);
            return module.CreatePruner(className, kwargs);
        }, lifetime);
    }

    /// <summary>Register an <see cref="OptunaHubPruner" /> produced by a factory (full control over module loading).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the hub pruner from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">
    ///     The registration lifetime; defaults to <see cref="ServiceLifetime.Transient" />, so every study the
    ///     study factory builds gets its own wrapper and shadow study. <see cref="ServiceLifetime.Singleton" />
    ///     shares one wrapper, and one shadow study, across all studies.
    /// </param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaHubPruner(
        this IServiceCollection services,
        Func<IServiceProvider, OptunaHubPruner> factory,
        ServiceLifetime lifetime = ServiceLifetime.Transient)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return AddPruner(services, factory, lifetime);
    }

    /// <summary>Registers the concrete hub sampler under the requested lifetime and forwards <see cref="BaseSampler" /> to it.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">The concrete hub-sampler factory.</param>
    /// <param name="lifetime">The lifetime shared by both registrations.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    private static IServiceCollection AddSampler(
        IServiceCollection services,
        Func<IServiceProvider, OptunaHubSampler> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(OptunaHubSampler), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(BaseSampler), sp => sp.GetRequiredService<OptunaHubSampler>(), lifetime));
        return services;
    }

    /// <summary>Registers the concrete hub pruner under the requested lifetime and forwards <see cref="BasePruner" /> to it.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">The concrete hub-pruner factory.</param>
    /// <param name="lifetime">The lifetime shared by both registrations.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    private static IServiceCollection AddPruner(
        IServiceCollection services,
        Func<IServiceProvider, OptunaHubPruner> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(OptunaHubPruner), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(BasePruner), sp => sp.GetRequiredService<OptunaHubPruner>(), lifetime));
        return services;
    }
}
