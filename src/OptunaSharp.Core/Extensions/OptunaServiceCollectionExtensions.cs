using System;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for OptunaSharp's core building blocks: register
///     the app-wide <see cref="BaseSampler" />, <see cref="BasePruner" /> and <see cref="BaseStorage" />
///     once, then resolve an <see cref="IOptunaStudyFactory" /> that composes them into studies. This is
///     a .NET-ergonomics layer over the functional <see cref="Optuna.CreateStudy" /> API — it changes
///     nothing about how a study behaves, only where it is configured.
///     <para>
///     The registrations compose with the satellite packages: e.g.
///     <c>services.AddSqliteStorage("sqlite:///study.db").AddBoTorchSampler().AddOptunaStudyFactory()</c>
///     wires storage + sampler + factory across three assemblies with no glue code.
///     </para>
/// </summary>
/// <example>
///     <code>
///     services.AddOptunaSampler(new TPESampler(seed: 0));
///     services.AddOptunaStorage("sqlite:///study.db");
///     services.AddOptunaStudyFactory(o => o.DefaultDirection = StudyDirection.Maximize);
///     // elsewhere:  var study = provider.GetRequiredService&lt;IOptunaStudyFactory&gt;().CreateStudy("run-42");
///     </code>
/// </example>
public static class OptunaServiceCollectionExtensions
{
    // -------------------------------------------------------------------------------- sampler

    /// <summary>
    ///     Register a pre-built sampler as the application's <see cref="BaseSampler" /> singleton.
    ///     <b>Footgun:</b> a stateful sampler (one that must not be shared across studies — e.g. a
    ///     constraints-carrying sampler) is a poor singleton; prefer the factory overload with a
    ///     transient lifetime for those.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="sampler">The sampler instance shared by every study built from this container.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="sampler" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaSampler(this IServiceCollection services, BaseSampler sampler)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(sampler);
        services.AddSingleton(sampler);
        return services;
    }

    /// <summary>
    ///     Register a sampler produced by a factory, so its construction can depend on other services and
    ///     its lifetime can be tuned (use <see cref="ServiceLifetime.Transient" /> for a per-study stateful
    ///     sampler).
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the sampler from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaSampler(
        this IServiceCollection services,
        Func<IServiceProvider, BaseSampler> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        services.Add(new ServiceDescriptor(typeof(BaseSampler), factory, lifetime));
        return services;
    }

    // -------------------------------------------------------------------------------- pruner

    /// <summary>Register a pre-built pruner as the application's <see cref="BasePruner" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="pruner">The pruner instance shared by every study built from this container.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="pruner" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaPruner(this IServiceCollection services, BasePruner pruner)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(pruner);
        services.AddSingleton(pruner);
        return services;
    }

    /// <summary>Register a pruner produced by a factory, so its construction can depend on other services.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the pruner from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaPruner(
        this IServiceCollection services,
        Func<IServiceProvider, BasePruner> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        services.Add(new ServiceDescriptor(typeof(BasePruner), factory, lifetime));
        return services;
    }

    // -------------------------------------------------------------------------------- storage

    /// <summary>Register a pre-built storage as the application's <see cref="BaseStorage" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="storage">The storage instance shared by every study built from this container.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="storage" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaStorage(this IServiceCollection services, BaseStorage storage)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(storage);
        services.AddSingleton(storage);
        return services;
    }

    /// <summary>
    ///     Register a storage resolved from an Optuna storage URL (the string form of
    ///     <c>optuna.create_study(storage=...)</c>). The satellite storage packages that own a URL scheme
    ///     (<c>sqlite:///…</c>, <c>redis://…</c>, …) must be referenced for their scheme to resolve; the
    ///     in-repo default handles the journal-file / in-memory forms.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="storageUrl">The storage URL passed to <see cref="StorageFunctions.GetStorage(string)" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="storageUrl" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaStorage(
        this IServiceCollection services,
        string storageUrl,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(storageUrl);
        services.Add(new ServiceDescriptor(typeof(BaseStorage), _ => StorageFunctions.GetStorage(storageUrl), lifetime));
        return services;
    }

    /// <summary>Register a storage produced by a factory, so its construction can depend on other services.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the storage from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaStorage(
        this IServiceCollection services,
        Func<IServiceProvider, BaseStorage> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        services.Add(new ServiceDescriptor(typeof(BaseStorage), factory, lifetime));
        return services;
    }

    // -------------------------------------------------------------------------------- study factory

    /// <summary>
    ///     Register the <see cref="IOptunaStudyFactory" /> (and its <see cref="OptunaStudyOptions" />) that
    ///     composes the container's sampler / pruner / storage into studies. Idempotent — a second call is a
    ///     no-op (it uses <c>TryAdd</c>), so the first <paramref name="configure" /> wins; call it once at the
    ///     composition root.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="configure">Optional callback to set the default study shape (direction, name, load-if-exists).</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddOptunaStudyFactory(
        this IServiceCollection services,
        Action<OptunaStudyOptions>? configure = null)
    {
        ArgumentNullException.ThrowIfNull(services);

        // Build and freeze the options here so the factory closes over the exact values configured by THIS
        // call — the factory does not resolve OptunaStudyOptions from the container, which keeps the
        // "first registration wins" semantics predictable even if the type is registered elsewhere.
        var options = new OptunaStudyOptions();
        configure?.Invoke(options);

        services.TryAddSingleton(options);
        services.TryAddSingleton<IOptunaStudyFactory>(sp => new OptunaStudyFactory(sp, options));
        return services;
    }
}
