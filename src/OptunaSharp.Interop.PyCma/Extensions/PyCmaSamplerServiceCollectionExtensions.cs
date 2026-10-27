using System;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Samplers;

namespace OptunaSharp.Interop.PyCma.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for <see cref="PyCmaSampler" /> (the bridge to
///     Nikolaus Hansen's <c>pycma</c>). Registers it as both its concrete type and the
///     <see cref="BaseSampler" /> abstraction, so it composes with Core's <c>AddOptunaStudyFactory()</c>.
///     <para>
///     <b>Lazy by design:</b> the convenience overload registers a <b>factory</b>, so the sampler (which
///     initializes embedded CPython and imports <c>cma</c> in its constructor, and emits a deprecation
///     <c>FutureWarning</c>) is built on first resolution, not when <c>AddPyCmaSampler</c> is called.
///     <b>One sampler per study:</b> a <c>PyCmaSampler</c> keeps per-study state (its search-space cache
///     and the seed it advances on every relative sample), so the convenience and factory overloads default to
///     <see cref="ServiceLifetime.Transient" /> and the study factory builds a sampler per study. The
///     instance overload registers a singleton shared by every study.
///     </para>
/// </summary>
public static class PyCmaSamplerServiceCollectionExtensions
{
    /// <summary>
    ///     Register a <see cref="PyCmaSampler" /> from its common options, built lazily on first
    ///     resolution. For the <c>x0</c> / <c>cma_stds</c> / <c>cma_opts</c> dictionaries or a custom
    ///     independent sampler, use the factory overload.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="sigma0">Initial standard deviation of CMA-ES, or <c>null</c> for the pycma default.</param>
    /// <param name="seed">RNG seed, or <c>null</c> for a pycma-random (non-reproducible) seed.</param>
    /// <param name="nStartupTrials">Independent trials before CMA-ES takes over; defaults to 1.</param>
    /// <param name="lifetime">
    ///     The registration lifetime; defaults to <see cref="ServiceLifetime.Transient" />, so every study the
    ///     study factory builds gets its own sampler. <see cref="ServiceLifetime.Singleton" /> shares one
    ///     sampler, and its per-study state, across all studies.
    /// </param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddPyCmaSampler(
        this IServiceCollection services,
        double? sigma0 = null,
        int? seed = null,
        int nStartupTrials = 1,
        ServiceLifetime lifetime = ServiceLifetime.Transient)
    {
        ArgumentNullException.ThrowIfNull(services);
        return Add(services, _ => new PyCmaSampler(sigma0: sigma0, seed: seed, nStartupTrials: nStartupTrials), lifetime);
    }

    /// <summary>Register a pre-built <see cref="PyCmaSampler" /> instance as the app's <see cref="BaseSampler" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="sampler">The sampler instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="sampler" /> is <c>null</c>.</exception>
    public static IServiceCollection AddPyCmaSampler(this IServiceCollection services, PyCmaSampler sampler)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(sampler);
        services.AddSingleton(sampler);
        services.AddSingleton<BaseSampler>(sampler);
        return services;
    }

    /// <summary>Register a <see cref="PyCmaSampler" /> produced by a factory (full control over all pycma options and the independent sampler).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the sampler from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">
    ///     The registration lifetime; defaults to <see cref="ServiceLifetime.Transient" />, so every study the
    ///     study factory builds gets its own sampler. <see cref="ServiceLifetime.Singleton" /> shares one
    ///     sampler, and its per-study state, across all studies.
    /// </param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddPyCmaSampler(
        this IServiceCollection services,
        Func<IServiceProvider, PyCmaSampler> factory,
        ServiceLifetime lifetime = ServiceLifetime.Transient)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return Add(services, factory, lifetime);
    }

    /// <summary>Registers the concrete sampler under the requested lifetime and forwards <see cref="BaseSampler" /> to it.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">The concrete-sampler factory.</param>
    /// <param name="lifetime">The lifetime shared by both registrations.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    private static IServiceCollection Add(
        IServiceCollection services,
        Func<IServiceProvider, PyCmaSampler> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(PyCmaSampler), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(BaseSampler), sp => sp.GetRequiredService<PyCmaSampler>(), lifetime));
        return services;
    }
}
