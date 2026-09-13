using System;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Samplers;

namespace OptunaSharp.Interop.Pymoo.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for <see cref="PymooSampler" /> (the bridge to
///     the <c>pymoo</c> multi-objective algorithm zoo). Registers it as both its concrete type and the
///     <see cref="BaseSampler" /> abstraction, so it composes with Core's <c>AddOptunaStudyFactory()</c>.
///     <para>
///     <b>Lazy by design:</b> the convenience overload registers a <b>factory</b>, so the sampler (which
///     initializes embedded CPython and imports <c>pymoo</c> in its constructor) is built on first
///     resolution, not when <c>AddPymooSampler</c> is called.
///     </para>
/// </summary>
public static class PymooSamplerServiceCollectionExtensions
{
    /// <summary>
    ///     Register a <see cref="PymooSampler" /> from its common options, built lazily on first
    ///     resolution. For a constraints function or a custom independent sampler, use the factory overload.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="algorithm">The pymoo algorithm to drive; defaults to <see cref="PymooAlgorithm.Nsga2" />.</param>
    /// <param name="populationSize">The evolutionary population size (≥ 2); defaults to 50.</param>
    /// <param name="seed">RNG seed, or <c>null</c> for a non-reproducible seed.</param>
    /// <param name="options">Algorithm-specific options (SBX/PM, reference directions, RVEA/R-NSGA knobs), or <c>null</c> for defaults.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddPymooSampler(
        this IServiceCollection services,
        PymooAlgorithm algorithm = PymooAlgorithm.Nsga2,
        int populationSize = 50,
        int? seed = null,
        PymooAlgorithmOptions? options = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return Add(
            services,
            _ => new PymooSampler(algorithm: algorithm, populationSize: populationSize, seed: seed, options: options),
            lifetime);
    }

    /// <summary>Register a pre-built <see cref="PymooSampler" /> instance as the app's <see cref="BaseSampler" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="sampler">The sampler instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="sampler" /> is <c>null</c>.</exception>
    public static IServiceCollection AddPymooSampler(this IServiceCollection services, PymooSampler sampler)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(sampler);
        services.AddSingleton(sampler);
        services.AddSingleton<BaseSampler>(sampler);
        return services;
    }

    /// <summary>Register a <see cref="PymooSampler" /> produced by a factory (full control over constraints and the independent sampler).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the sampler from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddPymooSampler(
        this IServiceCollection services,
        Func<IServiceProvider, PymooSampler> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
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
        Func<IServiceProvider, PymooSampler> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(PymooSampler), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(BaseSampler), sp => sp.GetRequiredService<PymooSampler>(), lifetime));
        return services;
    }
}
