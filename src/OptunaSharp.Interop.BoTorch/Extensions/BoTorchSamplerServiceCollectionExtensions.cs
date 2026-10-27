using System;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Samplers;

namespace OptunaSharp.Interop.BoTorch.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for <see cref="BoTorchSampler" />. Registers it
///     as both its concrete type and the <see cref="BaseSampler" /> abstraction, so it composes with
///     Core's <c>AddOptunaStudyFactory()</c>.
///     <para>
///     <b>Lazy by design:</b> the convenience overload registers a <b>factory</b>, so the sampler (which
///     initializes embedded CPython and imports <c>botorch</c> in its constructor) is built on first
///     resolution, not when <c>AddBoTorchSampler</c> is called — registration stays cheap and
///     Python-free. <b>One sampler per study:</b> a <c>BoTorchSampler</c> keeps per-study state (its
///     search-space cache, and the constraints bookkeeping when a constraints function is set), so the
///     convenience and factory overloads default to <see cref="ServiceLifetime.Transient" /> and the study
///     factory builds a sampler per study. CPython and <c>botorch</c> are imported once per process, so
///     later instances are cheap. The instance overload registers a singleton shared by every study.
///     </para>
/// </summary>
public static class BoTorchSamplerServiceCollectionExtensions
{
    /// <summary>
    ///     Register a <see cref="BoTorchSampler" /> from its common options, built lazily on first
    ///     resolution. For a constraints function or a custom independent sampler, use the factory overload.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="candidatesFunc">The acquisition function to optimize; defaults to <see cref="BoTorchCandidatesFunc.Auto" />.</param>
    /// <param name="nStartupTrials">Independent trials before BoTorch takes over; defaults to 10.</param>
    /// <param name="considerRunningTrials">Whether the acquisition function considers running trials; defaults to <c>false</c>.</param>
    /// <param name="seed">RNG seed, or <c>null</c> for non-reproducible sampling.</param>
    /// <param name="device">The torch device (<c>"cpu"</c>, <c>"cuda"</c>, …); <c>null</c> uses CPU.</param>
    /// <param name="lifetime">
    ///     The registration lifetime; defaults to <see cref="ServiceLifetime.Transient" />, so every study the
    ///     study factory builds gets its own sampler. <see cref="ServiceLifetime.Singleton" /> shares one
    ///     sampler, and its per-study state, across all studies.
    /// </param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddBoTorchSampler(
        this IServiceCollection services,
        BoTorchCandidatesFunc candidatesFunc = BoTorchCandidatesFunc.Auto,
        int nStartupTrials = 10,
        bool considerRunningTrials = false,
        int? seed = null,
        string? device = null,
        ServiceLifetime lifetime = ServiceLifetime.Transient)
    {
        ArgumentNullException.ThrowIfNull(services);
        return Add(
            services,
            _ => new BoTorchSampler(
                candidatesFunc: candidatesFunc,
                nStartupTrials: nStartupTrials,
                considerRunningTrials: considerRunningTrials,
                seed: seed,
                device: device),
            lifetime);
    }

    /// <summary>Register a pre-built <see cref="BoTorchSampler" /> instance as the app's <see cref="BaseSampler" /> singleton.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="sampler">The sampler instance to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="sampler" /> is <c>null</c>.</exception>
    public static IServiceCollection AddBoTorchSampler(this IServiceCollection services, BoTorchSampler sampler)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(sampler);
        services.AddSingleton(sampler);
        services.AddSingleton<BaseSampler>(sampler);
        return services;
    }

    /// <summary>Register a <see cref="BoTorchSampler" /> produced by a factory (full control over constraints, the independent sampler, and lifetime).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the sampler from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">
    ///     The registration lifetime; defaults to <see cref="ServiceLifetime.Transient" />, so every study the
    ///     study factory builds gets its own sampler. <see cref="ServiceLifetime.Singleton" /> shares one
    ///     sampler, and its per-study state, across all studies.
    /// </param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddBoTorchSampler(
        this IServiceCollection services,
        Func<IServiceProvider, BoTorchSampler> factory,
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
        Func<IServiceProvider, BoTorchSampler> factory,
        ServiceLifetime lifetime)
    {
        services.Add(new ServiceDescriptor(typeof(BoTorchSampler), factory, lifetime));
        services.Add(new ServiceDescriptor(typeof(BaseSampler), sp => sp.GetRequiredService<BoTorchSampler>(), lifetime));
        return services;
    }
}
