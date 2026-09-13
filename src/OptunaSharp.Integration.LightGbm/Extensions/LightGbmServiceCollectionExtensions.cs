using System;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Importance;

namespace OptunaSharp.Integration.LightGbm.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the LightGBM integration: the ML.NET-backed
///     LightGBM training engine (<see cref="MlNetLightGbmTrainer" />, the pluggable
///     <see cref="ILightGbmTrainer" /> / <see cref="ILightGbmBoosterStore" /> the tuner depends on) and
///     the off-label LightGBM importance evaluators.
///     <para>
///     <b>The trainer is the strong DI case</b> — it is the swappable strategy the LightGBM tuner takes,
///     so registering it (and injecting <see cref="ILightGbmTrainer" />) is idiomatic. The importance
///     evaluators are registered as their concrete type plus the <see cref="BaseImportanceEvaluator" />
///     abstraction; if you register more than one evaluator, resolve the concrete type for a deterministic
///     result (a bare <see cref="BaseImportanceEvaluator" /> resolves to the last registered).
///     <b>Reminder:</b> fANOVA/MDI over a LightGBM booster is a non-standard quantity with NO value-parity
///     to Optuna — prefer the canonical evaluators in Core when parity matters.
///     </para>
/// </summary>
public static class LightGbmServiceCollectionExtensions
{
    /// <summary>
    ///     Register the ML.NET LightGBM training engine as the app's <see cref="ILightGbmTrainer" /> and
    ///     <see cref="ILightGbmBoosterStore" /> (and its concrete type) — the engine the LightGBM tuner
    ///     resolves. One instance backs all three service types.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddLightGbmTrainer(
        this IServiceCollection services,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        services.Add(new ServiceDescriptor(typeof(MlNetLightGbmTrainer), _ => new MlNetLightGbmTrainer(), lifetime));
        // ILightGbmTrainer and ILightGbmBoosterStore forward to the ONE MlNetLightGbmTrainer instance so a
        // tuner's trainer and the booster store it persists to are the same object.
        services.Add(new ServiceDescriptor(typeof(ILightGbmTrainer), sp => sp.GetRequiredService<MlNetLightGbmTrainer>(), lifetime));
        services.Add(new ServiceDescriptor(typeof(ILightGbmBoosterStore), sp => sp.GetRequiredService<MlNetLightGbmTrainer>(), lifetime));
        return services;
    }

    /// <summary>Register a caller-supplied <see cref="ILightGbmTrainer" /> instance as the app's training engine.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="trainer">The training engine to share.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="trainer" /> is <c>null</c>.</exception>
    public static IServiceCollection AddLightGbmTrainer(this IServiceCollection services, ILightGbmTrainer trainer)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(trainer);
        services.AddSingleton(trainer);
        // Forward the booster store only when the supplied trainer also implements it.
        if (trainer is ILightGbmBoosterStore store)
            services.AddSingleton(store);
        return services;
    }

    /// <summary>
    ///     Register a LightGBM fANOVA importance evaluator (off-label — see the class remarks) as its
    ///     concrete type and <see cref="BaseImportanceEvaluator" />.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="nTrees">The number of boosting iterations; defaults to 64.</param>
    /// <param name="maxDepth">The tree depth the leaf count derives from; defaults to 64.</param>
    /// <param name="seed">The ensemble seed, or <c>null</c> for nondeterministic fitting.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddLightGbmFanovaImportanceEvaluator(
        this IServiceCollection services,
        int nTrees = 64,
        int maxDepth = 64,
        int? seed = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return AddEvaluator<LightGbmFanovaImportanceEvaluator>(
            services, _ => new LightGbmFanovaImportanceEvaluator(nTrees, maxDepth, seed), lifetime);
    }

    /// <summary>Register a LightGBM fANOVA importance evaluator produced by a factory (the full ML.NET options surface).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the evaluator from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddLightGbmFanovaImportanceEvaluator(
        this IServiceCollection services,
        Func<IServiceProvider, LightGbmFanovaImportanceEvaluator> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return AddEvaluator(services, factory, lifetime);
    }

    /// <summary>
    ///     Register a LightGBM mean-decrease-impurity (MDI) importance evaluator (off-label — see the class
    ///     remarks) as its concrete type and <see cref="BaseImportanceEvaluator" />.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="nTrees">The number of boosting iterations; defaults to 64.</param>
    /// <param name="maxDepth">The tree depth the leaf count derives from; defaults to 64.</param>
    /// <param name="seed">The ensemble seed, or <c>null</c> for nondeterministic fitting.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddLightGbmMeanDecreaseImpurityImportanceEvaluator(
        this IServiceCollection services,
        int nTrees = 64,
        int maxDepth = 64,
        int? seed = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return AddEvaluator<LightGbmMeanDecreaseImpurityImportanceEvaluator>(
            services, _ => new LightGbmMeanDecreaseImpurityImportanceEvaluator(nTrees, maxDepth, seed), lifetime);
    }

    /// <summary>Register a LightGBM MDI importance evaluator produced by a factory (the full ML.NET options surface).</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the evaluator from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddLightGbmMeanDecreaseImpurityImportanceEvaluator(
        this IServiceCollection services,
        Func<IServiceProvider, LightGbmMeanDecreaseImpurityImportanceEvaluator> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return AddEvaluator(services, factory, lifetime);
    }

    /// <summary>Registers a concrete evaluator under the requested lifetime and forwards <see cref="BaseImportanceEvaluator" /> to it.</summary>
    /// <typeparam name="TEvaluator">The concrete importance-evaluator type.</typeparam>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">The concrete-evaluator factory.</param>
    /// <param name="lifetime">The lifetime shared by both registrations.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    private static IServiceCollection AddEvaluator<TEvaluator>(
        IServiceCollection services,
        Func<IServiceProvider, TEvaluator> factory,
        ServiceLifetime lifetime)
        where TEvaluator : BaseImportanceEvaluator
    {
        services.Add(new ServiceDescriptor(typeof(TEvaluator), sp => factory(sp), lifetime));
        services.Add(new ServiceDescriptor(typeof(BaseImportanceEvaluator), sp => sp.GetRequiredService<TEvaluator>(), lifetime));
        return services;
    }
}
