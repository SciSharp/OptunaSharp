using System;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Importance;
using OptunaSharp.Importance.Booster;
using OptunaSharp.Importance.Forests;
using OptunaSharp.Integration;

namespace OptunaSharp.Integration.MLNet.Extensions;

/// <summary>
///     Microsoft.Extensions.DependencyInjection wire-up for the ML.NET integration: the FastForest /
///     FastTree / FastTreeTweedie importance evaluators and the <see cref="MLNetEstimator" /> cross-
///     validation adapter (the ML.NET backend for the Core <c>OptunaSearchCV</c>).
///     <para>
///     Each evaluator is registered as its concrete type plus the <see cref="BaseImportanceEvaluator" />
///     abstraction; when several evaluators are registered, resolve the concrete type for a deterministic
///     result (a bare <see cref="BaseImportanceEvaluator" /> resolves to the last registered). The
///     estimator is registered as <see cref="MLNetEstimator" /> and <see cref="ICrossValidationEstimator" />.
///     <b>Parity reminder:</b> the canonical fANOVA/MDI evaluators live in Core (value-parity with
///     Optuna); FastForest here is ranking-parity, and the FastTree/FastTreeTweedie boosters are off-label
///     (no value-parity) — see each class's remarks.
///     </para>
/// </summary>
public static class MLNetServiceCollectionExtensions
{
    // ------------------------------------------------------------------ FastForest (faithful bagging backend)

    /// <summary>Register the FastForest fANOVA importance evaluator as its concrete type and <see cref="BaseImportanceEvaluator" />.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="nTrees">The number of trees in the forest; defaults to 64.</param>
    /// <param name="maxDepth">The maximum tree depth; defaults to 64.</param>
    /// <param name="seed">The ensemble seed, or <c>null</c> for nondeterministic fitting.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddFastForestFanovaImportanceEvaluator(
        this IServiceCollection services, int nTrees = 64, int maxDepth = 64, int? seed = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return AddEvaluator(services, _ => new FastForestFanovaImportanceEvaluator(nTrees, maxDepth, seed), lifetime);
    }

    /// <summary>Register the FastForest mean-decrease-impurity (MDI) importance evaluator as its concrete type and <see cref="BaseImportanceEvaluator" />.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="nTrees">The number of trees in the forest; defaults to 64.</param>
    /// <param name="maxDepth">The maximum tree depth; defaults to 64.</param>
    /// <param name="seed">The ensemble seed, or <c>null</c> for nondeterministic fitting.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddFastForestMeanDecreaseImpurityImportanceEvaluator(
        this IServiceCollection services, int nTrees = 64, int maxDepth = 64, int? seed = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return AddEvaluator(services, _ => new FastForestMeanDecreaseImpurityImportanceEvaluator(nTrees, maxDepth, seed), lifetime);
    }

    // ------------------------------------------------------------------ FastTree / FastTreeTweedie (off-label boosters)

    /// <summary>Register the FastTree (booster, off-label) fANOVA importance evaluator as its concrete type and <see cref="BaseImportanceEvaluator" />.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="nTrees">The number of boosting iterations; defaults to 64.</param>
    /// <param name="maxDepth">The maximum tree depth; defaults to 64.</param>
    /// <param name="seed">The ensemble seed, or <c>null</c> for nondeterministic fitting.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddFastTreeFanovaImportanceEvaluator(
        this IServiceCollection services, int nTrees = 64, int maxDepth = 64, int? seed = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return AddEvaluator(services, _ => new FastTreeFanovaImportanceEvaluator(nTrees, maxDepth, seed), lifetime);
    }

    /// <summary>Register the FastTree (booster, off-label) MDI importance evaluator as its concrete type and <see cref="BaseImportanceEvaluator" />.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="nTrees">The number of boosting iterations; defaults to 64.</param>
    /// <param name="maxDepth">The maximum tree depth; defaults to 64.</param>
    /// <param name="seed">The ensemble seed, or <c>null</c> for nondeterministic fitting.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddFastTreeMeanDecreaseImpurityImportanceEvaluator(
        this IServiceCollection services, int nTrees = 64, int maxDepth = 64, int? seed = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return AddEvaluator(services, _ => new FastTreeMeanDecreaseImpurityImportanceEvaluator(nTrees, maxDepth, seed), lifetime);
    }

    /// <summary>Register the FastTreeTweedie (booster, off-label) fANOVA importance evaluator as its concrete type and <see cref="BaseImportanceEvaluator" />.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="nTrees">The number of boosting iterations; defaults to 64.</param>
    /// <param name="maxDepth">The maximum tree depth; defaults to 64.</param>
    /// <param name="seed">The ensemble seed, or <c>null</c> for nondeterministic fitting.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddFastTreeTweedieFanovaImportanceEvaluator(
        this IServiceCollection services, int nTrees = 64, int maxDepth = 64, int? seed = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return AddEvaluator(services, _ => new FastTreeTweedieFanovaImportanceEvaluator(nTrees, maxDepth, seed), lifetime);
    }

    /// <summary>Register the FastTreeTweedie (booster, off-label) MDI importance evaluator as its concrete type and <see cref="BaseImportanceEvaluator" />.</summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="nTrees">The number of boosting iterations; defaults to 64.</param>
    /// <param name="maxDepth">The maximum tree depth; defaults to 64.</param>
    /// <param name="seed">The ensemble seed, or <c>null</c> for nondeterministic fitting.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> is <c>null</c>.</exception>
    public static IServiceCollection AddFastTreeTweedieMeanDecreaseImpurityImportanceEvaluator(
        this IServiceCollection services, int nTrees = 64, int maxDepth = 64, int? seed = null,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        return AddEvaluator(services, _ => new FastTreeTweedieMeanDecreaseImpurityImportanceEvaluator(nTrees, maxDepth, seed), lifetime);
    }

    // ------------------------------------------------------------------ generic factory + estimator

    /// <summary>
    ///     Register any ML.NET importance evaluator produced by a factory — the full-control path for the
    ///     evaluators' <c>Options</c>-based / <c>configure</c> constructors. Registers the concrete type and
    ///     forwards <see cref="BaseImportanceEvaluator" /> to it.
    /// </summary>
    /// <typeparam name="TEvaluator">The concrete ML.NET importance-evaluator type.</typeparam>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the evaluator from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddMLNetImportanceEvaluator<TEvaluator>(
        this IServiceCollection services,
        Func<IServiceProvider, TEvaluator> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
        where TEvaluator : BaseImportanceEvaluator
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        return AddEvaluator(services, factory, lifetime);
    }

    /// <summary>
    ///     Register an <see cref="MLNetEstimator" /> (the ML.NET cross-validation adapter for the Core
    ///     <c>OptunaSearchCV</c>) as its concrete type and <see cref="ICrossValidationEstimator" />. Factory-only
    ///     because the estimator needs a live <c>MLContext</c>, an <c>IDataView</c>, a pipeline factory, and a
    ///     scorer that cannot be defaulted.
    /// </summary>
    /// <param name="services">The service collection to add to.</param>
    /// <param name="factory">Builds the estimator from the resolved <see cref="IServiceProvider" />.</param>
    /// <param name="lifetime">The registration lifetime; defaults to <see cref="ServiceLifetime.Singleton" />.</param>
    /// <returns>The same <paramref name="services" />, for chaining.</returns>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="factory" /> is <c>null</c>.</exception>
    public static IServiceCollection AddMLNetEstimator(
        this IServiceCollection services,
        Func<IServiceProvider, MLNetEstimator> factory,
        ServiceLifetime lifetime = ServiceLifetime.Singleton)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(factory);
        services.Add(new ServiceDescriptor(typeof(MLNetEstimator), sp => factory(sp), lifetime));
        services.Add(new ServiceDescriptor(typeof(ICrossValidationEstimator), sp => sp.GetRequiredService<MLNetEstimator>(), lifetime));
        return services;
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
