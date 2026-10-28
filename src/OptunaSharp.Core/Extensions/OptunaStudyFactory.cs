using System;
using System.Collections.Generic;
using Microsoft.Extensions.DependencyInjection;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Extensions;

/// <summary>
///     The default <see cref="IOptunaStudyFactory" />: resolves the optional
///     <see cref="BaseSampler" /> / <see cref="BasePruner" /> / <see cref="BaseStorage" /> from the
///     service provider at each call and delegates to <see cref="Optuna.CreateStudy" />. Registered by
///     <see cref="OptunaServiceCollectionExtensions.AddOptunaStudyFactory" />; construct it directly only
///     when composing a container by hand.
/// </summary>
/// <remarks>
///     <para>
///         Services are resolved with <see cref="ServiceProviderServiceExtensions.GetService{T}(IServiceProvider)" />
///         (not required-service resolution) precisely so that an unregistered sampler/pruner/storage yields
///         <c>null</c> and Optuna's built-in default is used — matching a plain <c>optuna.create_study()</c>.
///     </para>
///     <para>
///         Resolution happens per <see cref="CreateStudy" /> call, against the provider passed to the
///         constructor. <see cref="OptunaServiceCollectionExtensions.AddOptunaStudyFactory" /> registers this
///         type as transient, so that provider is the one that resolved the factory: a request scope, whose
///         scoped registrations are then honored instead of being captured by the root, or the root
///         provider for a singleton consumer. A transient sampler or pruner (the default of the factory
///         registrations) is therefore a new instance per study, as a fresh <c>TPESampler()</c> is per
///         Python <c>create_study()</c> call. Sharing one stateful sampler across studies mixes their state.
///     </para>
///     <para>
///         <b>Disposal follows the resolving provider.</b> A sampler, pruner or storage that implements
///         <see cref="IDisposable" /> and is not a singleton is disposed when that provider is: at the end
///         of the request scope, or at application shutdown for the root. Every OptunaSharp storage that owns a
///         resource also implements <see cref="IAsyncDisposable" />, so an asynchronously disposed scope (ASP.NET
///         Core's request scopes) closes it with <c>DisposeAsync</c> and a synchronously disposed one with
///         <c>Dispose</c>. A study that keeps optimizing after its request scope ends needs a singleton
///         registration, or a factory resolved from a scope that lives as long as the study.
///     </para>
/// </remarks>
public sealed class OptunaStudyFactory : IOptunaStudyFactory
{
    private readonly IServiceProvider _services;
    private readonly OptunaStudyOptions _options;

    /// <summary>Creates a factory that draws its algorithm/storage from <paramref name="services" /> and its defaults from <paramref name="options" />.</summary>
    /// <param name="services">The provider queried (per call) for the optional sampler, pruner, and storage.</param>
    /// <param name="options">The default study shape applied when a <see cref="CreateStudy" /> argument is omitted.</param>
    /// <exception cref="ArgumentNullException"><paramref name="services" /> or <paramref name="options" /> is <c>null</c>.</exception>
    public OptunaStudyFactory(IServiceProvider services, OptunaStudyOptions options)
    {
        _services = services ?? throw new ArgumentNullException(nameof(services));
        _options = options ?? throw new ArgumentNullException(nameof(options));
    }

    /// <inheritdoc />
    public Study CreateStudy(
        string? studyName = null,
        StudyDirection? direction = null,
        bool? loadIfExists = null,
        IEnumerable<StudyDirection>? directions = null)
    {
        // Pull the pieces from the container; a missing one stays null so Optuna.CreateStudy applies its
        // own default (TPESampler / MedianPruner / InMemoryStorage) — the DI story must not change the
        // behavior of an un-configured study.
        var sampler = _services.GetService<BaseSampler>();
        var pruner = _services.GetService<BasePruner>();
        var storage = _services.GetService<BaseStorage>();

        // Merge call arguments over the option defaults. direction and directions are mutually exclusive
        // (Optuna raises if both are set), so only fill from options when BOTH call arguments are absent,
        // and then fill AT MOST one — preferring the caller's explicit values verbatim.
        StudyDirection? effectiveDirection = direction;
        IEnumerable<StudyDirection>? effectiveDirections = directions;
        if (effectiveDirection is null && effectiveDirections is null)
        {
            if (_options.DefaultDirections is not null)
                effectiveDirections = _options.DefaultDirections;
            else
                effectiveDirection = _options.DefaultDirection; // may be null → Optuna defaults to minimize
        }

        string? effectiveName = studyName ?? _options.DefaultStudyName;
        bool effectiveLoadIfExists = loadIfExists ?? _options.LoadIfExists;

        return Optuna.CreateStudy(
            storage: storage,
            sampler: sampler,
            pruner: pruner,
            studyName: effectiveName,
            direction: effectiveDirection,
            loadIfExists: effectiveLoadIfExists,
            directions: effectiveDirections);
    }
}
