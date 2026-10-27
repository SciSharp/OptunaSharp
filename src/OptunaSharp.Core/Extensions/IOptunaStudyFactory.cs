using System.Collections.Generic;

namespace OptunaSharp.Extensions;

/// <summary>
///     A dependency-injection composition seam that builds <see cref="Study" /> instances from the
///     sampler / pruner / storage registered in the container, applying the container-bound
///     <see cref="OptunaStudyOptions" /> defaults. Resolve it (constructor-inject
///     <c>IOptunaStudyFactory</c>) instead of calling <see cref="Optuna.CreateStudy" /> directly, so a
///     study's algorithm and persistence are configured once at the composition root rather than at
///     every call site.
///     <para>
///     <b>Why an interface (and why a factory, not a singleton <see cref="Study" />):</b> a study is a
///     per-run object — its name and direction vary, and a persistent storage may already hold one — so
///     it cannot be a container singleton. Each call mints (or re-attaches to) a study. The interface
///     makes the seam mockable in tests.
///     </para>
///     <para>
///     <b>Lifetime:</b> <c>AddOptunaStudyFactory</c> registers the factory as transient, so it resolves its
///     sampler, pruner and storage from the provider that resolved it: a request scope resolves scoped
///     registrations from that scope, and the root provider (a singleton or hosted service) resolves
///     singletons and transients. The sampler and pruner factory registrations default to transient, so
///     each study gets its own sampler and pruner unless one instance was registered on purpose.
///     </para>
/// </summary>
public interface IOptunaStudyFactory
{
    /// <summary>
    ///     Create (or, with <c>load_if_exists</c>, re-attach to) a <see cref="Study" /> using the
    ///     container's registered <c>BaseSampler</c> / <c>BasePruner</c> / <c>BaseStorage</c> (each optional
    ///     — an unregistered one falls back to Optuna's default: <c>TPESampler</c>, <c>MedianPruner</c>,
    ///     in-memory storage). Every parameter overrides the matching <see cref="OptunaStudyOptions" />
    ///     default for this one call.
    /// </summary>
    /// <param name="studyName">
    ///     The study name, or <c>null</c> to use <see cref="OptunaStudyOptions.DefaultStudyName" /> (and, if
    ///     that is also <c>null</c>, a storage-generated unique name).
    /// </param>
    /// <param name="direction">
    ///     The single-objective direction, or <c>null</c> to fall back to the options. Must not be combined
    ///     with <paramref name="directions" />.
    /// </param>
    /// <param name="loadIfExists">
    ///     Whether to return an existing same-named study instead of raising on conflict; <c>null</c> uses
    ///     <see cref="OptunaStudyOptions.LoadIfExists" />.
    /// </param>
    /// <param name="directions">
    ///     The multi-objective directions, or <c>null</c> to fall back to the options. Must not be combined
    ///     with <paramref name="direction" />.
    /// </param>
    /// <returns>A study bound to the resolved storage, using the resolved sampler and pruner.</returns>
    /// <exception cref="NumSharp.ValueError">
    ///     Both a single and a multi direction are in effect (from arguments or options), the directions
    ///     sequence is empty, or a direction is neither minimize nor maximize.
    /// </exception>
    /// <exception cref="DuplicatedStudyError">
    ///     The effective study name already exists and the effective <c>load_if_exists</c> is <c>false</c>.
    /// </exception>
    Study CreateStudy(
        string? studyName = null,
        StudyDirection? direction = null,
        bool? loadIfExists = null,
        IEnumerable<StudyDirection>? directions = null);
}
