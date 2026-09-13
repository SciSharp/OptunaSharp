using System.Collections.Generic;

namespace OptunaSharp.Extensions;

/// <summary>
///     The container-bound defaults an <see cref="IOptunaStudyFactory" /> applies when a call to
///     <see cref="IOptunaStudyFactory.CreateStudy" /> leaves an argument unspecified. Configure it
///     from <c>AddOptunaStudyFactory(configure)</c> so a whole application shares one study shape
///     (direction, naming, conflict handling) while the sampler / pruner / storage come from the
///     other <c>AddOptuna*</c> registrations.
///     <para>
///     <b>Consequence:</b> a factory reads these only as fall-backs — an explicit argument to
///     <see cref="IOptunaStudyFactory.CreateStudy" /> always wins. Leaving <see cref="DefaultDirection" />
///     and <see cref="DefaultDirections" /> both <c>null</c> reproduces Optuna's own default (a
///     single-objective <c>minimize</c> study).
///     </para>
/// </summary>
/// <remarks>
///     <see cref="DefaultDirection" /> (single-objective) and <see cref="DefaultDirections" />
///     (multi-objective) are mutually exclusive, exactly like <c>optuna.create_study</c>'s
///     <c>direction</c> / <c>directions</c>: set at most one. Setting both is rejected when a study is
///     created (the factory forwards the conflict to <see cref="Optuna.CreateStudy" />, which raises
///     <see cref="NumSharp.ValueError" />), not when this object is populated.
/// </remarks>
public sealed class OptunaStudyOptions
{
    /// <summary>
    ///     The single-objective optimization direction used when a <see cref="IOptunaStudyFactory.CreateStudy" />
    ///     call supplies neither <c>direction</c> nor <c>directions</c>. <c>null</c> (the default) defers to
    ///     Optuna's own default of <see cref="StudyDirection.Minimize" />. Do not set this together with
    ///     <see cref="DefaultDirections" />.
    /// </summary>
    public StudyDirection? DefaultDirection { get; set; }

    /// <summary>
    ///     The multi-objective optimization directions used when a <see cref="IOptunaStudyFactory.CreateStudy" />
    ///     call supplies neither <c>direction</c> nor <c>directions</c>. <c>null</c> (the default) leaves the
    ///     study single-objective. Do not set this together with <see cref="DefaultDirection" />.
    /// </summary>
    public IReadOnlyList<StudyDirection>? DefaultDirections { get; set; }

    /// <summary>
    ///     The study name applied when a <see cref="IOptunaStudyFactory.CreateStudy" /> call passes no name.
    ///     <c>null</c> (the default) lets the storage generate a unique name. Pin a name only when the
    ///     factory targets a single named study (e.g. a persistent storage a worker reattaches to); reusing
    ///     one name across concurrent studies collides unless <see cref="LoadIfExists" /> is <c>true</c>.
    /// </summary>
    public string? DefaultStudyName { get; set; }

    /// <summary>
    ///     The default for <c>load_if_exists</c> — whether creating a study whose name already exists returns
    ///     the existing study (<c>true</c>) instead of raising <see cref="DuplicatedStudyError" />
    ///     (<c>false</c>, the default). Set <c>true</c> together with <see cref="DefaultStudyName" /> for the
    ///     "many workers, one shared study" pattern.
    /// </summary>
    public bool LoadIfExists { get; set; }
}
