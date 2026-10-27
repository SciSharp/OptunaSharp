using System;
using System.Collections.Generic;
using System.Linq;
using NumSharp;
using OptunaSharp.Collections;
using OptunaSharp.Logging;
using OptunaSharp.Pruners;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp;

/// <summary>
///     The top-level Optuna API surface: ports of the module-level functions re-exported by
///     <c>optuna/__init__.py</c> (<c>create_study</c>, <c>load_study</c>, <c>delete_study</c>,
///     <c>copy_study</c>, <c>get_all_study_summaries</c>, <c>get_all_study_names</c>).
/// </summary>
public static class Optuna
{
    private static readonly Logger _logger = OptunaLogging.GetLogger("optuna.study.study");

    /// <summary>
    ///     Create a new <see cref="Study" />. Port of <c>optuna.create_study</c>.
    /// </summary>
    /// <param name="storage">
    ///     Storage object. If this argument is set to <c>null</c>,
    ///     <see cref="InMemoryStorage" /> is used, and the <see cref="Study" /> will not be
    ///     persistent.
    /// </param>
    /// <param name="sampler">
    ///     A sampler object that implements background algorithm for value suggestion. If
    ///     <c>null</c> is specified, <see cref="TPESampler" /> is used as the default.
    /// </param>
    /// <param name="pruner">
    ///     A pruner object that decides early stopping of unpromising trials. If <c>null</c> is
    ///     specified, <see cref="MedianPruner" /> is used as the default.
    /// </param>
    /// <param name="studyName">
    ///     Study's name. If this argument is set to <c>null</c>, a unique name is generated
    ///     automatically.
    /// </param>
    /// <param name="direction">
    ///     Direction of optimization. <c>direction</c> and <c>directions</c> must not be
    ///     specified at the same time. If neither is specified, the direction is set to
    ///     minimize.
    /// </param>
    /// <param name="loadIfExists">
    ///     Flag to control the behavior to handle a conflict of study names. In the case where a
    ///     study named <c>studyName</c> already exists in the storage, a
    ///     <see cref="DuplicatedStudyError" /> is raised if <c>loadIfExists</c> is set to
    ///     <c>false</c>. Otherwise, the creation of the study is skipped, and the existing one is
    ///     returned.
    /// </param>
    /// <param name="directions">A sequence of directions during multi-objective optimization.</param>
    /// <returns>A study bound to the resolved storage, using the requested sampler and pruner.</returns>
    /// <exception cref="NumSharp.ValueError">Both direction forms are supplied, the directions sequence is empty, or a direction is neither Minimize nor Maximize.</exception>
    public static Study CreateStudy(
        BaseStorage? storage = null,
        BaseSampler? sampler = null,
        BasePruner? pruner = null,
        string? studyName = null,
        StudyDirection? direction = null,
        bool loadIfExists = false,
        IEnumerable<StudyDirection>? directions = null)
    {
        List<StudyDirection> directionList;
        if (direction is null && directions is null)
        {
            directionList = new List<StudyDirection> { StudyDirection.Minimize };
        }
        else if (direction is not null && directions is not null)
        {
            throw new ValueError("Specify only one of `direction` and `directions`.");
        }
        else if (direction is not null)
        {
            directionList = new List<StudyDirection> { direction.Value };
        }
        else
        {
            directionList = directions!.ToList();
        }

        // Direction count defines objective dimensionality throughout storage and sampling; reject
        // an ambiguous or empty configuration before creating a persistent study record.
        if (directionList.Count < 1)
            throw new ValueError("The number of objectives must be greater than 0.");
        if (directionList.Any(d => d != StudyDirection.Minimize && d != StudyDirection.Maximize))
            throw new ValueError(
                $"`directions` must be a list of `minimize` or `maximize`, but got " +
                $"[{string.Join(", ", directionList.Select(d => "StudyDirection." + d.PythonName()))}]. " +
                "For single-objective optimization, please use `direction` instead of `directions`.");

        var resolvedStorage = StorageFunctions.GetStorage(storage);
        int studyId;
        try
        {
            studyId = resolvedStorage.CreateNewStudy(directionList, studyName);
        }
        catch (DuplicatedStudyError)
        {
            if (loadIfExists)
            {
                System.Diagnostics.Debug.Assert(studyName is not null);

                _logger.Info(
                    $"Using an existing study with `study_name={Internal.PythonRepr.Repr(studyName)}` instead of creating a new one.");
                studyId = resolvedStorage.GetStudyIdFromName(studyName!);
            }
            else
            {
                throw new DuplicatedStudyError(
                    $"Another study with study_name={Internal.PythonRepr.Repr(studyName)} already exists. Please specify a name not in " +
                    "the storage, or reuse the existing one by setting `load_if_exists` (for " +
                    "Python API) or `--skip-if-exists` flag (for CLI).\n" +
                    "Use `optuna.study.get_all_study_names(storage)` to list all the used names.");
            }
        }

        string resolvedStudyName = resolvedStorage.GetStudyNameFromId(studyId);
        var study = new Study(studyName: resolvedStudyName, storage: resolvedStorage, sampler: sampler, pruner: pruner);

        return study;
    }

    /// <summary>
    ///     Load the existing <see cref="Study" /> that has the specified name.
    ///     Port of <c>optuna.load_study</c>.
    /// </summary>
    /// <param name="studyName">
    ///     Study's name. Each study has a unique name as an identifier. If <c>null</c>, checks
    ///     whether the storage contains a single study, and if so loads that study.
    /// </param>
    /// <param name="storage">
    ///     Storage object (or use the OptunaSharp.Storage.* packages for URLs). The <see cref="Study" />
    ///     constructor wraps an RDB backend in a <see cref="CachedStorage" />, as Python's <c>load_study</c>
    ///     does, so distributed workers that load a shared study read finished trials from the cache.
    /// </param>
    /// <param name="sampler">Sampler, defaulting to <see cref="TPESampler" />.</param>
    /// <param name="pruner">Pruner, defaulting to <see cref="MedianPruner" />.</param>
    /// <returns>The existing study with this invocation's sampler and pruner.</returns>
    /// <exception cref="NumSharp.ValueError">The name is omitted and storage does not contain exactly one study.</exception>
    /// <exception cref="NumSharp.KeyError">The named study does not exist.</exception>
    public static Study LoadStudy(
        string? studyName,
        BaseStorage storage,
        BaseSampler? sampler = null,
        BasePruner? pruner = null)
    {
        if (studyName is null)
        {
            var studyNames = GetAllStudyNames(storage);
            if (studyNames.Count != 1)
                throw new ValueError(
                    $"Could not determine the study name since the storage does not contain exactly" +
                    $" 1 study. Specify `study_name` from study_names=[{string.Join(", ", studyNames.Select(Internal.PythonRepr.StringRepr))}].");
            studyName = studyNames[0];
            _logger.Info(
                $"Study name was omitted but trying to load `study_name={Internal.PythonRepr.Repr(studyName)}` because that was the only " +
                "study found in the storage.");
        }

        return new Study(studyName: studyName, storage: storage, sampler: sampler, pruner: pruner);
    }

    /// <summary>
    ///     Delete a <see cref="Study" /> object. Port of <c>optuna.delete_study</c>.
    /// </summary>
    /// <param name="studyName">Name identifying the study in the selected storage.</param>
    /// <param name="storage">Storage backend holding study and trial records.</param>
    /// <exception cref="NumSharp.KeyError">The named study does not exist.</exception>
    public static void DeleteStudy(string studyName, BaseStorage storage)
    {
        var resolvedStorage = StorageFunctions.GetStorage(storage);
        int studyId = resolvedStorage.GetStudyIdFromName(studyName);
        resolvedStorage.DeleteStudy(studyId);
    }

    /// <summary>
    ///     Copy study from one storage to another. Port of <c>optuna.copy_study</c>.
    ///
    ///     The direction(s) of the objective(s) in the study, trials, user attributes and system
    ///     attributes are copied.
    /// </summary>
    /// <exception cref="DuplicatedStudyError">
    ///     If a study with a conflicting name already exists in the destination storage.
    /// </exception>
    /// <param name="fromStudyName">Name of the existing source study.</param>
    /// <param name="fromStorage">Storage from which metadata and trials are read.</param>
    /// <param name="toStorage">Storage in which the new study is created.</param>
    /// <param name="toStudyName">Destination name; null reuses the source name.</param>
    /// <exception cref="NumSharp.KeyError">The source study is absent.</exception>
    /// <exception cref="NumSharp.ValueError">A copied trial has an objective count incompatible with the study.</exception>
    /// <remarks>The copy is incremental. A failure can leave the destination study with only part of its metadata or trials.</remarks>
    public static void CopyStudy(
        string fromStudyName,
        BaseStorage fromStorage,
        BaseStorage toStorage,
        string? toStudyName = null)
    {
        var fromStudy = LoadStudy(studyName: fromStudyName, storage: fromStorage);
        // Backends need not share a transaction. The new study remains visible if a later metadata
        // or trial copy fails, matching the incremental copy contract.
        var toStudy = CreateStudy(
            studyName: toStudyName ?? fromStudyName,
            storage: toStorage,
            directions: fromStudy.Directions,
            loadIfExists: false);

        foreach (var (key, value) in fromStudy.Storage.GetStudySystemAttrs(fromStudy.StudyId))
            toStudy.Storage.SetStudySystemAttr(toStudy.StudyId, key, value);

        foreach (var (key, value) in fromStudy.UserAttrs)
            toStudy.SetUserAttr(key, value);

        // Trials are deep copied on `add_trials`.
        foreach (var trial in fromStudy.GetTrials(deepcopy: false))
        {
            if (trial.Values is not null && toStudy.Directions.Count != trial.Values.Count)
                throw new ValueError(
                    $"The added trial has {trial.Values.Count} values, which is different from the " +
                    $"number of objectives {toStudy.Directions.Count} in the study (determined by " +
                    "Study.directions).");
            toStudy.Storage.CreateNewTrial(toStudy.StudyId, templateTrial: trial);
        }
    }

    /// <summary>
    ///     Get all history of studies stored in a specified storage.
    ///     Port of <c>optuna.get_all_study_summaries</c>.
    /// </summary>
    /// <param name="storage">Storage object.</param>
    /// <param name="includeBestTrial">
    ///     Include the best trials if exist. It potentially increases the number of queries and
    ///     may take longer to fetch summaries depending on the storage.
    /// </param>
    /// <returns>Summaries in storage order, including counts and optional single-objective best trials.</returns>
    public static List<StudySummary> GetAllStudySummaries(BaseStorage storage, bool includeBestTrial = true)
    {
        var resolvedStorage = StorageFunctions.GetStorage(storage);
        var frozenStudies = resolvedStorage.GetAllStudies();
        var studySummaries = new List<StudySummary>();

        foreach (var s in frozenStudies)
        {
            var allTrials = resolvedStorage.GetAllTrials(s.StudyId);
            var completedTrials = allTrials.Where(t => t.State == TrialState.Complete).ToList();

            int nTrials = allTrials.Count;

            StudyDirection? direction;
            IEnumerable<StudyDirection>? directions;
            FrozenTrial? bestTrial;
            if (s.Directions.Count == 1)
            {
                direction = s.Direction;
                directions = null;
                if (includeBestTrial && completedTrials.Count != 0)
                {
                    bestTrial = direction == StudyDirection.Maximize
                        ? completedTrials.Aggregate((best, t) => t.Value! > best.Value! ? t : best)
                        : completedTrials.Aggregate((best, t) => t.Value! < best.Value! ? t : best);
                }
                else
                {
                    bestTrial = null;
                }
            }
            else
            {
                direction = null;
                directions = s.Directions;
                bestTrial = null;
            }

            var startTimes = allTrials
                .Where(t => t.DatetimeStart is not null)
                .Select(t => t.DatetimeStart!.Value)
                .ToList();
            DateTime? datetimeStart = startTimes.Count == 0 ? null : startTimes.Min();

            studySummaries.Add(new StudySummary(
                studyName: s.StudyName,
                direction: direction,
                bestTrial: bestTrial,
                userAttrs: s.UserAttrs,
                nTrials: nTrials,
                datetimeStart: datetimeStart,
                studyId: s.StudyId,
                directions: directions));
        }

        return studySummaries;
    }

    /// <summary>
    ///     Get all study names stored in a specified storage.
    ///     Port of <c>optuna.get_all_study_names</c>.
    /// </summary>
    /// <param name="storage">Storage backend holding study and trial records.</param>
    /// <returns>Names of all studies in the order returned by the storage backend.</returns>
    public static List<string> GetAllStudyNames(BaseStorage storage)
    {
        var resolvedStorage = StorageFunctions.GetStorage(storage);
        return resolvedStorage.GetAllStudies().Select(study => study.StudyName).ToList();
    }
}
