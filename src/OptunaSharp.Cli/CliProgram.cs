using System;
using System.Collections.Generic;
using System.CommandLine;
using System.CommandLine.Help;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.Data.Sqlite;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Distributions;
using OptunaSharp.Internal;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;

namespace OptunaSharp.Cli;

using Column = OutputFormat.Column;

/// <summary>
///     The OptunaSharp command-line tool — a faithful port of <c>optuna/cli.py</c> onto
///     System.CommandLine. Every subcommand mirrors an Optuna <c>_BaseCommand</c> subclass; the shared
///     <c>--storage</c>/<c>--storage-class</c>/<c>-v</c>/<c>-q</c>/<c>--log-file</c>/<c>--debug</c>
///     options are Optuna's parent-parser arguments, modeled as System.CommandLine recursive (global)
///     options so they may appear before or after the subcommand (Optuna's <c>_preprocess_argv</c>).
///     Command output goes to stdout through <see cref="OutputFormat" /> so it is byte-identical to the
///     <c>optuna</c> command; diagnostics go to stderr through Core's logging seam.
/// </summary>
public static class CliProgram
{
    /// <summary>
    ///     The command name users type, the tool's <c>ToolCommandName</c> in <c>OptunaSharp.Cli.csproj</c>. Every
    ///     usage line in <c>--help</c> starts with it, as Python's starts with <c>optuna</c>.
    /// </summary>
    /// <remarks>
    ///     The root is a plain command carrying this name because System.CommandLine's <c>RootCommand</c> takes its
    ///     name from the entry assembly and has no setter. The assembly stays <c>OptunaSharp.Cli</c> (the
    ///     InternalsVisibleTo friend key names it), so the installed tool's help printed <c>Usage: OptunaSharp.Cli
    ///     …</c>, a command nobody can type, and an in-process host printed its own name (<c>testhost</c>).
    ///     Keep it equal to <c>ToolCommandName</c>.
    /// </remarks>
    internal const string CommandName = "optunasharp";

    private static readonly Logger Log = OptunaLogging.GetLogger("optuna.cli");

    // ---- common (recursive) options — Optuna's _add_common_arguments ---------------------------
    private static readonly Option<string?> StorageOpt = new("--storage")
    {
        Recursive = true,
        Description = "DB URL. (e.g. sqlite:///example.db) Also can be specified via OPTUNA_STORAGE environment variable.",
    };

    private static readonly Option<string?> StorageClassOpt = MakeStorageClassOption();

    private static readonly Option<bool> VerboseOpt = new("--verbose", "-v")
    {
        Recursive = true,
        Description = "Increase verbosity of output. Can be repeated.",
    };

    private static readonly Option<bool> QuietOpt = new("--quiet", "-q")
    {
        Recursive = true,
        Description = "Suppress output except warnings and errors.",
    };

    private static readonly Option<string?> LogFileOpt = new("--log-file")
    {
        Recursive = true,
        Description = "Specify a file to log output. Disabled by default.",
    };

    private static readonly Option<bool> DebugOpt = new("--debug")
    {
        Recursive = true,
        Description = "Show tracebacks on errors.",
    };

    /// <summary>Creates the storage-class option with supported names and deprecated aliases.</summary>
    /// <returns>The configured storage-class option.</returns>
    private static Option<string?> MakeStorageClassOption()
    {
        var opt = new Option<string?>("--storage-class")
        {
            Recursive = true,
            Description = "Storage class hint (e.g. JournalFileBackend)",
        };
        opt.AcceptOnlyFromAmong(StorageResolver.StorageClassChoices);
        return opt;
    }

    /// <summary>Programmatic entry point (the process <c>Main</c> forwards here); returns the exit code.</summary>
    /// <param name="args">Command-line arguments excluding the executable name.</param>
    /// <returns>The process exit code: zero on success and nonzero on parser or command failure.</returns>
    public static int Run(string[] args)
    {
        Command root = BuildRoot();
        ParseResult parseResult = root.Parse(args);
        SetVerbosity(parseResult);
        SetLogFile(parseResult);
        return parseResult.Invoke();
    }

    // ---- Optuna's _set_verbosity / _set_log_file -----------------------------------------------

    /// <summary>Applies the parsed quiet or verbose options to CLI logging.</summary>
    /// <param name="pr">The parsed command-line options for the current invocation.</param>
    private static void SetVerbosity(ParseResult pr)
    {
        // Optuna maps verbose_level {0: WARNING, 1: INFO(default), 2: DEBUG}; -q -> 0, -v (any count) -> 2.
        int level = pr.GetValue(QuietOpt) ? LogLevel.Warning : (pr.GetValue(VerboseOpt) ? LogLevel.Debug : LogLevel.Info);
        OptunaLogging.SetVerbosity(level);
    }

    /// <summary>Attaches the optional log-file destination selected by the command line.</summary>
    /// <param name="pr">The parsed command-line options for the current invocation.</param>
    private static void SetLogFile(ParseResult pr)
    {
        string? logFile = pr.GetValue(LogFileOpt);
        if (logFile is null)
            return;
        var writer = new StreamWriter(logFile, append: true) { AutoFlush = true };
        OptunaLogging.AddHandler(new StreamLogHandler(writer));
    }

    // ---- root/parser assembly — Optuna's _get_parser / _add_commands ---------------------------

    /// <summary>
    ///     Builds the root command, named <see cref="CommandName" />, with the global options, every subcommand and
    ///     the deferred action handlers.
    /// </summary>
    /// <returns>The configured root command; its action runs only when invoked by the parser.</returns>
    private static Command BuildRoot()
    {
        var root = new Command(
            CommandName, "OptunaSharp — the .NET port of Optuna. Hyperparameter optimization from the command line.");
        // RootCommand would have added these two itself; a plain Command needs them explicitly. RootCommand's third
        // default, the [suggest] directive for dotnet-suggest completion, has no Command equivalent and is not used.
        root.Add(new HelpOption());
        root.Add(new VersionOption());
        foreach (var opt in new Option[] { StorageOpt, StorageClassOpt, VerboseOpt, QuietOpt, LogFileOpt, DebugOpt })
            root.Add(opt);

        root.Add(BuildCreateStudy());
        root.Add(BuildDeleteStudy());
        root.Add(BuildStudy());        // parent of `set-user-attr`
        root.Add(BuildStudyNames());
        root.Add(BuildStudies());
        root.Add(BuildTrials());
        root.Add(BuildBestTrial());
        root.Add(BuildBestTrials());
        root.Add(BuildStorage());      // parent of `upgrade`
        root.Add(BuildAsk());
        root.Add(BuildTell());

        var help = new Command("help", "Show help message and exit.");
        help.SetAction(pr => root.Parse(new[] { "--help" }).Invoke());
        root.Add(help);

        // No subcommand -> show help (Optuna defaults empty argv to `help`).
        root.SetAction(pr => root.Parse(new[] { "--help" }).Invoke());
        return root;
    }

    // ---- create-study --------------------------------------------------------------------------

    /// <summary>Builds the create-study command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildCreateStudy()
    {
        var studyName = new Option<string?>("--study-name")
        {
            Description = "A human-readable name of a study to distinguish it from others.",
        };
        var direction = new Option<string?>("--direction")
        {
            Description = "Set direction of optimization to a new study. Set 'minimize' for minimization and 'maximize' for maximization.",
        };
        direction.AcceptOnlyFromAmong("minimize", "maximize");
        var skipIfExists = new Option<bool>("--skip-if-exists")
        {
            Description = "If specified, the creation of the study is skipped without any error when the study name is duplicated.",
        };
        var directions = new Option<string[]?>("--directions")
        {
            AllowMultipleArgumentsPerToken = true,
            Arity = ArgumentArity.OneOrMore,
            Description = "Set directions of optimization to a new study. Put whitespace between directions. Each direction should be either \"minimize\" or \"maximize\".",
        };
        directions.AcceptOnlyFromAmong("minimize", "maximize");

        var cmd = new Command("create-study", "Create a new study.");
        cmd.Add(studyName);
        cmd.Add(direction);
        cmd.Add(skipIfExists);
        cmd.Add(directions);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            BaseStorage storage = GetStorage(pr);
            string[]? dirs = pr.GetValue(directions);
            Study study = Optuna.CreateStudy(
                storage: storage,
                studyName: pr.GetValue(studyName),
                direction: ParseDirectionOrNull(pr.GetValue(direction)),
                directions: dirs is null || dirs.Length == 0 ? null : dirs.Select(ParseDirection),
                loadIfExists: pr.GetValue(skipIfExists));
            Print(study.StudyName);
            return 0;
        }));
        return cmd;
    }

    // ---- delete-study --------------------------------------------------------------------------

    /// <summary>Builds the delete-study command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildDeleteStudy()
    {
        var studyName = new Option<string?>("--study-name") { Description = "The name of the study to delete." };
        var cmd = new Command("delete-study", "Delete a specified study.");
        cmd.Add(studyName);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            BaseStorage storage = GetStorage(pr);
            int studyId = storage.GetStudyIdFromName(pr.GetValue(studyName)!);
            storage.DeleteStudy(studyId);
            return 0;
        }));
        return cmd;
    }

    // ---- study set-user-attr -------------------------------------------------------------------

    /// <summary>Builds the study command group with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildStudy()
    {
        var studyName = new Option<string?>("--study-name")
        {
            Required = true,
            Description = "The name of the study to set the user attribute to.",
        };
        var key = new Option<string?>("--key", "-k") { Required = true, Description = "Key of the user attribute." };
        var value = new Option<string?>("--value") { Required = true, Description = "Value to be set." };

        var setUserAttr = new Command("set-user-attr", "Set a user attribute to a study.");
        setUserAttr.Add(studyName);
        setUserAttr.Add(key);
        setUserAttr.Add(value);
        setUserAttr.SetAction(pr => RunAction(pr, setUserAttr, () =>
        {
            BaseStorage storage = GetStorage(pr);
            Study study = Optuna.LoadStudy(pr.GetValue(studyName), storage);
            study.SetUserAttr(pr.GetValue(key)!, pr.GetValue(value));
            Log.Info("Attribute successfully written.");
            return 0;
        }));

        var study = new Command("study", "Study-related commands.");
        study.Add(setUserAttr);
        study.SetAction(pr => study.Parse(new[] { "--help" }).Invoke());
        return study;
    }

    // ---- study-names ---------------------------------------------------------------------------

    /// <summary>Builds the study-names command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildStudyNames()
    {
        var format = FormatOption("value");
        var cmd = new Command("study-names", "Get all study names stored in a specified storage.");
        cmd.Add(format);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            BaseStorage storage = GetStorage(pr);
            var recordKey = new Column("name", "");
            var records = Optuna.GetAllStudyNames(storage)
                .Select(name => new Dictionary<Column, object?> { [recordKey] = name })
                .ToList();
            Print(OutputFormat.FormatList(records, new List<Column> { recordKey }, pr.GetValue(format), flatten: false));
            return 0;
        }));
        return cmd;
    }

    // ---- studies -------------------------------------------------------------------------------

    /// <summary>Builds the studies command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildStudies()
    {
        var format = FormatOption("table");
        var flatten = FlattenOption("Flatten nested columns such as directions.");
        var cmd = new Command("studies", "Show a list of studies.");
        cmd.Add(format);
        cmd.Add(flatten);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            BaseStorage storage = GetStorage(pr);
            var summaries = Optuna.GetAllStudySummaries(storage, includeBestTrial: false);

            var nameCol = new Column("name", "");
            var directionCol = new Column("direction", "");
            var nTrialsCol = new Column("n_trials", "");
            var startCol = new Column("datetime_start", "");
            var userAttrsCol = new Column("user_attrs", "");

            var records = new List<Dictionary<Column, object?>>();
            foreach (var s in summaries)
            {
                object? start = s.DatetimeStart?.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture);
                records.Add(new Dictionary<Column, object?>
                {
                    [nameCol] = s.StudyName,
                    [directionCol] = new PyTuple(s.Directions.Select(d => (object?)DirectionName(d))),
                    [nTrialsCol] = (long)s.NTrials,
                    [startCol] = start,
                    [userAttrsCol] = s.UserAttrs,
                });
            }

            var header = new List<Column> { nameCol, directionCol, nTrialsCol, startCol };
            if (records.Any(r => r[userAttrsCol] is OptunaSharp.Collections.OrderedDict<string, object?> { Count: > 0 }))
                header.Add(userAttrsCol);

            Print(OutputFormat.FormatList(records, header, pr.GetValue(format), pr.GetValue(flatten)));
            return 0;
        }));
        return cmd;
    }

    // ---- trials / best-trial / best-trials -----------------------------------------------------

    /// <summary>Builds the trials command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildTrials()
    {
        var (studyName, format, flatten) = TrialListingOptions("The name of the study which includes trials.");
        var cmd = new Command("trials", "Show a list of trials.");
        cmd.Add(studyName);
        cmd.Add(format);
        cmd.Add(flatten);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            Log.Warning("'trials' is an experimental CLI command. The interface can change in the future.");
            Study study = LoadStudyForListing(pr, studyName);
            var (records, columns) = Records.CreateRecordsAndAggregateColumn(study, Records.TrialAttrs(study));
            Print(OutputFormat.FormatList(records, columns, pr.GetValue(format), pr.GetValue(flatten)));
            return 0;
        }));
        return cmd;
    }

    /// <summary>Builds the best-trial command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildBestTrial()
    {
        var (studyName, format, flatten) = TrialListingOptions("The name of the study to get the best trial.");
        var cmd = new Command("best-trial", "Show the best trial.");
        cmd.Add(studyName);
        cmd.Add(format);
        cmd.Add(flatten);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            Log.Warning("'best-trial' is an experimental CLI command. The interface can change in the future.");
            Study study = LoadStudyForListing(pr, studyName);
            var (records, columns) = Records.CreateRecordsAndAggregateColumn(study, Records.TrialAttrs(study));
            Print(OutputFormat.FormatSingle(records[study.BestTrial.Number], columns, pr.GetValue(format), pr.GetValue(flatten)));
            return 0;
        }));
        return cmd;
    }

    /// <summary>Builds the best-trials command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildBestTrials()
    {
        var (studyName, format, flatten) = TrialListingOptions("The name of the study to get the best trials (trials at the Pareto front).");
        var cmd = new Command("best-trials", "Show a list of trials located at the Pareto front.");
        cmd.Add(studyName);
        cmd.Add(format);
        cmd.Add(flatten);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            Log.Warning("'best-trials' is an experimental CLI command. The interface can change in the future.");
            Study study = LoadStudyForListing(pr, studyName);
            var bestNumbers = study.BestTrials.Select(t => t.Number).ToHashSet();
            var (records, columns) = Records.CreateRecordsAndAggregateColumn(study, Records.TrialAttrs(study));
            var numberCol = new Column("number", "");
            var best = records.Where(r => bestNumbers.Contains((int)(long)r[numberCol]!)).ToList();
            Print(OutputFormat.FormatList(best, columns, pr.GetValue(format), pr.GetValue(flatten)));
            return 0;
        }));
        return cmd;
    }

    /// <summary>Creates the shared study, output-format, and flatten options for trial-listing commands.</summary>
    /// <param name="studyNameHelp">Help text describing the study-name option.</param>
    /// <returns>The study-name, output-format, and flatten options shared by a listing command.</returns>
    private static (Option<string?> StudyName, Option<string> Format, Option<bool> Flatten) TrialListingOptions(string studyNameHelp)
    {
        var studyName = new Option<string?>("--study-name") { Required = true, Description = studyNameHelp };
        return (studyName, FormatOption("table"), FlattenOption("Flatten nested columns such as params and user_attrs."));
    }

    /// <summary>Loads the named study from the parsed storage options for a listing command.</summary>
    /// <param name="pr">The parsed command-line options for the current invocation.</param>
    /// <param name="studyName">The study-name option whose value selects the stored study.</param>
    /// <returns>The loaded study used to produce output rows.</returns>
    private static Study LoadStudyForListing(ParseResult pr, Option<string?> studyName)
        => Optuna.LoadStudy(pr.GetValue(studyName), GetStorage(pr));

    // ---- storage upgrade -----------------------------------------------------------------------

    /// <summary>Builds the storage command group with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildStorage()
    {
        var upgrade = new Command("upgrade", "Upgrade the schema of an RDB storage.");
        upgrade.SetAction(pr => RunAction(pr, upgrade, () =>
        {
            // Port of optuna/cli.py `_StorageUpgrade.take_action`. Open with the compatibility check and
            // table creation skipped — so an out-of-date schema opens without raising and without the
            // head tables being created underneath the migration — then replay the alembic chain from
            // the storage's current revision to head.
            string url = StorageResolver.CheckStorageUrl(pr.GetValue(StorageOpt), Log);
            RDBStorage storage;
            try
            {
                storage = new RDBStorage(url, skipCompatibilityCheck: true, skipTableCreation: true);
            }
            catch (Exception e) when (e is SqliteException or ArgumentException)
            {
                Log.Error("Invalid RDBStorage URL.");
                return 1;
            }

            using (storage)
            {
                string currentVersion = storage.GetCurrentVersion();
                string headVersion = storage.GetHeadVersion();
                if (currentVersion == headVersion)
                {
                    Log.Info("This storage is up-to-date.");
                }
                else if (storage.GetAllVersions().Contains(currentVersion))
                {
                    Log.Info("Upgrading the storage schema to the latest version.");
                    storage.Upgrade();
                    Log.Info("Completed to upgrade the storage.");
                }
                else
                {
                    Log.Warning(
                        "Your optuna version seems outdated against the storage version. "
                        + "Please try updating optuna to the latest version by `$ pip install -U optuna`.");
                }
            }
            return 0;
        }));

        var storage = new Command("storage", "Storage-related commands.");
        storage.Add(upgrade);
        storage.SetAction(pr => storage.Parse(new[] { "--help" }).Invoke());
        return storage;
    }

    // ---- ask -----------------------------------------------------------------------------------

    /// <summary>Builds the ask command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildAsk()
    {
        var studyName = new Option<string?>("--study-name") { Description = "Name of study." };
        var sampler = new Option<string?>("--sampler") { Description = "Class name of sampler object to create." };
        var samplerKwargs = new Option<string?>("--sampler-kwargs") { Description = "Sampler object initialization keyword arguments as JSON." };
        var searchSpace = new Option<string?>("--search-space")
        {
            Description = "Search space as JSON. Keys are names and values are outputs from optuna.distributions.distribution_to_json.",
        };
        var format = FormatOption("json");
        var flatten = FlattenOption("Flatten nested columns such as params.");

        var cmd = new Command("ask", "Create a new trial and suggest parameters.");
        cmd.Add(studyName);
        cmd.Add(sampler);
        cmd.Add(samplerKwargs);
        cmd.Add(searchSpace);
        cmd.Add(format);
        cmd.Add(flatten);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            Log.Warning("'ask' is an experimental CLI command. The interface can change in the future.");
            BaseStorage storage = GetStorage(pr);

            string? samplerName = pr.GetValue(sampler);
            string? samplerKwargsJson = pr.GetValue(samplerKwargs);
            BaseSampler? samplerObj = null;
            if (samplerName is not null)
            {
                samplerObj = SamplerFactory.Create(samplerName, samplerKwargsJson);
            }
            else if (samplerKwargsJson is not null)
            {
                throw new NumSharp.ValueError(
                    "`--sampler_kwargs` is set without `--sampler`. Please specify `--sampler` as well or"
                    + " omit `--sampler-kwargs`.");
            }

            var fixedDistributions = new OrderedDict<string, BaseDistribution>();
            string? searchSpaceJson = pr.GetValue(searchSpace);
            if (searchSpaceJson is not null)
            {
                if (PyJson.Loads(searchSpaceJson) is not OrderedDict<string, object?> space)
                    throw new NumSharp.ValueError("`--search-space` must be a JSON object.");
                foreach (var (name, dist) in space)
                    fixedDistributions[name] = Distribution.FromJson(PyJson.Dumps(dist));
            }

            Study study;
            try
            {
                study = Optuna.LoadStudy(pr.GetValue(studyName), storage, samplerObj);
            }
            catch (NumSharp.KeyError)
            {
                throw new NumSharp.KeyError(
                    "Implicit study creation within the 'ask' command was dropped in Optuna v4.0.0. Please"
                    + " use the 'create-study' command beforehand.");
            }

            Trial trial = study.Ask(fixedDistributions);
            Log.Info($"Asked trial {trial.Number} with parameters {ReprParams(trial.Params)}.");

            var record = new Dictionary<Column, object?>();
            var columns = new List<Column>();
            var numberCol = new Column("number", "");
            record[numberCol] = (long)trial.Number;
            columns.Add(numberCol);

            bool doFlatten = pr.GetValue(flatten);
            if (trial.Params.Count == 0 && !doFlatten)
            {
                var paramsCol = new Column("params", "");
                record[paramsCol] = new System.Collections.Specialized.OrderedDictionary();
                columns.Add(paramsCol);
            }
            else
            {
                foreach (var (paramName, paramValue) in trial.Params)
                {
                    var col = new Column("params", paramName);
                    record[col] = paramValue;
                    columns.Add(col);
                }
            }

            Print(OutputFormat.FormatSingle(record, columns, pr.GetValue(format), doFlatten));
            return 0;
        }));
        return cmd;
    }

    // ---- tell ----------------------------------------------------------------------------------

    /// <summary>Builds the tell command with its options and deferred action handlers.</summary>
    /// <returns>The configured command; its action runs only when invoked by the parser.</returns>
    private static Command BuildTell()
    {
        var studyName = new Option<string?>("--study-name") { Description = "Name of study." };
        var trialNumber = new Option<int?>("--trial-number") { Description = "Trial number." };
        var values = new Option<double[]?>("--values")
        {
            AllowMultipleArgumentsPerToken = true,
            Arity = ArgumentArity.OneOrMore,
            Description = "Objective values.",
        };
        var state = new Option<string?>("--state") { Description = "Trial state." };
        state.AcceptOnlyFromAmong("complete", "pruned", "fail");
        var skipIfFinished = new Option<bool>("--skip-if-finished")
        {
            Description = "If specified, tell is skipped without any error when the trial is already finished.",
        };

        var cmd = new Command("tell", "Finish a trial, which was created by the ask command.");
        cmd.Add(studyName);
        cmd.Add(trialNumber);
        cmd.Add(values);
        cmd.Add(state);
        cmd.Add(skipIfFinished);
        cmd.SetAction(pr => RunAction(pr, cmd, () =>
        {
            Log.Warning("'tell' is an experimental CLI command. The interface can change in the future.");
            BaseStorage storage = GetStorage(pr);
            Study study = Optuna.LoadStudy(pr.GetValue(studyName), storage);

            TrialState? trialState = pr.GetValue(state) is { } s ? ParseState(s) : null;
            int? number = pr.GetValue(trialNumber);
            if (number is null)
                throw new NumSharp.ValueError("`--trial-number` is required.");
            double[]? objectiveValues = pr.GetValue(values);

            study.Tell(number.Value, objectiveValues, trialState, pr.GetValue(skipIfFinished));
            Log.Info($"Told trial {number.Value} with values {ReprValues(objectiveValues)} and state {StateRepr(trialState)}.");
            return 0;
        }));
        return cmd;
    }

    // ---- shared helpers ------------------------------------------------------------------------

    /// <summary>Resolves the storage backend selected by the CLI options and environment fallback.</summary>
    /// <param name="pr">The parsed command-line options for the current invocation.</param>
    /// <returns>The resolved SQLite or journal storage backend.</returns>
    /// <exception cref="CLIUsageError">The storage configuration is absent, unsupported, or cannot be inferred.</exception>
    private static BaseStorage GetStorage(ParseResult pr)
        => StorageResolver.GetStorage(pr.GetValue(StorageOpt), pr.GetValue(StorageClassOpt), Log);

    // Python's print() writes the string and a trailing newline through a text-mode stdout that
    // translates every '\n' to os.linesep. The formatter joins its lines with '\n', so match that
    // translation (Environment.NewLine == os.linesep on every platform) instead of Console.WriteLine,
    // which would leave the string's internal '\n' untranslated.
    /// <summary>Writes formatted output using Python-compatible platform newline handling.</summary>
    /// <param name="text">Text to write to standard output with platform newline translation.</param>
    private static void Print(string text)
        => Console.Out.Write(text.Replace("\n", Environment.NewLine) + Environment.NewLine);

    /// <summary>Creates an output-format option restricted to the supported renderer names.</summary>
    /// <param name="default">The format selected when the user does not specify --format.</param>
    /// <returns>The configured output-format option.</returns>
    private static Option<string> FormatOption(string @default)
    {
        var opt = new Option<string>("--format", "-f")
        {
            DefaultValueFactory = _ => @default,
            Description = "Output format.",
        };
        opt.AcceptOnlyFromAmong("value", "json", "table", "yaml");
        return opt;
    }

    /// <summary>Creates the flag that expands nested values into separate output columns.</summary>
    /// <param name="description">The user-visible help text for the flatten option.</param>
    /// <returns>The configured Boolean flatten option.</returns>
    private static Option<bool> FlattenOption(string description) => new("--flatten") { Description = description };

    /// <summary>Maps the validated direction option to the managed optimization direction.</summary>
    /// <param name="value">The scalar or composite value to parse, classify, or render.</param>
    /// <returns>Maximize for the maximize spelling; otherwise Minimize, following prior option validation.</returns>
    private static StudyDirection ParseDirection(string value) => value == "maximize" ? StudyDirection.Maximize : StudyDirection.Minimize;
    /// <summary>Maps an optional validated direction option, preserving an omitted value.</summary>
    /// <param name="value">The scalar or composite value to parse, classify, or render.</param>
    /// <returns>The selected direction, or null when omitted.</returns>
    private static StudyDirection? ParseDirectionOrNull(string? value) => value is null ? null : ParseDirection(value);

    /// <summary>Formats an optimization direction using the uppercase Python enum spelling.</summary>
    /// <param name="direction">The study optimization direction to display.</param>
    /// <returns>The uppercase Optuna direction name.</returns>
    private static string DirectionName(StudyDirection direction) => direction switch
    {
        StudyDirection.Minimize => "MINIMIZE",
        StudyDirection.Maximize => "MAXIMIZE",
        _ => "NOT_SET",
    };

    /// <summary>Parses the terminal trial-state names accepted by the tell command.</summary>
    /// <param name="value">The scalar or composite value to parse, classify, or render.</param>
    /// <returns>The terminal trial state represented by the supplied spelling.</returns>
    /// <exception cref="NumSharp.ValueError">The spelling is not complete, pruned, or fail.</exception>
    private static TrialState ParseState(string value) => value switch
    {
        "complete" => TrialState.Complete,
        "pruned" => TrialState.Pruned,
        "fail" => TrialState.Fail,
        _ => throw new NumSharp.ValueError($"Invalid trial state: {value}"),
    };

    // Diagnostics-only reprs (stderr; not part of the stdout parity surface).
    /// <summary>Formats parameter names and values for CLI diagnostic messages.</summary>
    /// <param name="parameters">Sampled parameter names and values to render in diagnostics.</param>
    /// <returns>The Python-style parameter mapping for diagnostics.</returns>
    private static string ReprParams(IReadOnlyDictionary<string, object?> parameters)
        => "{" + string.Join(", ", parameters.Select(kv => $"'{kv.Key}': {PythonRepr.Repr(kv.Value)}")) + "}";

    /// <summary>Formats objective values for CLI diagnostic messages, preserving the None spelling.</summary>
    /// <param name="values">The objective-value array, or null when no values were supplied.</param>
    /// <returns>The Python-style objective list, or None.</returns>
    private static string ReprValues(double[]? values)
        => values is null ? "None" : "[" + string.Join(", ", values.Select(PythonRepr.FloatRepr)) + "]";

    /// <summary>Formats a trial state as a Python-style enum expression for diagnostics.</summary>
    /// <param name="state">The trial lifecycle state to display.</param>
    /// <returns>The Python-style trial-state expression, or None.</returns>
    private static string StateRepr(TrialState? state) => state is null ? "None" : "TrialState." + Records.StateName(state.Value);

    // Optuna's main() special-cases CLIUsageError (nice message + the subcommand's help; a full trace
    // under --debug). System.CommandLine swallows other exceptions in its pipeline and returns 1, which
    // matches Optuna letting non-CLIUsageError propagate to a traceback.
    /// <summary>Executes a command and renders usage help for CLIUsageError while preserving other failures.</summary>
    /// <param name="pr">The parsed command-line options for the current invocation.</param>
    /// <param name="command">The current subcommand whose usage is displayed on a CLI usage error.</param>
    /// <param name="body">The command action returning its exit code; usage errors are translated into help output.</param>
    /// <returns>The action's exit code, or one when a CLI usage error is rendered.</returns>
    private static int RunAction(ParseResult pr, Command command, Func<int> body)
    {
        try
        {
            return body();
        }
        catch (CLIUsageError e)
        {
            if (pr.GetValue(DebugOpt))
                Log.Error(e.ToString());
            else
                Log.Error(e.Message);
            command.Parse(new[] { "--help" }).Invoke();
            return 1;
        }
    }
}
