// MLNetModelTuning — tune an ML.NET model with OptunaSharp, with real training in-process: every trial trains
// ML.NET's FastTree on five cross-validation folds. No Python and no surrogate objective.
//
// The data is Friedman's #1 regression benchmark (FriedmanData.cs): 1,000 rows, of which 80% tune the model and
// 20% are held out. OptunaSearchCV samples four FastTree hyperparameters with TPE, and MLNetEstimator
// cross-validates each candidate. The study is saved in SQLite, so running the program again continues the same
// search. Afterwards the program refits the best configuration, scores it on the rows the search never saw, ranks
// the hyperparameters with fANOVA, and saves the model and two Plotly charts.
//
// This is OptunaSharp's functional API, without dependency injection: in a console app the constructors are the
// simplest composition. TuningService and TuningWorker show the container side.
//
// source (data):            https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_friedman1.html
//                           (Friedman 1991, https://doi.org/10.1214/aos/1176347963)
// source (OptunaSearchCV):  https://optuna-integration.readthedocs.io/en/stable/reference/generated/optuna_integration.OptunaSearchCV.html
// source (FastTree):        https://learn.microsoft.com/en-us/dotnet/api/microsoft.ml.treeextensions.fasttree
//
//   Run:  dotnet run --project examples/projects/MLNetModelTuning           (20 trials)
//         dotnet run --project examples/projects/MLNetModelTuning -- 40     (40 more, in the same study)

using System.Globalization;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using MLNetModelTuning;
using OptunaSharp;
using OptunaSharp.Distributions;
using OptunaSharp.Importance;
using OptunaSharp.Integration;
using OptunaSharp.Samplers;
using OptunaSharp.Storages;
using OptunaSharp.Visualization.Plotly;
using Plotly.NET.CSharp;   // SaveHtml

int nTrials = args.Length > 0 ? int.Parse(args[0], CultureInfo.InvariantCulture) : 20;

// everything a run writes goes to ./out; under `dotnet run --project` that is the project's folder, and the
// project's .gitignore excludes it
Directory.CreateDirectory("out");

// ---- data: the rows the search tunes on, and a test set it never sees
var ml = new MLContext(seed: 0);   // seeds ML.NET's fold split and its trainers
FriedmanRow[] rows = FriedmanData.Generate(samples: 1_000, noise: 1.0, seed: 0);
DataOperationsCatalog.TrainTestData split =
    ml.Data.TrainTestSplit(ml.Data.LoadFromEnumerable(rows), testFraction: 0.2, seed: 0);
Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
    $"Friedman #1: {rows.Length} rows, mean label {rows.Average(r => r.Label):F4}; 80% tune the model, 20% are held out."));

// ---- the study: in SQLite, so the next run continues this one's search (Python Optuna can open the file too)
using var storage = new RDBStorage("sqlite:///out/mlnet-tuning.db");
Study study = Optuna.CreateStudy(
    storage: storage,
    sampler: new TPESampler(seed: 0),
    studyName: "fasttree-friedman1",
    direction: StudyDirection.Maximize,   // the scorer is R², so higher is better
    loadIfExists: true);
Console.WriteLine($"Study '{study.StudyName}' in out/mlnet-tuning.db has {study.Trials.Count} trials; running {nTrials} more.");

// ---- the search: four FastTree hyperparameters, each candidate cross-validated on 5 folds of the tuning rows
var space = new Dictionary<string, BaseDistribution>
{
    ["number_of_leaves"] = new IntDistribution(4, 64, log: true),
    ["number_of_trees"] = new IntDistribution(20, 200, log: true),
    ["learning_rate"] = new FloatDistribution(0.01, 0.4, log: true),
    ["minimum_example_count_per_leaf"] = new IntDistribution(2, 40, log: true),
};
var estimator = new MLNetEstimator(ml, split.TrainSet, BuildPipeline, RSquared, folds: 5, seed: 0);

// nJobs stays at its default of 1: MLNetEstimator shares one MLContext between all trials, and concurrent
// training through one MLContext is not something docs/THREADING.md vouches for
var search = new OptunaSearchCV(estimator, space, nTrials: nTrials, study: study, refit: true);
search.Fit();   // OptunaSharp logs each trial on stderr as it finishes

Console.WriteLine();
Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
    $"Best mean cross-validation R²: {search.BestScore:F4}, trial {search.BestTrial.Number} of {study.Trials.Count}:"));
foreach ((string name, object? value) in search.BestParams)
    Console.WriteLine($"    {name,-32} {Format(value)}");

// refit: true trained the best configuration on all tuning rows; this is its score on rows it never saw
Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
    $"R² of the refit model on the held-out rows: {estimator.Score(split.TestSet):F4}"));

// ---- which hyperparameters mattered: fANOVA over the study's complete trials (it needs at least two)
var fanova = new FanovaImportanceEvaluator(seed: 0);
if (study.GetTrials(deepcopy: false, states: [TrialState.Complete]).Count >= 2)
{
    Console.WriteLine("Hyperparameter importance (fANOVA):");
    foreach ((string name, double importance) in ImportanceFunctions.GetParamImportances(study, fanova))
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture, $"    {name,-32} {importance:F3}"));
    PlotlyVisualization.PlotParamImportances(study, fanova).SaveHtml("out/importance.html");
}

// ---- artifacts: the refit model, loadable with MLContext.Model.Load, and the optimization history
ml.Model.Save(estimator.BestModel!, split.TrainSet.Schema, "out/best-model.zip");
PlotlyVisualization.PlotOptimizationHistory(study).SaveHtml("out/history.html");
Console.WriteLine($"Saved best-model.zip, history.html and importance.html in {Path.GetFullPath("out")}.");

// builds a fresh FastTree pipeline for one candidate; MLNetEstimator calls it once per fold and once for the refit
IEstimator<ITransformer> BuildPipeline(IReadOnlyDictionary<string, object?> parameters)
    => ml.Regression.Trainers.FastTree(new FastTreeRegressionTrainer.Options
    {
        LabelColumnName = nameof(FriedmanRow.Label),
        FeatureColumnName = nameof(FriedmanRow.Features),
        // integer parameters arrive as boxed long, ML.NET wants int
        NumberOfLeaves = Convert.ToInt32(parameters["number_of_leaves"], CultureInfo.InvariantCulture),
        NumberOfTrees = Convert.ToInt32(parameters["number_of_trees"], CultureInfo.InvariantCulture),
        MinimumExampleCountPerLeaf = Convert.ToInt32(parameters["minimum_example_count_per_leaf"], CultureInfo.InvariantCulture),
        LearningRate = Convert.ToDouble(parameters["learning_rate"], CultureInfo.InvariantCulture),
        // one thread: on folds of 640 rows, coordinating many threads costs more than it saves, and a
        // single-threaded fit is deterministic, so a candidate scores the same every time it is trained
        NumberOfThreads = 1,
    });

// the score of one fitted fold model on its validation fold; OptunaSearchCV maximizes the mean over the folds
static double RSquared(MLContext context, ITransformer model, IDataView validation)
    => context.Regression.Evaluate(model.Transform(validation)).RSquared;

// a parameter value as text, with doubles shortened to six significant digits
static string Format(object? value)
    => value is double d
        ? d.ToString("G6", CultureInfo.InvariantCulture)
        : Convert.ToString(value, CultureInfo.InvariantCulture) ?? "None";
