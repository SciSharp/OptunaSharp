using OptunaSharp;

namespace TuningCommon;

/// <summary>
///     The objective every worker in the tuning examples evaluates: a deterministic stand-in for training a
///     small neural network. It suggests five hyperparameters, "trains" for <see cref="Epochs" /> epochs and
///     reports the validation loss after each epoch, so the study's pruner can stop a hopeless run early.
/// </summary>
/// <remarks>
///     <para>
///         <b>A surrogate, not real training.</b> The loss curve is a closed-form function of the
///         hyperparameters (<see cref="ValidationLoss" />), with an optional sleep per epoch so that trials
///         take time and parallel workers visibly overlap. A sampler's trajectory depends only on its seed, the
///         search space and the values it is told, so the surrogate drives the real OptunaSharp machinery
///         (sampling, pruning, storage, parallel and remote workers) without a machine-learning runtime. The
///         landscape has one best region: Adam, a learning rate near 1e-3, dropout near 0.2, batch size 64 and
///         weight decay near 1e-4.
///     </para>
///     <para>
///         <b>Keep the copies identical.</b> <c>TuningWorker/python_worker.py</c> re-implements this function
///         so that a Python Optuna worker can join the same study. The parameter names, the distributions and
///         the formula must match, or the samplers see two different problems under one study name, and a
///         parameter suggested with a different distribution raises Optuna's
///         <c>Cannot set different distribution kind to the same parameter name.</c>
///     </para>
/// </remarks>
public static class SimulatedTraining
{
    /// <summary>The number of epochs one trial trains for. It is also the last step a trial reports.</summary>
    public const int Epochs = 20;

    /// <summary>
    ///     The user attribute that records which worker evaluated a trial. The service's dashboard and the
    ///     trials API show it, so a study shared by several processes shows who ran each trial.
    /// </summary>
    public const string WorkerAttribute = "worker";

    /// <summary>The optimizer choices, in the order the Python worker lists them.</summary>
    private static readonly string[] Optimizers = ["sgd", "momentum", "adam"];

    /// <summary>
    ///     The batch-size choices. They are <see cref="int" /> on purpose: Python suggests the same list of
    ///     ints, and both sides then store an identical categorical distribution.
    /// </summary>
    private static readonly int[] BatchSizes = [16, 32, 64, 128];

    /// <summary>Builds the objective that <c>Study.Optimize</c> calls once per trial.</summary>
    /// <param name="workerName">
    ///     The name recorded on every trial as the <see cref="WorkerAttribute" /> user attribute, for example
    ///     <c>worker:host:1234</c>.
    /// </param>
    /// <param name="epochDelay">How long each simulated epoch sleeps; <see cref="TimeSpan.Zero" /> runs a trial instantly.</param>
    /// <returns>
    ///     The objective. It captures only immutable state, so it is safe to run on several threads at once
    ///     under <c>Optimize(nJobs: n)</c>.
    /// </returns>
    /// <exception cref="ArgumentNullException"><paramref name="workerName" /> is <c>null</c>.</exception>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="epochDelay" /> is negative.</exception>
    public static Func<Trial, double> CreateObjective(string workerName, TimeSpan epochDelay)
    {
        ArgumentNullException.ThrowIfNull(workerName);
        ArgumentOutOfRangeException.ThrowIfLessThan(epochDelay, TimeSpan.Zero);
        return trial => Evaluate(trial, workerName, epochDelay);
    }

    /// <summary>
    ///     The validation loss after <paramref name="epoch" /> epochs: an exponential decay from 1.0 toward a
    ///     floor. A learning rate far from the optimizer's sweet spot both slows the decay and raises the floor;
    ///     dropout, batch size and weight decay away from their best values raise the floor.
    /// </summary>
    /// <param name="epoch">The epoch just finished, from 1 to <see cref="Epochs" />.</param>
    /// <param name="learningRate">The learning rate, between 1e-5 and 1e-1.</param>
    /// <param name="optimizer"><c>sgd</c>, <c>momentum</c> or <c>adam</c>.</param>
    /// <param name="batchSize">The batch size: 16, 32, 64 or 128.</param>
    /// <param name="dropout">The dropout rate, between 0 and 0.5.</param>
    /// <param name="weightDecay">The weight decay, between 1e-6 and 1e-2.</param>
    /// <returns>The loss, which lies between the floor (at least 0.05) and 1.0.</returns>
    /// <exception cref="ArgumentOutOfRangeException"><paramref name="optimizer" /> is not one of the three choices.</exception>
    public static double ValidationLoss(
        int epoch, double learningRate, string optimizer, long batchSize, double dropout, double weightDecay)
    {
        // each optimizer has its own best learning rate and its own top speed
        (double bestLearningRate, double topSpeed) = optimizer switch
        {
            "sgd" => (3e-2, 0.12),
            "momentum" => (1e-2, 0.18),
            "adam" => (1e-3, 0.25),
            _ => throw new ArgumentOutOfRangeException(nameof(optimizer), optimizer, "Expected sgd, momentum or adam."),
        };

        // how many decades the learning rate is from the sweet spot; squared, so too high and too low both hurt.
        // Math.Pow(x, 2.0) rather than x * x keeps the C# value bit-identical to the Python copy's x ** 2.0
        double learningRatePenalty = Math.Pow(Math.Log10(learningRate / bestLearningRate), 2.0);
        double speed = topSpeed / (1.0 + learningRatePenalty);
        double floor = 0.05
                       + 0.6 * Math.Pow(dropout - 0.2, 2.0)
                       + 0.015 * Math.Abs(Math.Log2(batchSize / 64.0))
                       + 0.04 * learningRatePenalty
                       + 0.005 * Math.Pow(Math.Log10(weightDecay) + 4.0, 2.0);
        return floor + (1.0 - floor) * Math.Exp(-speed * epoch);
    }

    /// <summary>
    ///     Runs one trial: records the worker, suggests the five hyperparameters, then reports each epoch's loss
    ///     and stops as soon as the study's pruner says so.
    /// </summary>
    /// <param name="trial">The trial to evaluate.</param>
    /// <param name="workerName">The value of the <see cref="WorkerAttribute" /> user attribute.</param>
    /// <param name="epochDelay">How long each simulated epoch sleeps.</param>
    /// <returns>The validation loss after the last epoch, which the study minimizes.</returns>
    /// <exception cref="TrialPruned">
    ///     The study's pruner stopped the trial. <c>Optimize</c> catches it and stores the trial as
    ///     <see cref="TrialState.Pruned" />, as Optuna does with <c>optuna.TrialPruned</c>.
    /// </exception>
    private static double Evaluate(Trial trial, string workerName, TimeSpan epochDelay)
    {
        trial.SetUserAttr(WorkerAttribute, workerName);

        double learningRate = trial.SuggestFloat("learning_rate", 1e-5, 1e-1, log: true);   // log scale: spans four decades
        string optimizer = trial.SuggestCategorical("optimizer", Optimizers);
        long batchSize = trial.SuggestCategorical("batch_size", BatchSizes);                // int choices come back as long
        double dropout = trial.SuggestFloat("dropout", 0.0, 0.5);
        double weightDecay = trial.SuggestFloat("weight_decay", 1e-6, 1e-2, log: true);

        double loss = double.NaN;
        for (int epoch = 1; epoch <= Epochs; epoch++)
        {
            // objectives run on OptunaSharp's own worker threads (or the caller's thread at nJobs: 1), never on
            // the .NET thread pool, so a blocking sleep here starves nothing
            if (epochDelay > TimeSpan.Zero)
                Thread.Sleep(epochDelay);

            loss = ValidationLoss(epoch, learningRate, optimizer, batchSize, dropout, weightDecay);
            trial.Report(loss, epoch);   // the step is the epoch, so the pruner compares trials at equal progress
            if (trial.ShouldPrune())
                throw new TrialPruned($"Trial was pruned at epoch {epoch}.");
        }

        return loss;
    }
}
