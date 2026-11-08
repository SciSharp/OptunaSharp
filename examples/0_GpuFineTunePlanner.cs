#:project ../src/OptunaSharp.Core/OptunaSharp.Core.csproj
#:property PublishAot=false

// GPU fine-tuning planner — pick the micro-batch, sequence packing, precision, activation checkpointing,
// optimizer and LoRA rank that give the highest training throughput for a 7B-parameter model on one 24 GB
// GPU. Configurations that do not fit throw an out-of-memory error, fp16 runs with a hot learning rate
// diverge to NaN — and the study keeps going: `catch=` turns the OOMs into FAIL trials, NaN values are
// recorded as FAIL with a warning, two workers share a wall-clock `timeout`, and MaxTrialsCallback stops
// the run after 60 successful trials. FixedTrial replays the winner to print its memory breakdown.
//
// source: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize
//         (`catch`, `timeout`, `n_jobs`, `callbacks`)
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.MaxTrialsCallback.html
//         https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.FixedTrial.html
//         https://huggingface.co/docs/transformers/perf_train_gpu_one (the memory/throughput levers modelled)
//
// Surrogate model with real accounting: memory = frozen base weights (bytes per precision) + LoRA weights,
// gradients and optimizer states (AdamW 8 B/param, Adafactor ≈ 4, SGD 0) + activations (∝ micro-batch ×
// sequence × hidden × layers, ÷ √layers with checkpointing) + a fixed CUDA/workspace reserve. Throughput is
// a roofline-style tokens/s: matmul FLOPs at the precision's peak × utilization that rises with the
// micro-batch, minus the ~33 % recompute cost of checkpointing. The evaluation also sleeps 15 ms so the
// two workers genuinely overlap. Results depend on thread timing, so trial counts vary slightly run to run.
//
// The lesson in the output: most trials FAIL with an OOM, because a sampler learns nothing from a FAIL
// trial (Optuna's samplers ignore them), so TPE keeps proposing configurations that do not fit. When the
// failure is predictable like this, compute the memory first and record it with `SetConstraint` instead of
// throwing (see 0_CloudRightSizing.cs) — then the sampler learns where the wall is.
//
//   Run:  dotnet run examples/0_GpuFineTunePlanner.cs

using System.Globalization;
using OptunaSharp;
using OptunaSharp.Collections;
using OptunaSharp.Logging;
using OptunaSharp.Samplers;

OptunaLogging.SetVerbosity(LogLevel.Error);     // the NaN-trial warnings are expected; keep the output tidy

var study = Optuna.CreateStudy(sampler: new TPESampler(seed: 10, nStartupTrials: 20), direction: StudyDirection.Maximize, studyName: "7b-on-24gb");
study.Optimize(
    t => Planner.Objective(t),
    nTrials: 500,                                                // an upper bound; the callback ends it sooner
    timeout: 60,                                                 // seconds of wall clock, whichever comes first
    nJobs: 2,
    @catch: new[] { typeof(InsufficientMemoryException) },      // an OOM is a FAILED trial, not a crash
    callbacks: new Action<Study, FrozenTrial>[] { new MaxTrialsCallback(60, new[] { TrialState.Complete }) });

var states = study.Trials.GroupBy(t => t.State).ToDictionary(g => g.Key, g => g.Count());
var failReasons = study.Trials.Where(t => t.State == TrialState.Fail)
    .GroupBy(t => (string?)t.UserAttrs.GetValueOrDefault("failure") ?? "unknown")   // tagged by the objective
    .ToDictionary(g => g.Key, g => g.Count());
Console.WriteLine($"trials: {study.Trials.Count}  complete: {states.GetValueOrDefault(TrialState.Complete)}  failed: {states.GetValueOrDefault(TrialState.Fail)}"
                  + $"  ({string.Join(", ", failReasons.Select(kv => $"{kv.Value} {kv.Key}"))})");
var best = study.BestTrial;
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "best: {0:N0} tokens/s (trial #{1})", best.Value, best.Number));
foreach (var kv in best.Params)
    Console.WriteLine($"    {kv.Key,-16} {kv.Value}");

// replay the winner outside the study: FixedTrial answers every suggest_* with the stored value.
var replay = new FixedTrial(new OrderedDict<string, object?>(best.Params));
double again = Planner.Objective(replay);
Console.WriteLine(string.Format(CultureInfo.InvariantCulture, "FixedTrial replay: {0:N0} tokens/s; memory {1}", again, replay.UserAttrs["memory_breakdown"]));

/// <summary>The memory and throughput model of LoRA fine-tuning a 7B model on one GPU.</summary>
static class Planner
{
    /// <summary>Evaluates one configuration.</summary>
    /// <param name="t">A study trial or a <see cref="FixedTrial" /> replaying stored parameters.</param>
    /// <returns>Training throughput in tokens per second, or NaN when fp16 training diverges.</returns>
    /// <exception cref="InsufficientMemoryException">The configuration needs more than the GPU's 24 GB.</exception>
    public static double Objective(BaseTrial t)
    {
        long micro = t.SuggestInt("micro_batch", 1, 64, log: true);
        long seq = t.SuggestCategorical("seq_len", new[] { 512, 1024, 2048, 4096 });
        string precision = t.SuggestCategorical("precision", new[] { "fp32", "bf16", "fp16" });
        bool checkpointing = t.SuggestCategorical("grad_checkpoint", new[] { false, true });
        string optimizer = t.SuggestCategorical("optimizer", new[] { "adamw", "adafactor", "sgd" });
        long rank = t.SuggestInt("lora_rank", 4, 256, log: true);
        double lr = t.SuggestFloat("lr", 1e-5, 1e-2, log: true);
        Thread.Sleep(15);                                        // the "profiling run"

        const double Params = 7e9, Hidden = 4096, Layers = 32, GpuBytes = 24e9;
        double bytesPer = precision == "fp32" ? 4 : 2;
        double loraParams = 2 * rank * Hidden * Layers * 4;      // A and B for q, k, v, o in every layer
        double weights = Params * bytesPer + loraParams * 4;     // frozen base + fp32 master LoRA weights
        double optState = loraParams * (optimizer == "adamw" ? 8 : optimizer == "adafactor" ? 4 : 0) + loraParams * 4;
        double activations = micro * seq * Hidden * Layers * 18 * bytesPer / (checkpointing ? Math.Sqrt(Layers) : 1);
        double reserve = 1.8e9;
        double total = weights + optState + activations + reserve;
        t.SetUserAttr("memory_breakdown", string.Format(CultureInfo.InvariantCulture,
            "{0:F1} GB = weights {1:F1} + optimizer {2:F2} + activations {3:F1} + reserve {4:F1}",
            total / 1e9, weights / 1e9, optState / 1e9, activations / 1e9, reserve / 1e9));
        if (total > GpuBytes)
        {
            t.SetUserAttr("failure", "out of memory");          // Optuna keeps no failure reason: tag it
            throw new InsufficientMemoryException($"CUDA OutOfMemory: needs {total / 1e9:F1} GB of 24 GB");
        }

        // fp16 without loss scaling headroom diverges at high learning rates (bf16 has fp32's exponent range).
        if (precision == "fp16" && lr > 2e-3)
        {
            t.SetUserAttr("failure", "diverged (NaN)");
            return double.NaN;                                   // Optuna records a NaN value as FAIL
        }

        double peakTflops = precision == "fp32" ? 40 : 165;     // dense matmul peak of the card
        double utilization = 0.55 * (1 - Math.Exp(-micro * seq / 4096.0));   // small batches starve the GPU
        double flopsPerToken = 2 * Params * 3 * (checkpointing ? 4.0 / 3.0 : 1.0);   // fwd + bwd (+ recompute)
        double tokensPerSecond = peakTflops * 1e12 * utilization / flopsPerToken;
        // too-low learning rates waste the run: count only the tokens that make progress.
        return tokensPerSecond * Math.Min(1.0, lr / 2e-4);
    }
}
