using System.Collections.Generic;
using OptunaSharp;

namespace OptunaSharp.Benchmark.VolleyEval;

/// <summary>
///     The VolleyEval problem interface — a one-to-one mirror of COCO's <c>coco_problem_t</c>
///     (Hansen et al., <i>COCO: The Experimental Procedure</i> §2.1), reduced to the members the v0
///     three-radical suite needs. A <b>problem</b> in COCO is the triple
///     <c>(dimension, function, instance)</c> plus a <b>target</b>; here a challenge fixes the function,
///     <see cref="InstanceId" /> selects the deterministic (seeded) shift/rotation, and
///     <see cref="Optimum" />/<see cref="RandomBaseline" /> supply the normalization band.
///
///     <para>
///         Objective evaluation is define-by-run: <see cref="Evaluate" /> pulls its parameters from a
///         live <see cref="Trial" /> (COCO's <c>evaluate_function</c> / the plan's
///         <c>BuildObjective(Trial)</c>). It always returns a vector; a single-objective challenge
///         returns a length-1 list so the <see cref="Evaluator" /> can stay generic across the
///         single/multi split.
///     </para>
///
///     <para>
///         Scoring lives on the challenge (<see cref="Score" />) because only the challenge knows its
///         own optimum, random baseline, and — for the multi-objective radical — its hypervolume
///         reference point and true-front ceiling. This keeps the <see cref="Evaluator" /> a pure
///         run/aggregate loop.
///     </para>
/// </summary>
public interface IChallenge
{
    /// <summary>Short stable identifier used as the profile-matrix column header (e.g. <c>"rosenbrock"</c>).</summary>
    string Name { get; }

    /// <summary>The "radical" this challenge stresses and any secondary axes, for reporting (e.g. <c>["continuous","coupled","single-obj"]</c>).</summary>
    IReadOnlyList<string> Tags { get; }

    /// <summary>COCO <c>get_dimension</c> — the number of decision variables.</summary>
    int Dimension { get; }

    /// <summary>COCO <c>get_number_of_objectives</c>. 1 = single-objective, ≥2 = multi-objective.</summary>
    int NumObjectives { get; }

    /// <summary>Optimization direction per objective; drives <see cref="Optuna.CreateStudy" /> and the single/multi dispatch.</summary>
    IReadOnlyList<StudyDirection> Directions { get; }

    /// <summary>
    ///     COCO's instance index. A deterministic transform (seeded shift/rotation/table draw) applied
    ///     to the base function so the optimum stays known while axis-aligned exploitation is defeated
    ///     and optimum-memorization is prevented (Best-Practice §1, "Generalization").
    /// </summary>
    int InstanceId { get; }

    /// <summary>
    ///     The exact optimal objective value the normalization treats as score = 1. For single-objective
    ///     challenges this is the scalar optimum (BBOB minimum 0; NK's DP maximum, sign-flipped). For the
    ///     multi-objective challenge this carries the analytic true-front <b>hypervolume</b> ceiling.
    /// </summary>
    double Optimum { get; }

    /// <summary>
    ///     The score = 0 reference: the expected objective under uniform sampling
    ///     (COCO/Best-Practice normalization <c>(best − random)/(optimum − random)</c>). For the
    ///     multi-objective challenge the score is a hypervolume ratio and this member is unused (0).
    /// </summary>
    double RandomBaseline { get; }

    /// <summary>The hypervolume reference (nadir) point for multi-objective scoring; empty for single-objective.</summary>
    IReadOnlyList<double> ReferencePoint { get; }

    /// <summary>The default per-run trial budget for this challenge (COCO fixed-budget).</summary>
    int Budget { get; }

    /// <summary>
    ///     True only for pure-categorical single-objective challenges, where the "inert" check
    ///     (trajectory ≡ RandomSampler fallback) is meaningful. Skipping it elsewhere avoids paying for
    ///     a comparison that can never fire.
    /// </summary>
    bool DetectInert { get; }

    /// <summary>True when <see cref="NumObjectives" /> &gt; 1.</summary>
    bool IsMultiObjective => NumObjectives > 1;

    /// <summary>
    ///     The define-by-run objective (COCO <c>evaluate_function</c> / plan <c>BuildObjective</c>).
    ///     Suggests this challenge's parameters from <paramref name="trial" /> and returns the objective
    ///     vector (length <see cref="NumObjectives" />).
    /// </summary>
    IReadOnlyList<double> Evaluate(Trial trial);

    /// <summary>
    ///     Normalized quality in roughly [0, 1] extracted from a <b>completed</b> study: 1 = optimum,
    ///     0 = random baseline. Single-objective challenges use
    ///     <c>(baseline − best)/(baseline − optimum)</c>; the multi-objective challenge uses
    ///     <c>hypervolume(study) / trueFrontHypervolume</c>. Encapsulated per-challenge because only the
    ///     challenge owns its optimum/baseline/reference-point.
    /// </summary>
    double Score(Study study);
}
