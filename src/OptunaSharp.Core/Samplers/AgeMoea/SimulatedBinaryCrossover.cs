using System;
using System.Collections.Generic;
using NumSharp;

namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Simulated Binary Crossover, ported draw-for-draw from pymoo <c>@0.6.1.3</c> —
///     <c>pymoo.operators.crossover.sbx.SBX</c> plus its <c>cross_sbx</c> function and the RNG-gating
///     in <c>pymoo.core.crossover.Crossover.do</c>. This is <b>not</b> Optuna's SBX
///     (<c>OptunaSharp.Samplers.Nsgaii.SBXCrossover</c>): pymoo draws whole
///     <c>(n_matings, n_var)</c> matrices and a single shared <c>rand</c> per betaq, in a different
///     order than Optuna's per-gene collapse — so byte-exactness with pymoo requires pymoo's code.
///
///     AGE-MOEA-2 constructs <c>SBX(prob=0.9, eta=15)</c>: <c>prob</c> is the base
///     <see cref="_prob" /> (the per-mating crossover gate), while <c>prob_var</c> stays 0.5.
///
///     Draw order per call (all from the same MT19937 instance):
///     <list type="number">
///       <item><c>random(n_matings)</c> — the crossover gate (<c>Crossover.do</c>).</item>
///       <item><c>random((n_matings, 1))</c> — <c>SBX._do</c> prob_bin/prob_exch (consumed even though
///             <c>prob_exch = 1.0</c> makes it a no-op).</item>
///       <item><c>random((n_matings, n_var))</c> — the per-variable crossover mask.</item>
///       <item><c>random(#cross)</c> — the betaq randoms (shared by both children).</item>
///       <item><c>random(#cross)</c> — the c1/c2 swap mask.</item>
///     </list>
/// </summary>
internal sealed class SimulatedBinaryCrossover
{
    private readonly double _probVar;
    private readonly double _eta;
    private readonly double _probExch;
    private readonly double _probBin;
    private readonly double _prob;
    private const double Eps = 1.0e-14;

    /// <summary>Number of parent individuals required by one crossover.</summary>
    public int NParents => 2;
    /// <summary>Number of children produced per mating.</summary>
    public int NOffsprings => 2;

    /// <summary>Configures mating-level and coordinate-level probabilities for two-child simulated-binary crossover.</summary>
    /// <param name="prob">Probability of applying the operator to an individual or mating.</param>
    /// <param name="eta">Distribution index controlling the concentration of crossover or mutation around the parents.</param>
    /// <param name="probVar">Probability of varying each coordinate.</param>
    /// <param name="probExch">Probability of enabling child-coordinate exchange for a mating.</param>
    /// <param name="probBin">Probability of exchanging a child coordinate when exchange is enabled.</param>
    public SimulatedBinaryCrossover(
        double prob = 0.9, double eta = 15, double probVar = 0.5, double probExch = 1.0, double probBin = 0.5)
    {
        _prob = prob;
        _eta = eta;
        _probVar = probVar;
        _probExch = probExch;
        _probBin = probBin;
    }

    /// <summary>
    ///     <c>Crossover.do</c>: build the parent tensor from mating indices, draw the crossover gate,
    ///     run <see cref="CrossSbx" /> on all matings, and take the crossed columns from the result and
    ///     the untouched columns from the parents. Returns the <c>2 * n_matings</c> offspring flattened
    ///     C-order: all first-children, then all second-children.
    /// </summary>
    /// <param name="popX">Decision vectors in population order.</param>
    /// <param name="parents">Selected parent individuals in crossover order.</param>
    /// <param name="problem">Decision bounds and objective/constraint evaluation surface.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>Offspring rows grouped in the reference child-major order.</returns>
    public double[][] Do(double[][] popX, int[][] parents, AgeMoeaProblem problem, NumPyRandom rng)
    {
        int nMatings = parents.Length;
        int nVar = problem.NVar;

        // X[parent][mating][var] — pymoo's (n_parents=2, n_matings, n_var) tensor.
        var xp0 = new double[nMatings][];
        var xp1 = new double[nMatings][];
        for (int m = 0; m < nMatings; m++)
        {
            xp0[m] = (double[])popX[parents[m][0]].Clone();
            xp1[m] = (double[])popX[parents[m][1]].Clone();
        }

        // the probability of executing the crossover (Crossover.do): random(n_matings) < prob.
        double[] gate = NpRandom.Random(rng, nMatings);
        var cross = new bool[nMatings];
        bool any = false;
        for (int m = 0; m < nMatings; m++)
        {
            cross[m] = gate[m] < _prob;
            any |= cross[m];
        }

        // offspring buffers (default = copy of parents; SBX result overwrites crossed matings).
        var q0 = new double[nMatings][];
        var q1 = new double[nMatings][];
        for (int m = 0; m < nMatings; m++)
        {
            q0[m] = (double[])xp0[m].Clone();
            q1[m] = (double[])xp1[m].Clone();
        }

        if (any)
        {
            (double[][] c0, double[][] c1) = InnerDo(xp0, xp1, problem, rng);
            for (int m = 0; m < nMatings; m++)
                if (cross[m])
                {
                    q0[m] = c0[m];
                    q1[m] = c1[m];
                }
        }

        // for k where NOT cross: n_offsprings == n_parents == 2 -> s = arange(2), parents kept as-is
        // (no RNG draw). Nothing more to do; q0/q1 already hold the parent copies.

        // flatten C-order: reshape((2, n_matings, n_var)) -> (2*n_matings, n_var) = all q0, then all q1.
        var result = new double[2 * nMatings][];
        for (int m = 0; m < nMatings; m++) result[m] = q0[m];
        for (int m = 0; m < nMatings; m++) result[nMatings + m] = q1[m];
        return result;
    }

    /// <summary>
    ///     <c>SBX._do</c>: draw the (consumed-but-inert) prob_bin/prob_exch mask, then run
    ///     <see cref="CrossSbx" />. With <c>prob_exch = 1.0</c> the drawn values never zero prob_bin,
    ///     but the <c>random((n_matings, 1))</c> draw is still consumed (stream position matters).
    /// </summary>
    /// <param name="x0">First parent coordinate batch.</param>
    /// <param name="x1">Second parent coordinate batch.</param>
    /// <param name="problem">Decision bounds and objective/constraint evaluation surface.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>The two offspring coordinate batches before mating-level child selection.</returns>
    private (double[][] C0, double[][] C1) InnerDo(double[][] x0, double[][] x1, AgeMoeaProblem problem, NumPyRandom rng)
    {
        int nMatings = x0.Length;

        // rand = random((n_matings, 1)); prob_bin[rand > prob_exch] = 0.0
        double[][] rand2d = NpRandom.Random(rng, nMatings, 1);
        var probBin = new double[nMatings];
        for (int m = 0; m < nMatings; m++)
            probBin[m] = rand2d[m][0] > _probExch ? 0.0 : _probBin;

        return CrossSbx(x0, x1, problem.Xl, problem.Xu, probBin, rng);
    }

    /// <summary><c>cross_sbx</c> — the vectorized SBX kernel over the flattened crossed positions.</summary>
    /// <param name="x0">First parent coordinate batch.</param>
    /// <param name="x1">Second parent coordinate batch.</param>
    /// <param name="xl">Lower decision bounds in coordinate order.</param>
    /// <param name="xu">Upper decision bounds in coordinate order.</param>
    /// <param name="probBinPerMating">Per-mating exchange probabilities.</param>
    /// <param name="rng">Random stream whose draw order is shared with the reference algorithm.</param>
    /// <returns>The two numerical offspring batches in corresponding mating order.</returns>
    private (double[][] C0, double[][] C1) CrossSbx(
        double[][] x0, double[][] x1, double[] xl, double[] xu, double[] probBinPerMating, NumPyRandom rng)
    {
        int nMatings = x0.Length;
        int nVar = xl.Length;

        // cross = random((n_matings, n_var)) < prob_var
        double[][] crossR = NpRandom.Random(rng, nMatings, nVar);
        var cross = new bool[nMatings][];
        for (int m = 0; m < nMatings; m++)
        {
            cross[m] = new bool[nVar];
            for (int j = 0; j < nVar; j++)
            {
                bool c = crossR[m][j] < _probVar;
                // too_close: |X0 - X1| <= eps  -> disable
                if (Math.Abs(x0[m][j] - x1[m][j]) <= Eps) c = false;
                // disable when the bound collapses
                if (xl[j] == xu[j]) c = false;
                cross[m][j] = c;
            }
        }

        // collect selected positions in C-order (mating-major, then variable).
        var sel = new List<(int M, int J)>();
        for (int m = 0; m < nMatings; m++)
            for (int j = 0; j < nVar; j++)
                if (cross[m][j]) sel.Add((m, j));
        int count = sel.Count;

        // y1 = min(X, axis=0)[cross], y2 = max(X, axis=0)[cross]
        var y1 = new double[count];
        var y2 = new double[count];
        var xlSel = new double[count];
        var xuSel = new double[count];
        var probBinSel = new double[count];
        for (int k = 0; k < count; k++)
        {
            (int m, int j) = sel[k];
            double a = x0[m][j], b = x1[m][j];
            y1[k] = Math.Min(a, b);
            y2[k] = Math.Max(a, b);
            xlSel[k] = xl[j];
            xuSel[k] = xu[j];
            probBinSel[k] = probBinPerMating[m];
        }

        // rand = random(len(eta)) where len(eta) == count
        double[] rand = NpRandom.Random(rng, count);

        double etaP1 = _eta + 1.0;
        double invEtaP1 = 1.0 / etaP1;

        double CalcBetaq(double beta, double r)
        {
            double alpha = 2.0 - Math.Pow(beta, -etaP1);
            return r <= 1.0 / alpha
                ? Math.Pow(r * alpha, invEtaP1)
                : Math.Pow(1.0 / (2.0 - r * alpha), invEtaP1);
        }

        var c1 = new double[count];
        var c2 = new double[count];
        for (int k = 0; k < count; k++)
        {
            double delta = y2[k] - y1[k];
            double beta1 = 1.0 + 2.0 * (y1[k] - xlSel[k]) / delta;
            double betaq1 = CalcBetaq(beta1, rand[k]);
            c1[k] = 0.5 * ((y1[k] + y2[k]) - betaq1 * delta);

            double beta2 = 1.0 + 2.0 * (xuSel[k] - y2[k]) / delta;
            double betaq2 = CalcBetaq(beta2, rand[k]);
            c2[k] = 0.5 * ((y1[k] + y2[k]) + betaq2 * delta);
        }

        // b = random(len(prob_bin)) < prob_bin  -> swap c1[b] <-> c2[b]
        double[] bDraw = NpRandom.Random(rng, count);
        for (int k = 0; k < count; k++)
            if (bDraw[k] < probBinSel[k])
            {
                (c1[k], c2[k]) = (c2[k], c1[k]);
            }

        // Q = copy(X); Q[0, cross] = c1; Q[1, cross] = c2; then clamp to [xl, xu].
        var q0 = new double[nMatings][];
        var q1 = new double[nMatings][];
        for (int m = 0; m < nMatings; m++)
        {
            q0[m] = (double[])x0[m].Clone();
            q1[m] = (double[])x1[m].Clone();
        }
        for (int k = 0; k < count; k++)
        {
            (int m, int j) = sel[k];
            q0[m][j] = c1[k];
            q1[m][j] = c2[k];
        }
        RepairClamp(q0, xl, xu);
        RepairClamp(q1, xl, xu);
        return (q0, q1);
    }

    /// <summary><c>repair_clamp</c>: clamp each element into <c>[xl, xu]</c>.</summary>
    /// <param name="x">Input numerical coordinates or observation rows.</param>
    /// <param name="xl">Lower decision bounds in coordinate order.</param>
    /// <param name="xu">Upper decision bounds in coordinate order.</param>
    private static void RepairClamp(double[][] x, double[] xl, double[] xu)
    {
        for (int i = 0; i < x.Length; i++)
            for (int j = 0; j < xl.Length; j++)
            {
                if (x[i][j] < xl[j]) x[i][j] = xl[j];
                if (x[i][j] > xu[j]) x[i][j] = xu[j];
            }
    }
}
