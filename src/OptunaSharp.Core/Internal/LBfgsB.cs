using System;

namespace OptunaSharp.Internal;

/// <summary>
///     L-BFGS-B (limited-memory BFGS with box constraints) — a faithful managed port of scipy's
///     <c>scipy/optimize/__lbfgsb.c</c> (v1.16.3), itself the C translation of the Fortran L-BFGS-B
///     v3.0 by Zhu, Byrd, Lu, Nocedal &amp; Morales. This is the optimizer Optuna's GP kernel-parameter
///     fit calls through <c>scipy.optimize.minimize(method="l-bfgs-b")</c> (PORTING_PLAN Phase 3.2);
///     porting it keeps the GP stack in Tier N — no scipy, no TorchSharp.
///
///     <para><b>Structure preserved for diffing against the C.</b> The compact-BFGS matrices stay in
///     the same column-major flat layout (<c>a[i + lda*j]</c>) and 0-based indexing as the C source;
///     the reverse-communication driver (<see cref="Mainlb" />) keeps the original <c>goto</c>
///     control flow and the <c>isave</c>/<c>dsave</c>/<c>lsave</c> persistent-state arrays; C pointer
///     arithmetic (<c>&amp;wa[2*m]</c>) becomes <see cref="Span{T}" /> slices. The small dense kernels
///     (<c>ddot</c>/<c>dnrm2</c>/<c>daxpy</c>/<c>dscal</c>/<c>dcopy</c> and the <c>dpotf2</c>/
///     <c>dtrsm</c> that back <c>dpotrf</c>/<c>dtrtrs</c>) are hand-written sequential routines — the
///     matrices here are tiny (dim ≤ <c>m</c> = 10), so they reproduce reference LAPACK closely.</para>
///
///     <para><b>Tolerance-faithful, not bit-exact.</b> The internal reductions and the system LAPACK
///     scipy links may differ from these managed kernels by ULPs, so the iterate trajectory tracks
///     scipy's to a tolerance rather than bit-for-bit — matching the plan's documented GP boundary
///     (the GP loss/gradient itself is already tolerance-exact vs autograd).</para>
/// </summary>
internal sealed class LBfgsB
{
    // enum Status
    private const int START = 0;
    private const int NEW_X = 1;
    private const int RESTART = 2;
    private const int FG = 3;
    private const int CONVERGENCE = 4;
    private const int STOP = 5;
    private const int WARNING = 6;
    private const int ERROR = 7;
    private const int ABNORMAL = 8;

    // enum StatusMsg
    private const int NO_MSG = 0;
    private const int FG_START = 301;
    private const int FG_LNSRCH = 302;
    private const int CONV_GRAD = 401;
    private const int CONV_F = 402;
    private const int STOP_CPU = 501;
    private const int STOP_ITER = 502;
    private const int STOP_GRAD = 503;
    private const int STOP_ITERC = 504;
    private const int STOP_CALLB = 505;
    private const int WARN_ROUND = 601;
    private const int WARN_STPMAX = 602;
    private const int WARN_STPMIN = 603;
    private const int WARN_XTOL = 604;
    private const int ERROR_NOFEAS = 701;
    private const int ERROR_FACTR = 702;
    private const int ERROR_INITG = 710;
    private const int ERROR_M = 711;
    private const int ERROR_N = 712;
    private const int ERROR_NBD = 713;
    private const int ERROR_STP1 = 706;
    private const int ERROR_STP2 = 707;
    private const int ERROR_FTOL = 703;
    private const int ERROR_GTOL = 704;
    private const int ERROR_XTOL = 705;
    private const int ERROR_STPMIN = 708;
    private const int ERROR_STPMAX = 709;

    private const double Epsmach = 2.220446049250313e-016;

    private readonly int _n;
    private readonly int _m;
    private readonly double[] _x;
    private readonly double[] _l;
    private readonly double[] _u;
    private readonly int[] _nbd;
    private readonly double[] _g;
    private double _f;
    private readonly double _factr;
    private readonly double _pgtol;
    private readonly int _maxls;

    // Compact-BFGS workspace (column-major flat arrays, mirroring the `wa` partition in `setulb`).
    private readonly double[] _ws;   // (n, m)
    private readonly double[] _wy;   // (n, m)
    private readonly double[] _sy;   // (m, m)
    private readonly double[] _ss;   // (m, m)
    private readonly double[] _wt;   // (m, m)
    private readonly double[] _wn;   // (2m, 2m)
    private readonly double[] _snd;  // (2m, 2m)
    private readonly double[] _z;
    private readonly double[] _r;
    private readonly double[] _d;
    private readonly double[] _t;
    private readonly double[] _xp;
    private readonly double[] _wa;   // 8m
    private readonly int[] _index;
    private readonly int[] _iwhere;
    private readonly int[] _indx2;

    // Reverse-communication persistent state (kept as arrays, exactly like the C `isave`/`dsave`).
    private int _task;
    private int _taskMsg;
    private int _lnTask;
    private int _lnTaskMsg;
    private readonly int[] _lsave = new int[4];
    private readonly int[] _isave = new int[44];
    private readonly double[] _dsave = new double[29];

    private LBfgsB(int n, int m, double[] x, double[] l, double[] u, int[] nbd,
        double factr, double pgtol, int maxls)
    {
        _n = n;
        _m = m;
        _x = x;
        _l = l;
        _u = u;
        _nbd = nbd;
        _factr = factr;
        _pgtol = pgtol;
        _maxls = maxls;
        _g = new double[n];

        _ws = new double[n * m];
        _wy = new double[n * m];
        _sy = new double[m * m];
        _ss = new double[m * m];
        _wt = new double[m * m];
        _wn = new double[4 * m * m];
        _snd = new double[4 * m * m];
        _z = new double[n];
        _r = new double[n];
        _d = new double[n];
        _t = new double[n];
        _xp = new double[n];
        _wa = new double[8 * m];
        _index = new int[n];
        _iwhere = new int[n];
        _indx2 = new int[n];
    }

    /// <summary>The result of a minimization: the optimum, its value, and run statistics.</summary>
    internal readonly record struct Result(double[] X, double F, double[] G, int NumIterations, int NumFuncEvals, int WarnFlag);

    /// <summary>
    ///     Minimize <paramref name="funcAndGrad" /> (value + gradient at a point) with L-BFGS-B —
    ///     the port of scipy's <c>_minimize_lbfgsb</c> driver loop. <paramref name="bounds" /> is a
    ///     per-dimension <c>(lower, upper)</c> with <see cref="double.NegativeInfinity" /> /
    ///     <see cref="double.PositiveInfinity" /> for one-sided or absent bounds; pass <c>null</c> for
    ///     a fully unconstrained problem (the GP fit's case).
    /// </summary>
    internal static Result Minimize(
        Func<double[], (double, double[])> funcAndGrad,
        double[] x0,
        (double Lower, double Upper)[]? bounds = null,
        int m = 10,
        double factr = 1e7,
        double pgtol = 1e-5,
        int maxfun = 15000,
        int maxiter = 15000,
        int maxls = 20)
    {
        int n = x0.Length;
        if (maxls <= 0)
            throw new ArgumentException("maxls must be positive.");

        var low = new double[n];
        var upp = new double[n];
        var nbd = new int[n];
        if (bounds != null)
        {
            if (bounds.Length != n)
                throw new ArgumentException("length of x0 != length of bounds");
            for (int i = 0; i < n; i++)
            {
                bool hasL = !double.IsNegativeInfinity(bounds[i].Lower);
                bool hasU = !double.IsPositiveInfinity(bounds[i].Upper);
                if (hasL) low[i] = bounds[i].Lower;
                if (hasU) upp[i] = bounds[i].Upper;
                // nbd: 0 unbounded, 1 lower, 2 both, 3 upper.
                nbd[i] = hasL ? (hasU ? 2 : 1) : (hasU ? 3 : 0);
            }
        }

        var x = (double[])x0.Clone();
        var solver = new LBfgsB(n, m, x, low, upp, nbd, factr, pgtol, maxls);
        solver._task = START;
        solver._taskMsg = NO_MSG;

        int nfev = 0;
        int nIterations = 0;
        double[] lastGrad = new double[n];

        while (true)
        {
            solver.Setulb();
            if (solver._task == FG)
            {
                (double fVal, double[] gVal) = funcAndGrad(x);
                nfev++;
                solver._f = fVal;
                Array.Copy(gVal, solver._g, n);
                lastGrad = gVal;
            }
            else if (solver._task == NEW_X)
            {
                nIterations++;
                if (nIterations >= maxiter)
                {
                    solver._task = STOP;
                    solver._taskMsg = STOP_ITERC;
                }
                else if (nfev > maxfun)
                {
                    solver._task = STOP;
                    solver._taskMsg = STOP_ITER;
                }
            }
            else
            {
                break;
            }
        }

        int warnFlag = solver._task == CONVERGENCE ? 0 : (nfev > maxfun || nIterations >= maxiter ? 1 : 2);
        return new Result(x, solver._f, lastGrad, nIterations, nfev, warnFlag);
    }

    /// <summary>Port of <c>setulb</c> — with the workspace held as named fields, it just drives <see cref="Mainlb" />.</summary>
    private void Setulb() => Mainlb();

    // ==== BLAS (inc = 1, sequential — every L-BFGS-B call uses unit stride) ======================

    private static double Ddot(int n, ReadOnlySpan<double> x, ReadOnlySpan<double> y)
    {
        double s = 0.0;
        for (int i = 0; i < n; i++)
            s += x[i] * y[i];
        return s;
    }

    private static double Dnrm2(int n, ReadOnlySpan<double> x)
    {
        // Reference netlib dnrm2: scaled sum of squares to avoid overflow/underflow.
        if (n < 1)
            return 0.0;
        if (n == 1)
            return Math.Abs(x[0]);
        double scale = 0.0;
        double ssq = 1.0;
        for (int i = 0; i < n; i++)
        {
            if (x[i] != 0.0)
            {
                double absxi = Math.Abs(x[i]);
                if (scale < absxi)
                {
                    double ratio = scale / absxi;
                    ssq = 1.0 + ssq * ratio * ratio;
                    scale = absxi;
                }
                else
                {
                    double ratio = absxi / scale;
                    ssq += ratio * ratio;
                }
            }
        }
        return scale * Math.Sqrt(ssq);
    }

    private static void Daxpy(int n, double alpha, ReadOnlySpan<double> x, Span<double> y)
    {
        if (alpha == 0.0)
            return;
        for (int i = 0; i < n; i++)
            y[i] += alpha * x[i];
    }

    private static void Dscal(int n, double alpha, Span<double> x)
    {
        for (int i = 0; i < n; i++)
            x[i] *= alpha;
    }

    private static void Dcopy(int n, ReadOnlySpan<double> x, Span<double> y)
    {
        for (int i = 0; i < n; i++)
            y[i] = x[i];
    }

    // ==== LAPACK: dpotrf (upper, column-major) + dtrtrs (upper, N/T, non-unit) ===================

    /// <summary>Port of <c>dpotrf(uplo='U')</c> via the unblocked <c>dpotf2</c> (n ≤ m is tiny).</summary>
    private static int Dpotrf(Span<double> a, int lda, int n)
    {
        for (int j = 0; j < n; j++)
        {
            double ajj = a[j + lda * j] - Ddot(j, Col(a, lda, j), Col(a, lda, j));
            if (!(ajj > 0.0))  // <= 0 or NaN
                return j + 1;
            ajj = Math.Sqrt(ajj);
            a[j + lda * j] = ajj;
            if (j < n - 1)
            {
                // Update row j: a[j, i] -= dot(col j, col i), then scale by 1/ajj (multiply).
                for (int i = j + 1; i < n; i++)
                    a[j + lda * i] -= Ddot(j, Col(a, lda, j), Col(a, lda, i));
                double recip = 1.0 / ajj;
                for (int i = j + 1; i < n; i++)
                    a[j + lda * i] *= recip;
            }
        }
        return 0;
    }

    // The top part of column c (rows 0..) of a column-major matrix with leading dimension lda.
    private static ReadOnlySpan<double> Col(Span<double> a, int lda, int c) => a.Slice(lda * c);

    /// <summary>Port of <c>dtrtrs(uplo='U', diag='N')</c> for <c>trans ∈ {'N','T'}</c> — column-major dtrsm (side='L').</summary>
    private static int Dtrtrs(char trans, Span<double> a, int lda, int n, Span<double> b, int ldb, int nrhs)
    {
        // Singularity check (LAPACK dtrtrs).
        for (int i = 0; i < n; i++)
            if (a[i + lda * i] == 0.0)
                return i + 1;

        if (trans == 'N')
        {
            // Solve A X = B, A upper — column-oriented back substitution (as reference dtrsm).
            for (int j = 0; j < nrhs; j++)
            {
                for (int k = n - 1; k >= 0; k--)
                {
                    b[k + ldb * j] /= a[k + lda * k];
                    double bkj = b[k + ldb * j];
                    for (int i = 0; i < k; i++)
                        b[i + ldb * j] -= bkj * a[i + lda * k];
                }
            }
        }
        else
        {
            // Solve A^T X = B (A^T lower) — row-oriented forward substitution (as reference dtrsm).
            for (int j = 0; j < nrhs; j++)
            {
                for (int i = 0; i < n; i++)
                {
                    double temp = b[i + ldb * j];
                    for (int k = 0; k < i; k++)
                        temp -= a[k + lda * i] * b[k + ldb * j];
                    b[i + ldb * j] = temp / a[i + lda * i];
                }
            }
        }
        return 0;
    }

    // ==== the reverse-communication driver (mainlb) and its subroutines =========================

    private void Mainlb()
    {
        Span<double> ws = _ws, wy = _wy, sy = _sy, ss = _ss, wt = _wt, wn = _wn;
        Span<double> z = _z, r = _r, d = _d, t = _t, xp = _xp, wa = _wa;
        Span<int> index = _index, iwhere = _iwhere, indx2 = _indx2;
        Span<int> isave = _isave.AsSpan(21);
        Span<double> dsave = _dsave;
        Span<int> lsave = _lsave;

        int i, k = 0, nintol = 0, iback = 0, nskip = 0, head = 0, col = 0, iter = 0, itail = 0, iupdat = 0;
        int nseg = 0, nfgv = 0, info = 0, ifun = 0, iword = 0, nfree = 0, nact = 0, ileave = 0, nenter = 0;
        double theta = 1.0, fold = 0.0, dr = 0.0, rr = 0.0, tol = 0.0, xstep = 0.0, sbgnrm = 0.0;
        double stpmx = 0.0, ddum = 0.0, dnorm = 0.0, dtd = 0.0, gd = 0.0, gdold = 0.0, stp = 0.0;
        int prjctd = 0, cnstnd = 0, boxed = 0, updatd = 0, wrk = 0;

        if (_task == START)
        {
            col = 0; head = 0; theta = 1.0; iupdat = 0; updatd = 0; iback = 0;
            itail = 0; iword = 0; nact = 0; ileave = 0; nenter = 0; fold = 0.0;
            dnorm = 0.0; gd = 0.0; sbgnrm = 0.0; stp = 0.0; gdold = 0.0; dtd = 0.0;
            iter = 0; nfgv = 0; nseg = 0; nintol = 0; nskip = 0; nfree = _n; ifun = 0;
            tol = _factr * Epsmach;
            info = 0;
            Errclb(ref info, out k);
            if (_task == ERROR) { return; }
            Active(out prjctd, out cnstnd, out boxed);
        }
        else
        {
            prjctd = lsave[0]; cnstnd = lsave[1]; boxed = lsave[2]; updatd = lsave[3];
            nintol = isave[0]; iback = isave[3]; nskip = isave[4]; head = isave[5];
            col = isave[6]; itail = isave[7]; iter = isave[8]; iupdat = isave[9];
            nseg = isave[11]; nfgv = isave[12]; info = isave[13]; ifun = isave[14];
            iword = isave[15]; nfree = isave[16]; nact = isave[17]; ileave = isave[18];
            nenter = isave[19];
            theta = dsave[0]; fold = dsave[1]; tol = dsave[2]; dnorm = dsave[3];
            gd = dsave[10]; stpmx = dsave[11]; sbgnrm = dsave[12]; stp = dsave[13];
            gdold = dsave[14]; dtd = dsave[15];

            if (_task == FG && _taskMsg == FG_LNSRCH) goto LINE666;
            if (_task == NEW_X) goto LINE777;
            if (_task == FG && _taskMsg == FG_START) goto LINE111;
            if (_task == STOP)
            {
                if (_taskMsg == STOP_CPU) { Dcopy(_n, t, _x); Dcopy(_n, r, _g); _f = fold; }
                goto LINE999;
            }
        }

        _task = FG;
        _taskMsg = FG_START;
        goto LINE1000;

    LINE111:
        nfgv = 1;
        Projgr(out sbgnrm);
        if (sbgnrm <= _pgtol) { _task = CONVERGENCE; _taskMsg = CONV_GRAD; goto LINE999; }

    LINE222:
        iword = -1;
        if (cnstnd == 0 && col > 0)
        {
            Dcopy(_n, _x, z);
            wrk = updatd;
            nseg = 0;
            goto LINE333;
        }

        Cauchy(t, d, z, wy, ws, sy, wt, theta, col, head,
               wa, wa.Slice(2 * _m), wa.Slice(4 * _m), wa.Slice(6 * _m), out nseg, sbgnrm, ref info);
        if (info != 0)
        {
            info = 0; col = 0; head = 0; theta = 1.0; iupdat = 0; updatd = 0;
            goto LINE222;
        }
        nintol += nseg;
        Freev(ref nfree, index, ref nenter, ref ileave, indx2, iwhere, ref wrk, updatd, cnstnd, iter);
        nact = _n - nfree;

    LINE333:
        if (nfree == 0 || col == 0) goto LINE555;
        if (wrk != 0)
            Formk(nfree, index, nenter, ileave, indx2, iupdat, updatd, wn, _snd, ws, wy, sy, theta, col, head, ref info);
        if (info != 0)
        {
            info = 0; col = 0; head = 0; theta = 1.0; iupdat = 0; updatd = 0;
            goto LINE222;
        }
        Cmprlb(_x, _g, ws, wy, sy, wt, z, r, wa, index, theta, col, head, nfree, cnstnd, ref info);
        if (info != 0) goto LINE444;
        Subsm(nfree, index, z, r, xp, ws, wy, theta, _x, _g, col, head, ref iword, wa, wn, ref info);

    LINE444:
        if (info != 0)
        {
            info = 0; col = 0; head = 0; theta = 1.0; iupdat = 0; updatd = 0;
            goto LINE222;
        }

    LINE555:
        for (i = 0; i < _n; i++)
            d[i] = z[i] - _x[i];

    LINE666:
        Lnsrlb(_f, ref fold, ref gd, ref gdold, _g, d, r, t, z, ref stp, ref dnorm, ref dtd,
               ref xstep, ref stpmx, iter, ref ifun, ref iback, ref nfgv, ref info,
               boxed, cnstnd, isave.Slice(21), dsave.Slice(16));

        if (info != 0 || iback >= _maxls)
        {
            Dcopy(_n, t, _x); Dcopy(_n, r, _g); _f = fold;
            if (col == 0)
            {
                if (info == 0) { info = -9; nfgv--; ifun--; iback--; }
                _task = ABNORMAL; _taskMsg = NO_MSG;
                iter++;
                goto LINE999;
            }
            else
            {
                if (info == 0) nfgv--;
                info = 0; col = 0; head = 0; theta = 1.0; iupdat = 0; updatd = 0;
                _task = RESTART; _taskMsg = NO_MSG;
                goto LINE222;
            }
        }
        else if (_task == FG && _taskMsg == FG_LNSRCH)
        {
            goto LINE1000;
        }
        else
        {
            iter++;
            Projgr(out sbgnrm);
            goto LINE1000;
        }

    LINE777:
        if (sbgnrm <= _pgtol) { _task = CONVERGENCE; _taskMsg = CONV_GRAD; goto LINE999; }
        ddum = Math.Max(Math.Max(Math.Abs(fold), Math.Abs(_f)), 1.0);
        if (fold - _f <= tol * ddum)
        {
            _task = CONVERGENCE; _taskMsg = CONV_F;
            if (iback >= 10) info = -5;
            goto LINE999;
        }

        for (i = 0; i < _n; i++)
            r[i] = _g[i] - r[i];
        rr = Math.Pow(Dnrm2(_n, r), 2.0);
        if (stp == 1.0) { dr = gd - gdold; ddum = -gdold; }
        else { dr = (gd - gdold) * stp; Dscal(_n, stp, d); ddum = -gdold * stp; }

        if (dr <= Epsmach * ddum) { nskip += 1; updatd = 0; goto LINE888; }

        updatd = 1;
        iupdat += 1;
        Matupd(ws, wy, sy, ss, d, r, ref itail, iupdat, ref col, ref head, ref theta, rr, dr, stp, dtd);
        Formt(wt, sy, ss, col, theta, ref info);
        if (info != 0)
        {
            info = 0; col = 0; head = 0; theta = 1.0; iupdat = 0; updatd = 0;
            goto LINE222;
        }

    LINE888:
        goto LINE222;

    LINE999:
        ;

    LINE1000:
        lsave[0] = prjctd; lsave[1] = cnstnd; lsave[2] = boxed; lsave[3] = updatd;
        isave[0] = nintol; isave[3] = iback; isave[4] = nskip; isave[5] = head;
        isave[6] = col; isave[7] = itail; isave[8] = iter; isave[9] = iupdat;
        isave[11] = nseg; isave[12] = nfgv; isave[13] = info; isave[14] = ifun;
        isave[15] = iword; isave[16] = nfree; isave[17] = nact; isave[18] = ileave;
        isave[19] = nenter;  // global isave[30] (scipy's n_bfgs_updates) == this view's isave[9] = iupdat
        dsave[0] = theta; dsave[1] = fold; dsave[2] = tol; dsave[3] = dnorm;
        dsave[10] = gd; dsave[11] = stpmx; dsave[12] = sbgnrm; dsave[13] = stp;
        dsave[14] = gdold; dsave[15] = dtd;
        _ = k;
    }

    private void Errclb(ref int info, out int k)
    {
        k = 0;
        if (_n <= 0) { _task = ERROR; _taskMsg = ERROR_N; }
        if (_m <= 0) { _task = ERROR; _taskMsg = ERROR_M; }
        if (_factr < 0) { _task = ERROR; _taskMsg = ERROR_FACTR; }
        for (int i = 0; i < _n; i++)
        {
            if (_nbd[i] < 0 || _nbd[i] > 3) { _task = ERROR; _taskMsg = ERROR_NBD; info = -6; k = i; }
            if (_nbd[i] == 2 && _l[i] > _u[i]) { _task = ERROR; _taskMsg = ERROR_NOFEAS; info = -7; k = i; }
        }
    }

    private void Active(out int prjctd, out int cnstnd, out int boxed)
    {
        prjctd = 0; cnstnd = 0; boxed = 1;
        for (int i = 0; i < _n; i++)
        {
            if (_nbd[i] > 0)
            {
                if (_nbd[i] <= 2 && _x[i] <= _l[i])
                {
                    if (_x[i] < _l[i]) { prjctd = 1; _x[i] = _l[i]; }
                }
                else if (_nbd[i] >= 2 && _x[i] >= _u[i])
                {
                    if (_x[i] > _u[i]) { prjctd = 1; _x[i] = _u[i]; }
                }
            }
        }
        for (int i = 0; i < _n; i++)
        {
            if (_nbd[i] != 2) boxed = 0;
            if (_nbd[i] == 0) { _iwhere[i] = -1; }
            else
            {
                cnstnd = 1;
                _iwhere[i] = (_nbd[i] == 2 && _u[i] - _l[i] <= 0.0) ? 3 : 0;
            }
        }
    }

    private void Projgr(out double sbgnrm)
    {
        sbgnrm = 0.0;
        for (int i = 0; i < _n; i++)
        {
            double gi = _g[i];
            if (double.IsNaN(gi)) { sbgnrm = gi; return; }
            if (_nbd[i] != 0)
            {
                if (gi < 0.0) { if (_nbd[i] >= 2) gi = Math.Max(_x[i] - _u[i], gi); }
                else { if (_nbd[i] <= 2) gi = Math.Min(_x[i] - _l[i], gi); }
            }
            sbgnrm = Math.Max(sbgnrm, Math.Abs(gi));
        }
    }

    private void Bmv(Span<double> sy, Span<double> wt, int col, Span<double> v, Span<double> p, ref int info)
    {
        int m = _m;
        if (col == 0) return;
        p[col] = v[col];
        for (int i = 1; i < col; i++)
        {
            int i2 = col + i;
            double ssum = 0.0;
            for (int kk = 0; kk <= i - 1; kk++)
                ssum += sy[i + m * kk] * v[kk] / sy[kk + m * kk];
            p[i2] = v[i2] + ssum;
        }
        info = Dtrtrs('T', wt, m, col, p.Slice(col), col, 1);
        if (info != 0) return;
        for (int i = 0; i < col; i++)
            p[i] = v[i] / Math.Sqrt(sy[i + m * i]);
        info = Dtrtrs('N', wt, m, col, p.Slice(col), col, 1);
        if (info != 0) return;
        for (int i = 0; i < col; i++)
            p[i] = -p[i] / Math.Sqrt(sy[i + m * i]);
        for (int i = 0; i < col; i++)
        {
            double ssum = 0.0;
            for (int kk = i + 1; kk < col; kk++)
                ssum += sy[kk + m * i] * p[col + kk] / sy[i + m * i];
            p[i] += ssum;
        }
    }

    private void Freev(ref int nfree, Span<int> idx, ref int nenter, ref int ileave, Span<int> idx2,
        Span<int> iwhere, ref int wrk, int updatd, int cnstnd, int iter)
    {
        nenter = 0;
        ileave = _n;
        if (iter > 0 && cnstnd != 0)
        {
            for (int i = 0; i < nfree; i++)
            {
                int kk = idx[i];
                if (iwhere[kk] > 0) { ileave -= 1; idx2[ileave] = kk; }
            }
            for (int i = nfree; i < _n; i++)
            {
                int kk = idx[i];
                if (iwhere[kk] <= 0) { idx2[nenter] = kk; nenter += 1; }
            }
        }
        wrk = (ileave < _n || nenter > 0 || updatd != 0) ? 1 : 0;
        nfree = 0;
        int iact = _n;
        for (int i = 0; i < _n; i++)
        {
            if (iwhere[i] <= 0) { idx[nfree] = i; nfree += 1; }
            else { iact -= 1; idx[iact] = i; }
        }
    }

    private void Hpsolb(int n, Span<double> t, Span<int> iorder, int iheap)
    {
        int i, j;
        double ddum;
        if (iheap == 0)
        {
            for (int kk = 2; kk <= n + 1; kk++)
            {
                ddum = t[kk - 1];
                int indxin = iorder[kk - 1];
                i = kk;
                while (true)
                {
                    if (i > 1)
                    {
                        j = i / 2;
                        if (ddum < t[j - 1]) { t[i - 1] = t[j - 1]; iorder[i - 1] = iorder[j - 1]; i = j; }
                        else break;
                    }
                    else break;
                }
                t[i - 1] = ddum;
                iorder[i - 1] = indxin;
            }
        }
        if (n > 0)
        {
            i = 1;
            double outv = t[0];
            int indxout = iorder[0];
            ddum = t[n];
            int indxin = iorder[n];
            while (true)
            {
                j = i + i;
                if (j <= n)
                {
                    if (t[j] < t[j - 1]) j++;
                    if (t[j - 1] < ddum) { t[i - 1] = t[j - 1]; iorder[i - 1] = iorder[j - 1]; i = j; }
                    else break;
                }
                else break;
            }
            t[i - 1] = ddum;
            iorder[i - 1] = indxin;
            t[n] = outv;
            iorder[n] = indxout;
        }
    }

    private void Cauchy(Span<double> t, Span<double> d, Span<double> xcp, Span<double> wy, Span<double> ws,
        Span<double> sy, Span<double> wt, double theta, int col, int head,
        Span<double> p, Span<double> c, Span<double> wbp, Span<double> v, out int nseg, double sbgnrm, ref int info)
    {
        int n = _n, m = _m;
        Span<int> iorder = _indx2, iwhere = _iwhere;
        nseg = 0;
        double f1 = 0.0, tl = 0.0, tu = 0.0;
        if (sbgnrm <= 0.0) { Dcopy(n, _x, xcp); return; }
        int bnded = 1, nfree = n, nbreak = -1, ibkmin = 0;
        double bkmin = 0.0;
        int col2 = 2 * col;
        for (int i = 0; i < col2; i++) p[i] = 0.0;

        for (int i = 0; i < n; i++)
        {
            double neggi = -_g[i];
            if (iwhere[i] != 3 && iwhere[i] != -1)
            {
                if (_nbd[i] <= 2) tl = _x[i] - _l[i];
                if (_nbd[i] >= 2) tu = _u[i] - _x[i];
                bool xlower = _nbd[i] <= 2 && tl <= 0.0;
                bool xupper = _nbd[i] >= 2 && tu <= 0.0;
                iwhere[i] = 0;
                if (xlower) { if (neggi <= 0.0) iwhere[i] = 1; }
                else if (xupper) { if (neggi >= 0.0) iwhere[i] = 2; }
                else { if (Math.Abs(neggi) <= 0.0) iwhere[i] = -3; }
            }
            int pointr = head;
            if (iwhere[i] != 0 && iwhere[i] != -1) { d[i] = 0.0; }
            else
            {
                d[i] = neggi;
                f1 -= neggi * neggi;
                for (int j = 0; j < col; j++)
                {
                    p[j] += wy[i + n * pointr] * neggi;
                    p[col + j] += ws[i + n * pointr] * neggi;
                    pointr = (pointr + 1) % m;
                }
                if (_nbd[i] <= 2 && _nbd[i] != 0 && neggi < 0.0)
                {
                    nbreak += 1; iorder[nbreak] = i; t[nbreak] = tl / (-neggi);
                    if (nbreak == 0 || t[nbreak] < bkmin) { bkmin = t[nbreak]; ibkmin = nbreak; }
                }
                else if (_nbd[i] >= 2 && neggi > 0.0)
                {
                    nbreak += 1; iorder[nbreak] = i; t[nbreak] = tu / neggi;
                    if (nbreak == 0 || t[nbreak] < bkmin) { bkmin = t[nbreak]; ibkmin = nbreak; }
                }
                else { nfree -= 1; iorder[nfree] = i; if (Math.Abs(neggi) > 0.0) bnded = 0; }
            }
        }

        if (theta != 1.0) Dscal(col, theta, p.Slice(col));
        Dcopy(n, _x, xcp);
        if (nbreak == -1 && nfree == n) return;
        for (int j = 0; j < col2; j++) c[j] = 0.0;

        double f2 = -theta * f1;
        double f2_org = f2;
        if (col > 0)
        {
            Bmv(sy, wt, col, p, v, ref info);
            if (info != 0) return;
            f2 -= Ddot(col2, v, p);
        }
        double dtm = -f1 / f2;
        double tsum = 0.0;
        nseg = 1;
        if (nbreak == -1) goto LINE888;
        int nleft = nbreak, iter = 0;
        double tj = 0.0;

        while (true)
        {
            double tj0 = tj;
            int ibp;
            if (iter == 0) { tj = bkmin; ibp = iorder[ibkmin]; }
            else
            {
                if (iter == 1)
                {
                    if (ibkmin != nbreak) { t[ibkmin] = t[nbreak]; iorder[ibkmin] = iorder[nbreak]; }
                }
                Hpsolb(nleft, t, iorder, iter - 1);
                tj = t[nleft]; ibp = iorder[nleft];
            }
            double dt = tj - tj0;
            if (dtm < dt) goto LINE888;
            tsum += dt;
            nleft -= 1;
            iter += 1;
            double dibp = d[ibp];
            d[ibp] = 0.0;
            double zibp;
            if (dibp > 0.0) { zibp = _u[ibp] - _x[ibp]; xcp[ibp] = _u[ibp]; iwhere[ibp] = 2; }
            else { zibp = _l[ibp] - _x[ibp]; xcp[ibp] = _l[ibp]; iwhere[ibp] = 1; }
            if (nleft == -1 && nbreak == n) { dtm = dt; goto LINE999; }
            nseg += 1;
            double dibp2 = Math.Pow(dibp, 2.0);
            f1 = f1 + dt * f2 + dibp2 - theta * dibp * zibp;
            f2 -= theta * dibp2;
            if (col > 0)
            {
                Daxpy(col2, dt, p, c);
                int pointr = head;
                for (int j = 0; j < col; j++)
                {
                    wbp[j] = wy[ibp + n * pointr];
                    wbp[col + j] = theta * ws[ibp + n * pointr];
                    pointr = (pointr + 1) % m;
                }
                Bmv(sy, wt, col, wbp, v, ref info);
                if (info != 0) return;
                double wmc = Ddot(col2, c, v);
                double wmp = Ddot(col2, p, v);
                double wmw = Ddot(col2, wbp, v);
                dibp = -dibp;
                Daxpy(col2, dibp, wbp, p);
                dibp = -dibp;
                f1 += dibp * wmc;
                f2 += dibp * 2.0 * wmp - dibp2 * wmw;
            }
            f2 = Math.Max(Epsmach * f2_org, f2);
            if (nleft >= 0) { dtm = -f1 / f2; }
            else if (bnded != 0) { dtm = 0.0; break; }
            else { dtm = -f1 / f2; break; }
        }

    LINE888:
        if (dtm <= 0.0) dtm = 0.0;
        tsum += dtm;
        Daxpy(n, tsum, d, xcp);

    LINE999:
        if (col > 0) Daxpy(col2, dtm, p, c);
    }

    private void Cmprlb(Span<double> x, Span<double> g, Span<double> ws, Span<double> wy, Span<double> sy,
        Span<double> wt, Span<double> z, Span<double> r, Span<double> wa, Span<int> index,
        double theta, int col, int head, int nfree, int cnstnd, ref int info)
    {
        int n = _n, m = _m;
        if (cnstnd == 0 && col > 0)
        {
            for (int i = 0; i < n; i++) r[i] = -g[i];
        }
        else
        {
            for (int i = 0; i < nfree; i++)
            {
                int kk = index[i];
                r[i] = -theta * (z[kk] - x[kk]) - g[kk];
            }
            Bmv(sy, wt, col, wa.Slice(2 * m), wa, ref info);
            if (info != 0) { info = -8; return; }
            int pointr = head;
            for (int j = 0; j < col; j++)
            {
                double a1 = wa[j];
                double a2 = theta * wa[col + j];
                for (int i = 0; i < nfree; i++)
                {
                    int kk = index[i];
                    r[i] += wy[kk + n * pointr] * a1 + ws[kk + n * pointr] * a2;
                }
                pointr = (pointr + 1) % m;
            }
        }
    }

    private void Matupd(Span<double> ws, Span<double> wy, Span<double> sy, Span<double> ss,
        Span<double> d, Span<double> r, ref int itail, int iupdat, ref int col, ref int head,
        ref double theta, double rr, double dr, double stp, double dtd)
    {
        int n = _n, m = _m;
        if (iupdat <= m) { col = iupdat; itail = (head + iupdat - 1) % m; }
        else { itail = (itail + 1) % m; head = (head + 1) % m; }
        Dcopy(n, d, ws.Slice(itail * n));
        Dcopy(n, r, wy.Slice(itail * n));
        theta = rr / dr;
        if (iupdat > m)
        {
            for (int j = 1; j < col; j++)
            {
                Dcopy(j, ss.Slice(1 + m * j), ss.Slice(m * (j - 1)));
                Dcopy(col - j, sy.Slice(j + m * j), sy.Slice((j - 1) + m * (j - 1)));
            }
        }
        int pointr = head;
        for (int j = 0; j < col - 1; j++)
        {
            sy[col - 1 + m * j] = Ddot(n, d, wy.Slice(pointr * n));
            ss[j + m * (col - 1)] = Ddot(n, ws.Slice(pointr * n), d);
            pointr = (pointr + 1) % m;
        }
        ss[(col - 1) + m * (col - 1)] = stp == 1.0 ? dtd : stp * stp * dtd;
        sy[(col - 1) + m * (col - 1)] = dr;
    }

    private void Formt(Span<double> wt, Span<double> sy, Span<double> ss, int col, double theta, ref int info)
    {
        int m = _m;
        for (int j = 0; j < col; j++)
            wt[m * j] = theta * ss[m * j];
        for (int i = 1; i < col; i++)
        {
            for (int j = i; j < col; j++)
            {
                int k1 = i > j ? j : i;
                double ddum = 0.0;
                for (int kk = 0; kk < k1; kk++)
                    ddum += sy[i + m * kk] * sy[j + m * kk] / sy[kk + m * kk];
                wt[i + m * j] = ddum + theta * ss[i + m * j];
            }
        }
        info = Dpotrf(wt, m, col);
        if (info != 0) info = -3;
    }

    private void Formk(int nsub, Span<int> ind, int nenter, int ileave, Span<int> indx2, int iupdat,
        int updatd, Span<double> wn, Span<double> wn1, Span<double> ws, Span<double> wy, Span<double> sy,
        double theta, int col, int head, ref int info)
    {
        int n = _n, m = _m;
        int m2 = 2 * m;
        int upcl;
        if (updatd != 0)
        {
            if (iupdat > m)
            {
                for (int jy = 0; jy < m - 1; jy++)
                {
                    int js = m + jy;
                    int len = m - (jy + 1);
                    Dcopy(len, wn1.Slice((jy + 1) + 2 * m * (jy + 1)), wn1.Slice(jy + 2 * m * jy));
                    Dcopy(len, wn1.Slice((js + 1) + 2 * m * (js + 1)), wn1.Slice(js + 2 * m * js));
                    Dcopy(m - 1, wn1.Slice((m + 1) + 2 * m * (jy + 1)), wn1.Slice(m + 2 * m * jy));
                }
            }
            int pbegin = 0, pend = nsub, dbegin = nsub, dend = n;
            int iy = col - 1;
            int is0 = m + col - 1;
            int ipntr = head + col - 1;
            if (ipntr >= m) ipntr -= m;
            int jpntr = head;
            for (int jy = 0; jy < col; jy++)
            {
                int js = m + jy;
                double temp1 = 0.0, temp2 = 0.0, temp3 = 0.0;
                for (int kk = pbegin; kk < pend; kk++)
                {
                    int k1 = ind[kk];
                    temp1 += wy[k1 + n * ipntr] * wy[k1 + n * jpntr];
                }
                for (int kk = dbegin; kk < dend; kk++)
                {
                    int k1 = ind[kk];
                    temp2 += ws[k1 + n * ipntr] * ws[k1 + n * jpntr];
                    temp3 += ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
                }
                wn1[iy + 2 * m * jy] = temp1;
                wn1[is0 + 2 * m * js] = temp2;
                wn1[is0 + 2 * m * jy] = temp3;
                jpntr = (jpntr + 1) % m;
            }
            int jy2 = col - 1;
            jpntr = head + col - 1;
            if (jpntr >= m) jpntr -= m;
            ipntr = head;
            for (int i = 0; i < col; i++)
            {
                int is1 = m + i;
                double temp3 = 0.0;
                for (int kk = pbegin; kk < pend; kk++)
                {
                    int k1 = ind[kk];
                    temp3 += ws[k1 + n * ipntr] * wy[k1 + n * jpntr];
                }
                ipntr = (ipntr + 1) % m;
                wn1[is1 + 2 * m * jy2] = temp3;
            }
            upcl = col - 1;
        }
        else { upcl = col; }

        int ip = head;
        for (int iy = 0; iy < upcl; iy++)
        {
            int is0 = m + iy;
            int jp = head;
            for (int jy = 0; jy <= iy; jy++)
            {
                int js = m + jy;
                double temp1 = 0.0, temp2 = 0.0, temp3 = 0.0, temp4 = 0.0;
                for (int kk = 0; kk < nenter; kk++)
                {
                    int k1 = indx2[kk];
                    temp1 += wy[k1 + n * ip] * wy[k1 + n * jp];
                    temp2 += ws[k1 + n * ip] * ws[k1 + n * jp];
                }
                for (int kk = ileave; kk < n; kk++)
                {
                    int k1 = indx2[kk];
                    temp3 += wy[k1 + n * ip] * wy[k1 + n * jp];
                    temp4 += ws[k1 + n * ip] * ws[k1 + n * jp];
                }
                wn1[iy + 2 * m * jy] += temp1 - temp3;
                wn1[is0 + 2 * m * js] += -temp2 + temp4;
                jp = (jp + 1) % m;
            }
            ip = (ip + 1) % m;
        }
        ip = head;
        for (int is0 = m; is0 < m + upcl; is0++)
        {
            int jp = head;
            for (int jy = 0; jy < upcl; jy++)
            {
                double temp1 = 0.0, temp3 = 0.0;
                for (int kk = 0; kk < nenter; kk++)
                {
                    int k1 = indx2[kk];
                    temp1 += ws[k1 + n * ip] * wy[k1 + n * jp];
                }
                for (int kk = ileave; kk < n; kk++)
                {
                    int k1 = indx2[kk];
                    temp3 += ws[k1 + n * ip] * wy[k1 + n * jp];
                }
                if (is0 <= jy + m) wn1[is0 + 2 * m * jy] += temp1 - temp3;
                else wn1[is0 + 2 * m * jy] += -temp1 + temp3;
                jp = (jp + 1) % m;
            }
            ip = (ip + 1) % m;
        }

        for (int iy = 0; iy < col; iy++)
        {
            int is0 = col + iy;
            int is1 = m + iy;
            for (int jy = 0; jy <= iy; jy++)
            {
                int js = col + jy;
                int js1 = m + jy;
                wn[jy + m2 * iy] = wn1[iy + m2 * jy] / theta;
                wn[js + m2 * is0] = wn1[is1 + m2 * js1] * theta;
            }
            for (int jy = 0; jy < iy; jy++)
                wn[jy + m2 * is0] = -wn1[is1 + m2 * jy];
            for (int jy = iy; jy < col; jy++)
                wn[jy + m2 * is0] = wn1[is1 + m2 * jy];
            wn[iy + m2 * iy] += sy[iy + m * iy];
        }

        info = Dpotrf(wn, m2, col);
        if (info != 0) { info = -1; return; }
        int col2 = 2 * col;
        info = Dtrtrs('T', wn, m2, col, wn.Slice(m2 * col), m2, col);
        for (int is0 = col; is0 < col2; is0++)
        {
            for (int js = is0; js < col2; js++)
                wn[is0 + m2 * js] += Ddot(col, wn.Slice(m2 * is0), wn.Slice(m2 * js));
        }
        info = Dpotrf(wn.Slice(col + m2 * col), m2, col);
        if (info != 0) info = -2;
    }

    private void Subsm(int nsub, Span<int> ind, Span<double> x, Span<double> d, Span<double> xp,
        Span<double> ws, Span<double> wy, double theta, Span<double> xx, Span<double> gg,
        int col, int head, ref int iword, Span<double> wv, Span<double> wn, ref int info)
    {
        int n = _n, m = _m;
        if (nsub <= 0) return;
        int pointr = head;
        for (int i = 0; i < col; i++)
        {
            double temp1 = 0.0, temp2 = 0.0;
            for (int j = 0; j < nsub; j++)
            {
                int kk = ind[j];
                temp1 += wy[kk + n * pointr] * d[j];
                temp2 += ws[kk + n * pointr] * d[j];
            }
            wv[i] = temp1;
            wv[col + i] = theta * temp2;
            pointr = (pointr + 1) % m;
        }
        int m2 = 2 * m, col2 = 2 * col;
        info = Dtrtrs('T', wn, m2, col2, wv, m2, 1);
        if (info != 0) return;
        for (int i = 0; i < col; i++) wv[i] = -wv[i];
        info = Dtrtrs('N', wn, m2, col2, wv, m2, 1);
        if (info != 0) return;
        pointr = head;
        for (int jy = 0; jy < col; jy++)
        {
            int js = col + jy;
            for (int i = 0; i < nsub; i++)
            {
                int kk = ind[i];
                d[i] += wy[kk + n * pointr] * wv[jy] / theta + ws[kk + n * pointr] * wv[js];
            }
            pointr = (pointr + 1) % m;
        }
        Dscal(nsub, 1.0 / theta, d);

        iword = 0;
        Dcopy(n, x, xp);
        for (int i = 0; i < nsub; i++)
        {
            int kk = ind[i];
            double dk = d[i], xk = x[kk];
            if (_nbd[kk] != 0)
            {
                if (_nbd[kk] == 1) { x[kk] = Math.Max(_l[kk], xk + dk); if (x[kk] == _l[kk]) iword = 1; }
                else if (_nbd[kk] == 2)
                {
                    xk = Math.Max(_l[kk], xk + dk);
                    x[kk] = Math.Min(_u[kk], xk);
                    if (x[kk] == _l[kk] || x[kk] == _u[kk]) iword = 1;
                }
                else if (_nbd[kk] == 3) { x[kk] = Math.Min(_u[kk], xk + dk); if (x[kk] == _u[kk]) iword = 1; }
            }
            else { x[kk] = xk + dk; }
        }
        if (iword == 0) return;

        double dd_p = 0.0;
        for (int i = 0; i < n; i++) dd_p += (x[i] - xx[i]) * gg[i];
        if (dd_p > 0.0) { Dcopy(n, xp, x); }
        else return;

        double alpha = 1.0, temp1b = 1.0;
        int ibd = 0;
        for (int i = 0; i < nsub; i++)
        {
            int kk = ind[i];
            double dk = d[i];
            if (_nbd[kk] != 0)
            {
                if (dk < 0.0 && _nbd[kk] <= 2)
                {
                    double temp2 = _l[kk] - x[kk];
                    if (temp2 >= 0.0) temp1b = 0.0;
                    else if (dk * alpha < temp2) temp1b = temp2 / dk;
                }
                else if (dk > 0.0 && _nbd[kk] >= 2)
                {
                    double temp2 = _u[kk] - x[kk];
                    if (temp2 <= 0.0) temp1b = 0.0;
                    else if (dk * alpha > temp2) temp1b = temp2 / dk;
                }
                if (temp1b < alpha) { alpha = temp1b; ibd = i; }
            }
        }
        if (alpha < 1.0)
        {
            double dk = d[ibd];
            int kk = ind[ibd];
            if (dk > 0.0) { x[kk] = _u[kk]; d[ibd] = 0.0; }
            else if (dk < 0.0) { x[kk] = _l[kk]; d[ibd] = 0.0; }
        }
        for (int i = 0; i < nsub; i++)
        {
            int kk = ind[i];
            x[kk] += alpha * d[i];
        }
    }

    private void Lnsrlb(double f, ref double fold, ref double gd, ref double gdold,
        Span<double> g, Span<double> d, Span<double> r, Span<double> t, Span<double> z,
        ref double stp, ref double dnorm, ref double dtd, ref double xstep, ref double stpmx,
        int iter, ref int ifun, ref int iback, ref int nfgv, ref int info,
        int boxed, int cnstnd, Span<int> dcsrchIsave, Span<double> dcsrchDsave)
    {
        int n = _n;
        double ftol = 1e-3, gtol = 0.9, xtol = 0.1;
        if (_taskMsg != FG_LNSRCH)
        {
            dnorm = Dnrm2(n, d);
            dtd = Math.Pow(dnorm, 2.0);
            stpmx = 1.0e10;
            if (cnstnd != 0)
            {
                if (iter == 0) { stpmx = 1.0; }
                else
                {
                    for (int i = 0; i < n; i++)
                    {
                        double a1 = d[i];
                        if (_nbd[i] != 0)
                        {
                            if (a1 < 0.0 && _nbd[i] <= 2)
                            {
                                double a2 = _l[i] - _x[i];
                                if (a2 >= 0.0) stpmx = 0.0;
                                else if (a1 * stpmx < a2) stpmx = a2 / a1;
                            }
                            else if (a1 > 0.0 && _nbd[i] >= 2)
                            {
                                double a2 = _u[i] - _x[i];
                                if (a2 <= 0.0) stpmx = 0.0;
                                else if (a1 * stpmx > a2) stpmx = a2 / a1;
                            }
                        }
                    }
                }
            }
            if (iter == 0 && boxed == 0) stp = Math.Min(1.0 / dnorm, stpmx);
            else stp = 1.0;
            for (int i = 0; i < n; i++) { t[i] = _x[i]; r[i] = g[i]; }
            fold = f;
            ifun = 0;
            iback = 0;
            _lnTask = START;
            _lnTaskMsg = NO_MSG;
        }

        gd = Ddot(n, g, d);
        if (ifun == 0)
        {
            gdold = gd;
            if (gd >= 0.0) { info = -4; return; }
        }
        Dcsrch(f, gd, ref stp, ftol, gtol, xtol, 0.0, stpmx, ref _lnTask, ref _lnTaskMsg, dcsrchIsave, dcsrchDsave);
        xstep = stp * dnorm;
        if (_lnTask != CONVERGENCE && _lnTask != WARNING)
        {
            _task = FG; _taskMsg = FG_LNSRCH;
            ifun += 1; nfgv += 1; iback = ifun - 1;
            if (stp == 1.0) { Dcopy(n, z, _x); }
            else
            {
                for (int i = 0; i < n; i++)
                {
                    _x[i] = stp * d[i] + t[i];
                    if (_nbd[i] == 1 || _nbd[i] == 2) _x[i] = Math.Max(_x[i], _l[i]);
                    if (_nbd[i] == 2 || _nbd[i] == 3) _x[i] = Math.Min(_x[i], _u[i]);
                }
            }
        }
        else { _task = NEW_X; _taskMsg = NO_MSG; }
    }

    private void Dcsrch(double f, double g, ref double stp, double ftol, double gtol, double xtol,
        double stpmin, double stpmax, ref int task, ref int taskMsg, Span<int> isave, Span<double> dsave)
    {
        double xtrapl = 1.1, xtrapu = 4.0;
        int brackt, stage;
        double finit, ftest, ginit, gtest, gx, gy, fx, fy, stx, sty, stmin, stmax, width, width1;

        if (task == START)
        {
            if (stp < stpmin) { task = ERROR; taskMsg = ERROR_STP1; }
            if (stp > stpmax) { task = ERROR; taskMsg = ERROR_STP2; }
            if (g >= 0.0) { task = ERROR; taskMsg = ERROR_INITG; }
            if (ftol < 0.0) { task = ERROR; taskMsg = ERROR_FTOL; }
            if (gtol < 0.0) { task = ERROR; taskMsg = ERROR_GTOL; }
            if (xtol < 0.0) { task = ERROR; taskMsg = ERROR_XTOL; }
            if (stpmin < 0.0) { task = ERROR; taskMsg = ERROR_STPMIN; }
            if (stpmax < stpmin) { task = ERROR; taskMsg = ERROR_STPMAX; }
            if (task == ERROR) return;

            brackt = 0; stage = 1; finit = f; ginit = g; gtest = ftol * ginit;
            width = stpmax - stpmin; width1 = width / 0.5;
            stx = 0.0; fx = finit; gx = ginit; sty = 0.0; fy = finit; gy = ginit;
            stmin = 0.0; stmax = stp + xtrapu * stp;
            task = FG; taskMsg = NO_MSG;
            goto SAVEVARS;
        }
        else
        {
            brackt = isave[0] == 1 ? 1 : 0;
            stage = isave[1];
            ginit = dsave[0]; gtest = dsave[1]; gx = dsave[2]; gy = dsave[3];
            finit = dsave[4]; fx = dsave[5]; fy = dsave[6]; stx = dsave[7]; sty = dsave[8];
            stmin = dsave[9]; stmax = dsave[10]; width = dsave[11]; width1 = dsave[12];
        }

        ftest = finit + stp * gtest;
        if (stage == 1 && f <= ftest && g >= 0.0) stage = 2;
        if (brackt != 0 && (stp <= stmin || stp >= stmax)) { task = WARNING; taskMsg = WARN_ROUND; }
        if (brackt != 0 && stmax - stmin <= xtol * stmax) { task = WARNING; taskMsg = WARN_XTOL; }
        if (stp == stpmax && f <= ftest && g <= gtest) { task = WARNING; taskMsg = WARN_STPMAX; }
        if (stp == stpmin && (f > ftest || g >= gtest)) { task = WARNING; taskMsg = WARN_STPMIN; }
        if (f <= ftest && Math.Abs(g) <= gtol * (-ginit)) task = CONVERGENCE;
        if (task == WARNING || task == CONVERGENCE) goto SAVEVARS;

        if (stage == 1 && f <= fx && f > ftest)
        {
            double fm = f - stp * gtest;
            double fxm = fx - stx * gtest;
            double fym = fy - sty * gtest;
            double gm = g - gtest;
            double gxm = gx - gtest;
            double gym = gy - gtest;
            Dcstep(ref stx, ref fxm, ref gxm, ref sty, ref fym, ref gym, ref stp, fm, gm, ref brackt, stmin, stmax);
            fx = fxm + stx * gtest; fy = fym + sty * gtest;
            gx = gxm + gtest; gy = gym + gtest;
        }
        else
        {
            Dcstep(ref stx, ref fx, ref gx, ref sty, ref fy, ref gy, ref stp, f, g, ref brackt, stmin, stmax);
        }

        if (brackt != 0)
        {
            if (Math.Abs(sty - stx) >= 0.66 * width1) stp = stx + 0.5 * (sty - stx);
            width1 = width;
            width = Math.Abs(sty - stx);
        }
        if (brackt != 0) { stmin = Math.Min(stx, sty); stmax = Math.Max(stx, sty); }
        else { stmin = stp + xtrapl * (stp - stx); stmax = stp + xtrapu * (stp - stx); }
        stp = Math.Max(stp, stmin);
        stp = Math.Min(stp, stmax);
        if ((brackt != 0 && (stp <= stmin || stp >= stmax)) ||
            (brackt != 0 && stmax - stmin <= xtol * stmax))
            stp = stx;
        task = FG; taskMsg = NO_MSG;

    SAVEVARS:
        isave[0] = brackt != 0 ? 1 : 0;
        isave[1] = stage;
        dsave[0] = ginit; dsave[1] = gtest; dsave[2] = gx; dsave[3] = gy;
        dsave[4] = finit; dsave[5] = fx; dsave[6] = fy; dsave[7] = stx; dsave[8] = sty;
        dsave[9] = stmin; dsave[10] = stmax; dsave[11] = width; dsave[12] = width1;
    }

    private static void Dcstep(ref double stx, ref double fx, ref double dx, ref double sty,
        ref double fy, ref double dy, ref double stp, double fp, double dp, ref int brackt,
        double stpmin, double stpmax)
    {
        double sgnd = dp * (dx / Math.Abs(dx));
        double gamma, p, q, r, s, stpc, stpf, stpq, theta;

        if (fp > fx)
        {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = Math.Max(Math.Abs(theta), Math.Max(Math.Abs(dx), Math.Abs(dp)));
            gamma = s * Math.Sqrt(Math.Pow(theta / s, 2.0) - (dx / s) * (dp / s));
            if (stp < stx) gamma = -gamma;
            p = (gamma - dx) + theta;
            q = ((gamma - dx) + gamma) + dp;
            r = p / q;
            stpc = stx + r * (stp - stx);
            stpq = stx + ((dx / ((fx - fp) / (stp - stx) + dx)) / 2.0) * (stp - stx);
            stpf = Math.Abs(stpc - stx) < Math.Abs(stpq - stx) ? stpc : stpc + (stpq - stpc) / 2.0;
            brackt = 1;
        }
        else if (sgnd < 0.0)
        {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = Math.Max(Math.Abs(theta), Math.Max(Math.Abs(dx), Math.Abs(dp)));
            gamma = s * Math.Sqrt(Math.Pow(theta / s, 2.0) - (dx / s) * (dp / s));
            if (stp > stx) gamma = -gamma;
            p = (gamma - dp) + theta;
            q = ((gamma - dp) + gamma) + dx;
            r = p / q;
            stpc = stp + r * (stx - stp);
            stpq = stp + (dp / (dp - dx)) * (stx - stp);
            stpf = Math.Abs(stpc - stp) > Math.Abs(stpq - stp) ? stpc : stpq;
            brackt = 1;
        }
        else if (Math.Abs(dp) < Math.Abs(dx))
        {
            theta = 3.0 * (fx - fp) / (stp - stx) + dx + dp;
            s = Math.Max(Math.Abs(theta), Math.Max(Math.Abs(dx), Math.Abs(dp)));
            gamma = s * Math.Sqrt(Math.Max(0.0, Math.Pow(theta / s, 2.0) - (dx / s) * (dp / s)));
            if (stp > stx) gamma = -gamma;
            p = (gamma - dp) + theta;
            q = (gamma + (dx - dp)) + gamma;
            r = p / q;
            if (r < 0.0 && gamma != 0.0) stpc = stp + r * (stx - stp);
            else if (stp > stx) stpc = stpmax;
            else stpc = stpmin;
            stpq = stp + (dp / (dp - dx)) * (stx - stp);
            if (brackt != 0)
            {
                stpf = Math.Abs(stpc - stp) < Math.Abs(stpq - stp) ? stpc : stpq;
                if (stp > stx) stpf = Math.Min(stp + 0.66 * (sty - stp), stpf);
                else stpf = Math.Max(stp + 0.66 * (sty - stp), stpf);
            }
            else
            {
                stpf = Math.Abs(stpc - stp) > Math.Abs(stpq - stp) ? stpc : stpq;
                stpf = Math.Min(stpmax, stpf);
                stpf = Math.Max(stpmin, stpf);
            }
        }
        else
        {
            if (brackt != 0)
            {
                theta = 3.0 * (fp - fy) / (sty - stp) + dy + dp;
                s = Math.Max(Math.Abs(theta), Math.Max(Math.Abs(dy), Math.Abs(dp)));
                gamma = s * Math.Sqrt(Math.Pow(theta / s, 2.0) - (dy / s) * (dp / s));
                if (stp > sty) gamma = -gamma;
                p = (gamma - dp) + theta;
                q = ((gamma - dp) + gamma) + dy;
                r = p / q;
                stpc = stp + r * (sty - stp);
                stpf = stpc;
            }
            else if (stp > stx) stpf = stpmax;
            else stpf = stpmin;
        }

        if (fp > fx) { sty = stp; fy = fp; dy = dp; }
        else
        {
            if (sgnd < 0.0) { sty = stx; fy = fx; dy = dx; }
            stx = stp; fx = fp; dx = dp;
        }
        stp = stpf;
    }
}
