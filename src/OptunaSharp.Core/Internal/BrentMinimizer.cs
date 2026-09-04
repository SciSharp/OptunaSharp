using System;

namespace OptunaSharp.Internal;

/// <summary>
///     Brent's method scalar minimizer — a faithful port of the <c>Brent</c> class driven by
///     <c>scipy.optimize.minimize_scalar(method="brent")</c> (<c>scipy/optimize/_optimize.py</c>).
///     Only the three-point-bracket entry is ported: the GP acqf optimizer's
///     <c>_discrete_line_search</c> always calls it with an explicit <c>(xa, xb, xc)</c> bracket, so
///     the <c>bracket()</c> auto-bracketing routine is not needed. Structure (constants
///     <c>_mintol = 1e-11</c>, <c>_cg = 0.3819660</c>, the parabolic/golden-section step selection) is
///     kept line-for-line so the C# can be diffed against scipy. Numeric drift is ULP-level (the same
///     BLAS-free arithmetic scipy uses here), consistent with the GP path's tolerance boundary.
/// </summary>
internal static class BrentMinimizer
{
    private const double MinTol = 1.0e-11;
    private const double Cg = 0.3819660;

    /// <summary>
    ///     Minimize <paramref name="func" /> over the bracket <c>(xa, xb, xc)</c> — the port of
    ///     <c>Brent.get_bracket_info</c> (three-sequence branch) + <c>Brent.optimize</c>. Returns the
    ///     minimizer <c>xmin</c> (scipy's <c>OptimizeResult.x</c>). <paramref name="tol" /> is scipy's
    ///     <c>xtol</c> (the <c>tol=</c> kwarg of <c>minimize_scalar</c>).
    /// </summary>
    internal static double MinimizeScalar(
        Func<double, double> func, double xa, double xb, double xc, double tol, int maxiter = 500)
    {
        // ---- get_bracket_info (len(brack) == 3) --------------------------------------------------
        if (xa > xc)
            (xa, xc) = (xc, xa);  // swap so xa < xc can be assumed
        if (!(xa < xb && xb < xc))
            throw new ArgumentException(
                "Bracketing values (xa, xb, xc) do not fulfill this requirement: (xa < xb) and (xb < xc)");
        double fa = func(xa);
        double fb = func(xb);
        double fc = func(xc);
        if (!(fb < fa && fb < fc))
            throw new ArgumentException(
                "Bracketing values (xa, xb, xc) do not fulfill this requirement: "
                + "(f(xb) < f(xa)) and (f(xb) < f(xc))");

        // ---- optimize (CORE ALGORITHM) -----------------------------------------------------------
        double x = xb, w = xb, v = xb;
        double fw = fb, fv = fb, fx = fb;
        double a, b;
        if (xa < xc) { a = xa; b = xc; }
        else { a = xc; b = xa; }
        double deltax = 0.0;
        double rat = 0.0;  // See scipy's gh-4140 note; set on the first (golden) iteration regardless.
        int iter = 0;

        while (iter < maxiter)
        {
            double tol1 = tol * Math.Abs(x) + MinTol;
            double tol2 = 2.0 * tol1;
            double xmid = 0.5 * (a + b);
            // check for convergence
            if (Math.Abs(x - xmid) < tol2 - 0.5 * (b - a))
                break;

            if (Math.Abs(deltax) <= tol1)
            {
                deltax = x >= xmid ? a - x : b - x;  // do a golden section step
                rat = Cg * deltax;
            }
            else
            {
                double tmp1 = (x - w) * (fx - fv);
                double tmp2 = (x - v) * (fx - fw);
                double p = (x - v) * tmp2 - (x - w) * tmp1;
                tmp2 = 2.0 * (tmp2 - tmp1);
                if (tmp2 > 0.0)
                    p = -p;
                tmp2 = Math.Abs(tmp2);
                double dxTemp = deltax;
                deltax = rat;
                // check parabolic fit
                if (p > tmp2 * (a - x) && p < tmp2 * (b - x) && Math.Abs(p) < Math.Abs(0.5 * tmp2 * dxTemp))
                {
                    rat = p * 1.0 / tmp2;  // if parabolic step is useful.
                    double uu = x + rat;
                    if (uu - a < tol2 || b - uu < tol2)
                        rat = xmid - x >= 0 ? tol1 : -tol1;
                }
                else
                {
                    deltax = x >= xmid ? a - x : b - x;  // if it's not, do a golden section step
                    rat = Cg * deltax;
                }
            }

            double u;
            if (Math.Abs(rat) < tol1)  // update by at least tol1
                u = rat >= 0 ? x + tol1 : x - tol1;
            else
                u = x + rat;

            double fu = func(u);  // calculate new output value

            if (fu > fx)  // if it's bigger than current
            {
                if (u < x) a = u;
                else b = u;
                if (fu <= fw || w == x)
                {
                    v = w; w = u; fv = fw; fw = fu;
                }
                else if (fu <= fv || v == x || v == w)
                {
                    v = u; fv = fu;
                }
            }
            else
            {
                if (u >= x) a = x;
                else b = x;
                v = w; w = x; x = u;
                fv = fw; fw = fx; fx = fu;
            }

            iter++;
        }

        return x;  // self.xmin
    }
}
