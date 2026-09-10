namespace OptunaSharp.Samplers.AgeMoea;

/// <summary>
///     Pareto-dominance relation — port of <c>pymoo.util.dominator.Dominator</c> (<c>@0.6.1.3</c>).
///     Objectives are minimized. With constraint violations supplied, feasibility dominates
///     (<c>cva &lt; cvb ⇒ a</c> wins) before objectives are compared.
/// </summary>
internal static class Dominator
{
    /// <summary>
    ///     <c>get_relation(a, b, cva, cvb)</c>: returns <c>+1</c> if <paramref name="a" /> dominates
    ///     <paramref name="b" />, <c>-1</c> if dominated, <c>0</c> if non-dominated / equal.
    /// </summary>
    public static int GetRelation(double[] a, double[] b, double? cva = null, double? cvb = null)
    {
        if (cva is not null && cvb is not null)
        {
            if (cva < cvb) return 1;
            if (cvb < cva) return -1;
        }

        int val = 0;
        for (int i = 0; i < a.Length; i++)
        {
            if (a[i] < b[i])
            {
                if (val == -1) return 0;
                val = 1;
            }
            else if (b[i] < a[i])
            {
                if (val == 1) return 0;
                val = -1;
            }
        }
        return val;
    }
}
