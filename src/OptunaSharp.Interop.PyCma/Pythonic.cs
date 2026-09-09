using Python.Runtime;
using OptunaSharp.Interop.PythonNet;

namespace OptunaSharp.Interop.PyCma;

/// <summary>
///     The <b>pythonic</b> surface for the PyCma bridge — a single root <c>static class</c>
///     <c>Python</c> whose nested subclasses form a tree that reaches the pycma calls, so the consuming
///     C# reads for what it is: <c>Python.cma.CMAEvolutionStrategy.New(x0, sigma0, opts)</c> constructs
///     the real <c>cma.CMAEvolutionStrategy</c>, and <c>Python.cma.CMAEvolutionStrategy.ask(es)</c> /
///     <c>.tell(es, xs, ys)</c> / <c>.popsize_t(es)</c> are its object-protocol methods. It is always
///     spelled <c>Python.*</c> (never a <c>using</c> of a namespace), and the tree — not extension
///     members — is how the <c>CMAEvolutionStrategy</c> object's methods/attributes are exposed
///     (mirroring how Python spells <c>cma.CMAEvolutionStrategy(...).ask()</c>).
///     <para>
///     This is a typed FACADE (one C# member per pycma call), not a source-snippet runner: every call
///     here is a single fixed pycma call known at compile time. The array DATA (x0, bounds, the ask/tell
///     solution matrices) crosses zero-copy through NumSharp's interop (<see cref="PyCmaInterop" />),
///     never element by element.
///     </para>
///     <para>Every member calls the C-API and MUST run under the GIL (from a
///     <see cref="PythonRuntime.Run(System.Action)" /> body). A returned <see cref="PyObject" /> is
///     owned by the caller (dispose it); the cached module handle is process-scoped and must NOT be
///     disposed.</para>
/// </summary>
internal static class Python
{
    /// <summary>
    ///     <c>import cma</c> — the pycma surface (Nikolaus Hansen's <c>cma</c> package), reached as
    ///     <c>Python.cma.*</c>. Only the fixed calls the bridge makes live here:
    ///     <see cref="BoundTransform" /> (the boundary-handler class object put into the options dict)
    ///     and the nested <see cref="CMAEvolutionStrategy" /> (the optimizer's factory + protocol
    ///     methods).
    /// </summary>
    internal static class cma
    {
        /// <summary>The cached, process-scoped <c>cma</c> module handle — never dispose it (§3).</summary>
        private static PyObject Module => PythonRuntime.Import("cma");

        /// <summary>Ensure pycma imports (cma.py's <c>_imports.check()</c>) — fail fast if it is missing.
        /// Under the GIL.</summary>
        internal static void check() => _ = PythonRuntime.Import("cma");

        /// <summary>
        ///     <c>cma.BoundTransform</c> — the boundary-handler <b>class object</b> (not an instance) that
        ///     PyCmaSampler puts into <c>cma_opts["BoundaryHandler"]</c>. A new reference to the module
        ///     attribute; dispose it. Under the GIL.
        /// </summary>
        internal static PyObject BoundTransform => Module.GetAttr("BoundTransform");

        /// <summary>
        ///     <c>cma.CMAEvolutionStrategy</c> — the optimizer object. <see cref="New" /> is the
        ///     constructor factory (<c>cma.CMAEvolutionStrategy(x0, sigma0, opts)</c>); the remaining
        ///     members are its object-protocol methods, exposed as a nested class (NOT extensions) and
        ///     called unbound with the instance passed in, mirroring Python. The plain members return a
        ///     <see cref="PyObject" /> (dispose it); <c>popsize_t</c> returns the final CLR value the C#
        ///     replay logic needs. Under the GIL.
        /// </summary>
        internal static class CMAEvolutionStrategy
        {
            /// <summary><c>cma.CMAEvolutionStrategy(x0, sigma0, inopts=opts)</c> — construct the optimizer.
            /// Dispose the returned instance (under the GIL) when done.</summary>
            internal static PyObject New(PyObject x0, double sigma0, PyObject opts)
            {
                using PyObject ctor = Module.GetAttr("CMAEvolutionStrategy");
                using PyObject sigma = new PyFloat(sigma0);
                return ctor.Invoke(x0, sigma, opts);
            }

            /// <summary><c>es.ask()</c> — a fresh generation of candidate solutions (a Python list of numpy
            /// arrays). Dispose it.</summary>
            internal static PyObject ask(PyObject es) => es.InvokeMethod("ask");

            /// <summary><c>es.tell(solutions, function_values)</c> — update the distribution from the given
            /// solutions and their objective values (returns <c>None</c>). Under the GIL.</summary>
            internal static void tell(PyObject es, PyObject solutions, PyObject functionValues)
            {
                using PyObject _ = es.InvokeMethod("tell", solutions, functionValues);
            }

            /// <summary><c>es.popsize</c> as a CLR <see cref="long" /> — the population size pycma resolved
            /// (the default <c>4 + floor(3 ln N)</c>, or the user's <c>popsize</c> option).</summary>
            internal static long popsize_t(PyObject es)
            {
                using PyObject p = es.GetAttr("popsize");
                return p.As<long>();
            }
        }
    }
}
