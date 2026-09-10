using System;
using Python.Runtime;
using OptunaSharp.Interop.PythonNet;

namespace OptunaSharp.Interop.Pymoo;

/// <summary>
///     The <b>pythonic</b> surface for the pymoo bridge — a single root <c>static class</c>
///     <c>Python</c> whose nested subclasses form a tree that reaches the pymoo calls, so the consuming
///     C# reads for what it is: <c>Python.pymoo.CreateAlgorithm(module, cls, kwargs)</c> constructs a
///     real <c>pymoo.algorithms.moo.*</c> optimizer, <c>Python.pymoo.Problem(...)</c> builds the bare
///     <c>pymoo.core.problem.Problem</c> whose box the operators sample, and
///     <c>Python.pymoo.Algorithm.ask(algo)</c> / <c>.tell(algo, pop)</c> / <c>.pop_size_t(algo)</c> and
///     <c>Python.pymoo.Population.get(pop, "X")</c> / <c>.set(pop, "F", f)</c> are the object-protocol
///     methods of the returned <c>Algorithm</c> / <c>Population</c>. It is always spelled
///     <c>Python.*</c> (never a <c>using</c> of a namespace), and the tree — not extension members — is
///     how those objects' methods/attributes are exposed (mirroring how pymoo spells
///     <c>algorithm.ask()</c> / <c>pop.get("X")</c>).
///     <para>
///     This is a typed FACADE (one C# member per pymoo call): every call here is fixed and known at
///     compile time. The algorithm CLASS is chosen at runtime from a closed <see cref="PymooAlgorithm" />
///     set, so its (module, class) pair is resolved in C# (<see cref="PymooAlgorithmMap" />) and passed
///     to <see cref="CreateAlgorithm" /> — a fixed constructor family, not an arbitrary user class. The
///     array DATA (the problem box, the reference directions, the ask/tell X/F matrices) crosses
///     zero-copy through NumSharp's interop (<see cref="PymooInterop" />), never element by element.
///     </para>
///     <para>Every member calls the C-API and MUST run under the GIL (from a
///     <see cref="PythonRuntime.Run(System.Action)" /> body). A returned <see cref="PyObject" /> is owned
///     by the caller (dispose it); the cached module handles are process-scoped and must NOT be
///     disposed.</para>
/// </summary>
internal static class Python
{
    /// <summary>
    ///     <c>import pymoo</c> and the handful of fixed pymoo calls the bridge makes, reached as
    ///     <c>Python.pymoo.*</c>: the import guard (<see cref="check" />), the bare
    ///     <see cref="Problem" /> factory, the <see cref="get_reference_directions" /> factory, the
    ///     <see cref="CreateAlgorithm" /> constructor dispatch, and the nested <see cref="Algorithm" /> /
    ///     <see cref="Population" /> object protocols.
    /// </summary>
    internal static class pymoo
    {
        /// <summary>Ensure pymoo imports — fail fast (via <c>PythonRuntime.Import</c>'s helpful message)
        /// if it is missing. Under the GIL.</summary>
        internal static void check() => _ = PythonRuntime.Import("pymoo");

        /// <summary>
        ///     <c>pymoo.core.problem.Problem(n_var=…, n_obj=…, xl=…, xu=…)</c> — a bare problem with no
        ///     <c>_evaluate</c>. The bridge never asks pymoo to evaluate (objective values come from the
        ///     study's completed trials, set onto each population's <c>F</c>), so only the box
        ///     (<paramref name="nVar" />/<paramref name="nObj" />/<paramref name="xl" />/<paramref name="xu" />)
        ///     the operators sample and the survival reads is needed. Dispose the returned problem.
        ///     Under the GIL.
        /// </summary>
        internal static PyObject Problem(int nVar, int nObj, int nIeqConstr, PyObject xl, PyObject xu)
        {
            PyObject module = PythonRuntime.Import("pymoo.core.problem");
            using PyObject ctor = module.GetAttr("Problem");
            using var kwargs = new PyDict();
            using (PyObject nv = new PyInt(nVar)) kwargs["n_var"] = nv;
            using (PyObject no = new PyInt(nObj)) kwargs["n_obj"] = no;
            using (PyObject nc = new PyInt(nIeqConstr)) kwargs["n_ieq_constr"] = nc;
            kwargs["xl"] = xl;
            kwargs["xu"] = xu;
            return ctor.Invoke(Array.Empty<PyObject>(), kwargs);
        }

        /// <summary><c>pymoo.operators.crossover.sbx.SBX(prob=…, eta=…)</c> — the simulated-binary
        /// crossover operator. A <c>null</c> argument is omitted so pymoo's own default fills in. Dispose
        /// the returned operator. Under the GIL.</summary>
        internal static PyObject SBX(double? eta, double? prob)
        {
            PyObject module = PythonRuntime.Import("pymoo.operators.crossover.sbx");
            using PyObject ctor = module.GetAttr("SBX");
            using var kwargs = new PyDict();
            if (eta is { } etaValue)
                using (PyObject e = etaValue.ToPython()) kwargs["eta"] = e;
            if (prob is { } probValue)
                using (PyObject p = probValue.ToPython()) kwargs["prob"] = p;
            return ctor.Invoke(Array.Empty<PyObject>(), kwargs);
        }

        /// <summary><c>pymoo.operators.mutation.pm.PM(prob=…, eta=…)</c> — the polynomial mutation
        /// operator. A <c>null</c> argument is omitted so pymoo's own default fills in. Dispose the
        /// returned operator. Under the GIL.</summary>
        internal static PyObject PM(double? eta, double? prob)
        {
            PyObject module = PythonRuntime.Import("pymoo.operators.mutation.pm");
            using PyObject ctor = module.GetAttr("PM");
            using var kwargs = new PyDict();
            if (eta is { } etaValue)
                using (PyObject e = etaValue.ToPython()) kwargs["eta"] = e;
            if (prob is { } probValue)
                using (PyObject p = probValue.ToPython()) kwargs["prob"] = p;
            return ctor.Invoke(Array.Empty<PyObject>(), kwargs);
        }

        /// <summary>
        ///     <c>pymoo.util.ref_dirs.get_reference_directions(name, n_dim, *extra, **kwargs)</c> — the
        ///     reference-direction factory for the reference-direction algorithms. <paramref name="extra" />
        ///     is an optional extra positional (the Riesz-energy <c>n_points</c>); the structured
        ///     <c>n_partitions</c> (Das–Dennis) and any <c>seed</c> ride in <paramref name="kwargs" />.
        ///     Dispose the returned array. Under the GIL.
        /// </summary>
        internal static PyObject get_reference_directions(string name, int nDim, PyDict kwargs, PyObject? extra = null)
        {
            PyObject module = PythonRuntime.Import("pymoo.util.ref_dirs");
            using PyObject fn = module.GetAttr("get_reference_directions");
            using PyObject namePy = new PyString(name);
            using PyObject nDimPy = new PyInt(nDim);
            PyObject[] args = extra is null ? new[] { namePy, nDimPy } : new[] { namePy, nDimPy, extra };
            return fn.Invoke(args, kwargs);
        }

        /// <summary>
        ///     Construct a pymoo algorithm by its (module, class) — e.g.
        ///     <c>pymoo.algorithms.moo.nsga2.NSGA2(**kwargs)</c>. The class is one of the closed
        ///     <see cref="PymooAlgorithm" /> set resolved by <see cref="PymooAlgorithmMap" />; the family's
        ///     keyword arguments (<c>pop_size</c> / <c>ref_dirs</c> / <c>ref_points</c> / …) are built by
        ///     the caller. Dispose the returned algorithm. Under the GIL.
        /// </summary>
        internal static PyObject CreateAlgorithm(string moduleName, string className, PyDict kwargs)
        {
            PyObject module = PythonRuntime.Import(moduleName);
            using PyObject ctor = module.GetAttr(className);
            return ctor.Invoke(Array.Empty<PyObject>(), kwargs);
        }

        /// <summary>
        ///     <c>pymoo.core.algorithm.Algorithm</c> — the optimizer object. Its ask/tell protocol methods
        ///     are exposed as a nested class (NOT extensions), called unbound with the instance passed in,
        ///     mirroring Python. Under the GIL.
        /// </summary>
        internal static class Algorithm
        {
            /// <summary>
            ///     <c>algorithm.setup(problem, termination=("n_gen", &lt;huge&gt;), seed=seed,
            ///     verbose=False)</c> — bind the problem and seed the global RNG. The termination is set to
            ///     a practically-unreachable generation count: the bridge drives ask/tell by hand (one
            ///     population per generation, replayed from storage) and never consults
            ///     <c>has_next()</c>, so termination must never trip.
            /// </summary>
            internal static void setup(PyObject algo, PyObject problem, int seed)
            {
                using var kwargs = new PyDict();
                using (PyObject key = new PyString("n_gen"))
                using (PyObject big = new PyInt(1_000_000_000L))
                using (PyObject termination = new PyTuple(new[] { key, big }))
                    kwargs["termination"] = termination;
                using (PyObject seedPy = new PyInt(seed))
                    kwargs["seed"] = seedPy;
                using (PyObject verbose = false.ToPython())
                    kwargs["verbose"] = verbose;
                using PyObject _ = algo.InvokeMethod("setup", new[] { problem }, kwargs);
            }

            /// <summary><c>algorithm.pop_size</c> as a CLR <see cref="long" /> — the population size pymoo
            /// resolved (from <c>pop_size</c>, or derived from the reference directions/points).</summary>
            internal static long pop_size_t(PyObject algo)
            {
                using PyObject p = algo.GetAttr("pop_size");
                return p.As<long>();
            }

            /// <summary><c>algorithm.ask()</c> — the next infill <c>Population</c> (a full generation of
            /// <c>pop_size</c> individuals for every algorithm this bridge exposes). Dispose it.</summary>
            internal static PyObject ask(PyObject algo) => algo.InvokeMethod("ask");

            /// <summary><c>algorithm.tell(infills=pop)</c> — advance the algorithm with the given
            /// population (whose <c>F</c> the caller has set from the study's objective values). Returns
            /// <c>None</c>. Under the GIL.</summary>
            internal static void tell(PyObject algo, PyObject infills)
            {
                using PyObject _ = algo.InvokeMethod("tell", infills);
            }
        }

        /// <summary>
        ///     <c>pymoo.core.population.Population</c> — the individual container. Its <c>get</c>/<c>set</c>
        ///     protocol is exposed as a nested class (NOT extensions), mirroring
        ///     <c>pop.get("X")</c> / <c>pop.set("F", …)</c>. Under the GIL.
        /// </summary>
        internal static class Population
        {
            /// <summary><c>population.get(key)</c> — e.g. the stacked design matrix <c>get("X")</c>
            /// (a numpy <c>(pop_size, n_var)</c> array). Dispose the returned object.</summary>
            internal static PyObject get(PyObject pop, string key)
            {
                using PyObject k = new PyString(key);
                return pop.InvokeMethod("get", k);
            }

            /// <summary><c>population.set(key, values)</c> — e.g. <c>set("F", f)</c> to stamp the objective
            /// matrix onto the individuals before <c>tell</c>. Returns the population (discarded).
            /// Under the GIL.</summary>
            internal static void set(PyObject pop, string key, PyObject values)
            {
                using PyObject k = new PyString(key);
                using PyObject _ = pop.InvokeMethod("set", k, values);
            }
        }
    }
}
