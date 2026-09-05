using System.Runtime.CompilerServices;

// Friend assemblies. Every name carries PublicKey= because OptunaSharp is strong-named (see
// Directory.Build.props at the repo root): the compiler rejects a keyless InternalsVisibleTo from a
// signed assembly with CS1726.
//
// The key is Microsoft's PUBLISHED open-source key (token cc7b13ffcd2ddd51), shared with the rest of
// the SciSharp STACK. Its private half is public by design, so this list is NOT an access control —
// anyone can build an assembly with a matching name and key. It is a compile-time convenience.
//
// The literal lives once, in OptunaSharpFriendKey.PublicKey below; `"name, PublicKey=" + const` is a
// compile-time constant, which is what an attribute argument requires. Global attributes must precede
// type declarations, so the holder class follows them.

[assembly: InternalsVisibleTo("OptunaSharp.Tests, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The Python.NET interop suite exercises the same internals while comparing against real Optuna.
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Interop, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// `dotnet run` file-based scripts that override AssemblyName to this (see the repo README ->
// "Scripting with dotnet run"). A script UNDER the repo root inherits Directory.Build.props and is
// signed automatically; one run from elsewhere must pass the signing properties itself.
[assembly: InternalsVisibleTo("OptunaSharp.DotNetRunScript, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The Redis journal backends (satellite packages, one per Redis client) serialize logs through the
// same internal PyJson (so the on-wire bytes stay byte-compatible with Python Optuna) and raise the
// experimental-class warning via OptunaWarnings/WarningCategory — both internal to Core.
[assembly: InternalsVisibleTo("OptunaSharp.Storage.StackExchange.Redis, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
[assembly: InternalsVisibleTo("OptunaSharp.Storage.NRedisStack, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
[assembly: InternalsVisibleTo("OptunaSharp.Storage.ServiceStack.Redis, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The SQLite RDB storage (satellite package OptunaSharp.Storage.Sqlite, Phase 4.1) is an external
// implementer of the ICachedStorageBackend seam (declared in CachedStorage.cs). Like InMemoryStorage
// it must construct FrozenTrial/FrozenStudy with their internal identity fields set
// (FrozenTrial.TrialId / SetValues, FrozenStudy.StudyId), serialize attr value_json through the same
// internal PyJson (byte-compatible with Python Optuna), and map the trial/direction enums to their
// stored names via the internal PythonName() helpers.
[assembly: InternalsVisibleTo("OptunaSharp.Storage.Sqlite, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The gRPC storage proxy (satellite package OptunaSharp.Storage.Grpc, Phase 4.4) is a network client +
// server for any BaseStorage. Both sides serialize/deserialize FrozenTrial through the same internal
// helpers so the on-wire proto strings stay byte-compatible with Python Optuna: PyJson (attr json.dumps),
// Distribution JSON, FrozenTrial.TrialId (the proto trial_id + the client cache's finished/unfinished
// bookkeeping), and the server's two info logs + the experimental-class/func warning via OptunaLogging /
// OptunaWarnings / WarningCategory — all internal to Core.
[assembly: InternalsVisibleTo("OptunaSharp.Storage.Grpc, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The ML.NET importance backends (satellite package OptunaSharp.Integration.ML.NET, Phase 5.1) reuse
// Core's internal forest machinery: SklearnLikeTree / FanovaTree / Fanova / Mdi / ForestImportance
// (the shared algorithm + the ForestFit delegate), plus the linked MlNetTreeEnsembleAdapter,
// SearchSpaceTransform, the optuna/importance/_base.py helpers in ImportanceBase, the intersection
// search space, and the ValueError/OptunaWarnings seam. The one package hosts both the faithful
// FastForest backend and the EXPERIMENTAL / off-label FastTree/FastTreeTweedie gradient boosters
// (boosting is not part of Optuna's importance). (Core owns the OptunaSharp.Importance.Forests /
// .Booster namespaces; the ML.NET backends are a subordinate, optional package — hence the .ML.NET
// assembly-name suffix. LightGBM keeps its own package, OptunaSharp.Integration.LightGbm, below.)
[assembly: InternalsVisibleTo("OptunaSharp.Integration.ML.NET, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The forests test project reaches the internal fANOVA tree machinery (SklearnLikeTree, FanovaTree,
// the native CART port) directly — the port of Optuna's test_tree.py asserts against the tree
// algorithm with no forest involved, and the native-forest tests reach CartRandomForest.
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Importance.Forests, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
// The booster test project mirrors the forests test project (backend-parameterised invariants).
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Importance.Booster, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The LightGBM integration (satellite package OptunaSharp.Integration.LightGbm) hosts an EXPERIMENTAL /
// off-label LightGBM booster-importance backend (Phase 5.1) — and, later, the LightGBM tuner (Phase
// 5.7). Its importance backend reuses the same Core-internal forest machinery as the other ML.NET
// backends: SklearnLikeTree / ForestImportance + the linked MlNetTreeEnsembleAdapter, over the
// SearchSpaceTransform / ImportanceBase / ValueError seam.
[assembly: InternalsVisibleTo("OptunaSharp.Integration.LightGbm, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
// Its test project mirrors the forests test project (backend-parameterised invariants).
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Integration.LightGbm, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The Redis backends' shared test project asserts that experimental warning fired via the internal
// OptunaWarnings.WarningRaised event (same reason the interop suite is a friend).
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Storage, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

// The CLI (satellite package OptunaSharp.Cli, Phase 5.3 — the port of optuna/cli.py) reproduces the
// exact stdout of the `optuna` command, so it renders through the same two internal Python-fidelity
// helpers Optuna's output goes through: PyJson (json.dumps — the value/json formats) and PythonRepr
// (str()/repr() of floats/tuples/dicts — the value/table formats). Everything else the CLI needs
// (records/columns, enum names, multi-objective, distribution JSON) it builds over Core's public API.
[assembly: InternalsVisibleTo("OptunaSharp.Cli, PublicKey=" + OptunaSharpFriendKey.PublicKey)]
// Its test project asserts stdout parity against the same internals (and against live optuna's CLI).
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Cli, PublicKey=" + OptunaSharpFriendKey.PublicKey)]

/// <summary>
///     The public half of the strong-name key OptunaSharp is signed with, as the hex literal
///     <see cref="InternalsVisibleToAttribute" /> expects.
/// </summary>
/// <remarks>
///     Mirrors the <c>OptunaSharpPublicKey</c> property in the repo-root <c>Directory.Build.props</c>
///     and the key asserted by <c>.github/scripts/verify_strong_name.cs</c>. It is Microsoft's
///     published open-source key (token <c>cc7b13ffcd2ddd51</c>), shared across the SciSharp STACK.
/// </remarks>
internal static class OptunaSharpFriendKey
{
    internal const string PublicKey =
        "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
        "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
        "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
        "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";
}
