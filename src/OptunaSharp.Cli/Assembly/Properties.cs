using System.Runtime.CompilerServices;

// The CLI test project drives the CLI in-process (capturing stdout) and unit-tests the internal
// output-formatting helpers (OutputFormat / Records / PyTimedelta) directly. Strong-named, so
// PublicKey= is required (Microsoft's published open-source key, token cc7b13ffcd2ddd51 — shared
// across the SciSharp STACK; not an access control).
[assembly: InternalsVisibleTo("OptunaSharp.Tests.Cli, PublicKey=" + OptunaSharpCliFriendKey.PublicKey)]

internal static class OptunaSharpCliFriendKey
{
    internal const string PublicKey =
        "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
        "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
        "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
        "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb";
}
