using System.Runtime.CompilerServices;

// The test suite exercises the internal R2 endpoint-configuration helper (BuildR2Config), so the
// exact R2 endpoint/region/path-style is pinned by a test. This assembly is strong-named (repo-wide
// Directory.Build.props), so the PublicKey= is mandatory (a keyless InternalsVisibleTo is CS1726).
// The key is the shared SciSharp open key (token cc7b13ffcd2ddd51), same as OptunaSharp Core's.
[assembly: InternalsVisibleTo(
    "OptunaSharp.Tests.Artifacts, PublicKey=" +
    "00240000048000009400000006020000002400005253413100040000010001004b86c4cb78549b34" +
    "bab61a3b1800e23bfeb5b3ec390074041536a7e3cbd97f5f04cf0f857155a8928eaa29ebfd11cfbb" +
    "ad3ba70efea7bda3226c6a8d370a4cd303f714486b6ebc225985a638471e6ef571cc92a4613c00b8" +
    "fa65d61ccee0cbe5f36330c9a01f4183559f1bef24cc2917c6d913e3a541333a1d05d9bed22b38cb")]
