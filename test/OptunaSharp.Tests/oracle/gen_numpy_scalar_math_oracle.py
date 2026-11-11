"""Record NumPy's float64 log1p / expm1 / logaddexp over a fixed input set, as raw bit patterns.

OptunaSharp's MathUtil.Log1p / Expm1 / LogAddExp delegate to NumSharp's public models of the UCRT routines NumPy calls
on Windows (NumSharp.Utilities.NDLog1pExpm1Math / NDLogAddExpMath). Internal/MathUtilNumSharpModelTests replays this
corpus with no Python at test time: log1p, logaddexp and expm1 outside [-ln 2, ln 1.5] must match bit for bit, expm1
inside that band within 2 ulp (the UCRT's in-band polynomial is undocumented; no public formula reproduces it).

The corpus is a win-amd64 pin. NumPy's float64 log1p / expm1 call the platform's libm: the UCRT on Windows, glibc (or
SVML loops on AVX-512 machines) on Linux, Apple's libm on macOS, so the recorded bits are those of Windows NumPy and
the replay runs on Windows only.

Run on Windows in the repo's main Python env (the interpreter the interop suite uses):
    python test/OptunaSharp.Tests/oracle/gen_numpy_scalar_math_oracle.py
"""
import json
import os
import platform
import struct

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "numpy_scalar_math.json")
LN2 = float(np.log(2.0))
LN15 = float(np.log(1.5))


def bits(x):
    """Encode a float64 as its IEEE-754 bit pattern, so the C# replay reads back the exact value (NaN payloads and
    the sign of zero included), which a decimal repr would not guarantee across formatters.

    Args:
        x: The value to encode; a Python float or a NumPy float64 scalar.

    Returns:
        The big-endian bit pattern as 16 lowercase hex digits (``"3ff0000000000000"`` for ``1.0``).
    """
    return struct.pack(">d", float(x)).hex()


def main():
    """Record NumPy's outputs over the fixed, seeded input set and write ``numpy_scalar_math.json`` beside this script.

    The inputs cover each function's special values, both sides of expm1's UCRT band, the two ranges where
    TruncNorm.NdtriExp calls expm1, tiny and huge magnitudes, and logaddexp's cancellation regime (near-equal pairs).
    Rerunning it on the same NumPy rewrites an identical file.

    Raises:
        SystemExit: When run off Windows, where NumPy's log1p/expm1 come from another libm and the corpus would no
            longer describe what the Windows replay checks.
    """
    if platform.system() != "Windows":
        raise SystemExit("record on Windows: NumPy's log1p/expm1 call the platform libm (see the module docstring)")
    rng = np.random.default_rng(20261111)

    log1p_x = np.concatenate([
        np.array([0.0, -0.0, -1.0, np.inf, 5e-324, -5e-324, 1e-300, -1e-300, 1e-17, -1e-17, 2.0 ** -53, 1.0, 2.0]),
        rng.uniform(-1.0, 1.0, 60),
        -1.0 + 10.0 ** rng.uniform(-15, -1, 20),
        rng.choice([-1.0, 1.0], 30) * 10.0 ** rng.uniform(-300, -1, 30),
        10.0 ** rng.uniform(0, 300, 40),
    ])

    expm1_x = np.concatenate([
        np.array([0.0, -0.0, -LN2, LN15, -1000.0, -745.2, 709.7, 710.0, -np.inf, np.inf, 5e-324, -1e-300, 1e-17]),
        rng.uniform(-40.0, -LN2, 40),
        rng.uniform(LN15, 700.0, 40),
        rng.uniform(-LN2, LN15, 60),
        -rng.uniform(0.0, 1e-2, 30),  # TruncNorm.NdtriExp's flip branch, expm1(y) for y in (-1e-2, 0].
        rng.uniform(1e-2, 5.0, 30),   # its initial guess, expm1(-z) for -z in [1e-2, 5].
        rng.choice([-1.0, 1.0], 20) * 10.0 ** rng.uniform(-300, -3, 20),
    ])

    a = rng.uniform(-50.0, 50.0, 80)
    near_b = a + rng.choice([-1.0, 1.0], 80) * 10.0 ** rng.uniform(-17, 2, 80)
    wide_a = rng.uniform(-1000.0, 1000.0, 40)
    wide_b = rng.uniform(-1000.0, 1000.0, 40)
    special = np.array([
        [0.0, 0.0], [-1e-5, -1e-5], [np.inf, np.inf], [-np.inf, -np.inf], [np.inf, 1.0], [-np.inf, 1.0],
        [1.0, -np.inf], [-700.0, -701.0], [-1.0, -2.0], [745.0, 745.0], [-745.0, -746.0],
    ])
    lae_a = np.concatenate([special[:, 0], a, wide_a])
    lae_b = np.concatenate([special[:, 1], near_b, wide_b])

    with np.errstate(all="ignore"):
        corpus = {
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "band": {"low": bits(-LN2), "high": bits(LN15)},
            "log1p": [{"x": bits(x), "y": bits(y)} for x, y in zip(log1p_x, np.log1p(log1p_x))],
            "expm1": [{"x": bits(x), "y": bits(y)} for x, y in zip(expm1_x, np.expm1(expm1_x))],
            "logaddexp": [{"a": bits(p), "b": bits(q), "y": bits(y)} for p, q, y in zip(lae_a, lae_b, np.logaddexp(lae_a, lae_b))],
        }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(corpus, f, indent=1)
        f.write("\n")
    print(f"wrote {OUT}: log1p {len(corpus['log1p'])}, expm1 {len(corpus['expm1'])}, logaddexp {len(corpus['logaddexp'])}")


if __name__ == "__main__":
    main()
