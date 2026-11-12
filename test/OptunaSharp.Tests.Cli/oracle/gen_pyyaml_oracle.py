"""Regenerates oracle/pyyaml_safe_dump.json: value trees and PyYAML's yaml.safe_dump output for each.

The CLI's `-f yaml` output must be PyYAML's safe_dump byte for byte (QA-61 of docs/qa/PARITY_QA_2026-11-10.md), so
`PyYamlSafeDumpOracleTests` replays this corpus through `PyYamlSafeDump.Dump` with no Python at test time.

Each case is {"tree": <tagged tree>, "yaml": <safe_dump output>}. The tree is tagged so C# rebuilds exact Python
types: {"d": [[key, value], ...]} a dict (items in insertion order; safe_dump sorts them), {"l": [...]} a list,
{"s": "..."} a str, {"i": "123"} an int (decimal text), {"f": <int64 bits>} a float (its IEEE-754 bits, so NaN,
infinities and -0.0 survive JSON), {"b": true} a bool, {"n": null} None.

The cases are the CLI's shapes (a record or a list of records, nested params / user_attrs, empty collections) filled
with the scalars that decide PyYAML's style: YAML 1.1 look-alikes (bools, ints, floats, sexagesimals, timestamps,
null, merge, value), indicators in every position, quotes, leading / trailing / doubled spaces, line breaks, control
and non-ASCII characters (allow_unicode=False escapes them), astral characters, and strings long enough to fold at
80 columns, plus keys of 127 to 129 characters around the simple-key limit. Seeded, so a rerun is identical for the
same PyYAML.

usage: python gen_pyyaml_oracle.py          (writes pyyaml_safe_dump.json next to this script)
"""
import json
import math
import os
import random
import struct

import yaml

TRICKY = [
    "", " ", "a", "y", "n", "yes", "No", "OFF", "on", "true", "False", "null", "Null", "~", "<<", "=", "!", "&", "*",
    "0", "1", "-1", "+1", "007", "0b101", "0x1F", "0o7", "1_000", "1:30", "190:20:30", "1.5", "-1.5", ".5", "1e3",
    "1.0e+3", "1.e3", ".inf", "-.Inf", ".NaN", "nan", "inf", "1:30.5", "2026-11-10", "2026-11-10 22:51:17",
    "2026-11-10T22:51:17Z", "2026-1-2 3:04:05.5 +09:00", "0:00:00.012345", "2001-12-14t21:59:43.10-05:00",
    "- a", "-a", "-", "--", "---", "...", "--- x", "? a", "?a", ": a", ":a", "a:b", "a: b", "a :b", "a:", "a #b", "a#b",
    "#a", "[a]", "a[0]", "{a}", "a,b", "a, b", "&x", "*x", "!x", "|", "|x", ">", ">x", "'", "it's", "''", '"', 'say "hi"',
    "%x", "@x", "`x`", "a\tb", "\ta", "a\t", " a", "a ", "a  b", "a\nb", "a\n", "\na", "a\n\nb", "a \nb", "a\n b",
    "\x07", "a\x00b", "\x85", "\u2028", "\u2029", "\ufeff", "\xa0", "é", "naïve", "日本語", "😀", "a😀b", "\\", "a\\b",
    "C:\\path\\to", "http://x.y/z?q=1#f", "key: value # comment", "{\"a\": [1, 2]}",
    "word " * 20, "word " * 40, "x" * 100, "a " * 50 + "b", "aa bb " * 15 + "é", "é " * 60, "line one\nline two " * 6,
    "it's " * 25, "  two leading", "trailing two  ", "a" * 79 + " b", "a" * 80 + " b c",
]

FLOATS = [0.0, -0.0, 1.0, -1.0, 2.0, 0.1, 1e-07, 1e16, 1e17, 1.5e300, 5e-324, 1e22, 123456789.123, -2.5e-10,
          float("nan"), float("inf"), float("-inf"), 3.141592653589793, 100.0, 1e100]
INTS = [0, 1, -1, 7, 42, 2 ** 31, 2 ** 53 + 1, -(2 ** 40), 10 ** 18]


def tag(value):
    """Tags a Python value for the C# side (see the module docstring).

    Args:
        value: None, a bool, int, float, str, dict, list or tuple (tuples tag as lists, as safe_dump writes them).

    Returns:
        The tagged form: a one-key dict whose key names the type.

    Raises:
        TypeError: The value (or something nested in it) has another type, which the corpus never produces.
    """
    if value is None:
        return {"n": None}
    if isinstance(value, bool):
        return {"b": value}
    if isinstance(value, int):
        return {"i": str(value)}
    if isinstance(value, float):
        return {"f": struct.unpack("<q", struct.pack("<d", value))[0]}
    if isinstance(value, str):
        return {"s": value}
    if isinstance(value, dict):
        return {"d": [[tag(k), tag(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return {"l": [tag(v) for v in value]}
    raise TypeError(type(value))


def scalar(rng):
    """A random scalar of any kind safe_dump represents.

    Args:
        rng: The seeded random.Random driving the corpus, so a rerun is identical.

    Returns:
        A tricky string (55%), a float (20%), an int (15%), or True / False / None.
    """
    kind = rng.random()
    if kind < 0.55:
        return rng.choice(TRICKY)
    if kind < 0.75:
        return rng.choice(FLOATS)
    if kind < 0.9:
        return rng.choice(INTS)
    return rng.choice([True, False, None])


def key(rng):
    """A random mapping key: mostly strings, sometimes long ones around the 128-character simple-key limit.

    Args:
        rng: The seeded random.Random driving the corpus.

    Returns:
        A string key; one in twenty is 127, 128 or 129 characters, so both sides of the limit are covered.
    """
    if rng.random() < 0.05:
        return "k" * rng.choice([127, 128, 129])
    return rng.choice(TRICKY + ["x", "y", "lr", "n_estimators", "max_depth", "params", "value", "values", "z9"])


def tree(rng, depth):
    """A random value: nested dicts and lists to `depth`, scalars at the leaves.

    Args:
        rng: The seeded random.Random driving the corpus.
        depth: How many more levels of nesting are allowed; 0 forces a scalar.

    Returns:
        A scalar, a dict of up to four items or a list of up to four items (empty ones included).
    """
    if depth == 0 or rng.random() < 0.35:
        return scalar(rng)
    if rng.random() < 0.55:
        return {key(rng): tree(rng, depth - 1) for _ in range(rng.randint(0, 4))}
    return [tree(rng, depth - 1) for _ in range(rng.randint(0, 4))]


def record(rng):
    """A CLI-like trial record: the columns `optuna trials -f yaml` prints, with random contents.

    Args:
        rng: The seeded random.Random driving the corpus.

    Returns:
        A dict with number, value, the formatted datetimes and duration, params, user_attrs, state and, for about a
        third, a multi-objective values list.
    """
    rec = {
        "number": rng.randint(0, 99),
        "value": rng.choice(FLOATS + INTS + [None]),
        "datetime_start": "2026-11-10 22:51:17",
        "datetime_complete": rng.choice(["2026-11-10 22:51:18", None]),
        "duration": rng.choice(["0:00:00.012345", "0:00:01", None]),
        "params": {key(rng): scalar(rng) for _ in range(rng.randint(0, 4))},
        "user_attrs": {key(rng): tree(rng, 2) for _ in range(rng.randint(0, 3))},
        "state": rng.choice(["COMPLETE", "FAIL", "PRUNED", "RUNNING"]),
    }
    if rng.random() < 0.3:
        rec["values"] = [rng.choice(FLOATS) for _ in range(rng.randint(1, 3))]
    return rec


def main():
    """Builds the corpus and writes pyyaml_safe_dump.json next to this script.

    Raises:
        OSError: The JSON file cannot be written.
    """
    rng = random.Random(20261112)
    cases = []
    # Every tricky string as a value and as a key, alone, so each one's style is pinned on its own.
    for s in TRICKY:
        cases.append({"v": s})
        cases.append({s: 1})
        cases.append([s])
    for f in FLOATS:
        cases.append({"f": f})
    for i in INTS:
        cases.append({"i": i})
    cases += [{}, [], [{}], [[]], {"a": {}}, {"a": []}, [[1, [2, [3]]]], {"a": [{"b": [1, 2]}, {"c": {}}]},
              {3: "a", 1: "b", 2.5: "c"}, {"k" * 127: 1, "k" * 128: 2, "k" * 129: [1, 2]}]
    for _ in range(150):
        cases.append(record(rng))
    for _ in range(40):
        cases.append([record(rng) for _ in range(rng.randint(1, 3))])
    for _ in range(150):
        cases.append(tree(rng, 4))

    out = [{"tree": tag(c), "yaml": yaml.safe_dump(c)} for c in cases]
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pyyaml_safe_dump.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump({"pyyaml": yaml.__version__, "cases": out}, f, ensure_ascii=True, indent=0)
        f.write("\n")
    print("wrote %d cases (PyYAML %s) to %s" % (len(out), yaml.__version__, path))


if __name__ == "__main__":
    main()
