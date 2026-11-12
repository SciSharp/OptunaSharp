"""Record CPython's iteration order of int-set differences and the order list.sort() leaves NaN-bearing tuples in.

Two orders Optuna inherits from CPython itself, not from Optuna's code, which OptunaSharp reproduces in
OptunaSharp.Internal.PyIntSet and OptunaSharp.Internal.PyListSort and replays from this corpus with no Python at test
time (Internal/CPythonOrderOracleTests):

* ``set_difference``: GridSampler's ``list(set(range(n)) - set(visited) - set(running))`` and
  ``list(set(range(n)) - set(visited))`` (QA-68). The order is CPython's hash-table slot order, which depends on the
  table sizes, the probe sequence, and which of ``set_difference``'s two strategies runs.
* ``list_sort``: NSGA-III's ``list.sort(reverse=True)`` over ``(np.float64 distance, int index)`` tuples (QA-56). A
  NaN distance compares neither smaller nor larger than anything, so the result is whatever CPython's timsort (with
  its powersort merge policy since 3.11) does with an inconsistent order: only an exact port reproduces it. The
  distances come from :data:`POOL` (NaN, the infinities, both zeros, k / 4090), two characters per distance; the
  replay rebuilds the pool, as IEEE division makes ``k / 4090`` the same double on both sides.

The corpus pins CPython 3.12, the interpreter the interop suite runs on (CI installs 3.12). CPython 3.13 changed
``count_run`` (equal-element runs), which can change the order NaN-bearing lists end up in, so the generator refuses
another minor version rather than record a different order under the same file name.

Run in the repo's main Python env:
    python test/OptunaSharp.Tests/oracle/gen_cpython_order_oracle.py
"""
import base64
import hashlib
import json
import os
import random
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cpython_order.json")
FULL_LIST_LIMIT = 400
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
# The distances a sort case draws from, written to the corpus once: NaN, the infinities, both zeros, then 4,091 values
# k / 4090 in [0, 1]. A case stores each distance as two ALPHABET characters, its pool index in base 64.
POOL = [float("nan"), float("inf"), float("-inf"), 0.0, -0.0] + [k / 4090 for k in range(4091)]


def bitmap(values, n):
    """Encode a set of ints in ``range(n)`` as a little-endian bitmap, base64, so large cases stay small.

    Args:
        values: The ints to encode, possibly with duplicates; every one is in ``range(n)``.
        n: The size of the universe; the bitmap holds ``ceil(n / 8)`` bytes.

    Returns:
        The base64 text of the bitmap, bit ``k % 8`` of byte ``k // 8`` set for each ``k`` in ``values``.
    """
    data = bytearray((n + 7) // 8)
    for k in values:
        data[k >> 3] |= 1 << (k & 7)
    return base64.b64encode(bytes(data)).decode("ascii")


def describe(order):
    """Describe an iteration order so the replay can check it whole: in full when short, else by digest.

    Args:
        order: The list of ints in the order CPython produced.

    Returns:
        A dict with ``count`` and either ``items`` (the full list) or ``head`` / ``tail`` (the first and last 32) and
        ``sha256`` (of the decimal ints joined by commas, UTF-8).
    """
    if len(order) <= FULL_LIST_LIMIT:
        return {"count": len(order), "items": order}
    text = ",".join(str(v) for v in order).encode("utf-8")
    return {
        "count": len(order),
        "head": order[:32],
        "tail": order[-32:],
        "sha256": hashlib.sha256(text).hexdigest(),
    }


def set_case(n, visited, running):
    """Record GridSampler's two set differences for one history.

    Args:
        n: The grid size, ``len(all_grids)``.
        visited: Grid ids of finished trials (duplicates allowed, as GridSampler's list has them).
        running: Grid ids of running trials (duplicates allowed; may overlap ``visited``).

    Returns:
        The case: ``n``, the two bitmaps, and the orders of ``set(range(n)) - set(visited) - set(running)``
        (``unvisited``) and of ``set(range(n)) - set(visited)`` (``unfinished``).
    """
    unvisited = list(set(range(n)) - set(visited) - set(running))
    unfinished = list(set(range(n)) - set(visited))
    return {
        "n": n,
        "visited": bitmap(visited, n),
        "running": bitmap(running, n),
        "unvisited": describe(unvisited),
        "unfinished": describe(unfinished),
    }


def set_cases(rng):
    """Build the set-difference cases: the journal's example, edge sizes, and random histories.

    The random histories cover both of ``set_difference``'s strategies (copy-then-discard when the left set is more
    than four times the right one, a fresh table otherwise), the dummy purge after discards, and the table growth of
    ``set(range(n))`` on both sides of 50,000 elements.

    Args:
        rng: The seeded ``random.Random`` drawing the histories.

    Returns:
        The list of cases (see :func:`set_case`).
    """
    cases = [set_case(16, [0, 1, 2, 3, 5, 6, 7, 9, 10, 12, 13, 14, 15], [])]  # QA-68: [8, 11, 4]
    for n in (0, 1, 2, 5, 7, 8, 9, 31, 32, 33, 64, 100):
        cases.append(set_case(n, [], []))
        cases.append(set_case(n, list(range(0, n, 2)), list(range(1, n, 3))))
    for _ in range(240):
        n = rng.choice([rng.randint(1, 40), rng.randint(1, 300), rng.randint(1, 3000)])
        visited_density = rng.choice([0.0, 0.01, 0.1, 0.3, 0.6, 0.9, 0.99, 1.0])
        running_density = rng.choice([0.0, 0.0, 0.02, 0.1, 0.5])
        visited = [k for k in range(n) if rng.random() < visited_density]
        running = [k for k in range(n) if rng.random() < running_density]
        # Duplicates, as GridSampler's lists hold one entry per trial.
        visited += [rng.choice(visited) for _ in range(len(visited) // 7)] if visited else []
        rng.shuffle(visited)
        cases.append(set_case(n, visited, running))
    for n in (49999, 50001, 70000):
        visited = [k for k in range(n) if rng.random() < 0.97]
        running = [k for k in range(n) if rng.random() < 0.001]
        cases.append(set_case(n, visited, running))
        cases.append(set_case(n, [k for k in range(n) if rng.random() < 0.05], []))
    return cases


def encode(codes):
    """Encode pool indices as the compact text a sort case stores.

    Args:
        codes: Indices into :data:`POOL`, each below 4,096.

    Returns:
        Two :data:`ALPHABET` characters per index, the high base-64 digit first.
    """
    return "".join(ALPHABET[c >> 6] + ALPHABET[c & 63] for c in codes)


def sort_case(codes, reverse):
    """Record the order ``list.sort`` leaves NSGA-III's ``(distance, index)`` tuples in.

    Args:
        codes: The distances as indices into :data:`POOL`. NSGA-III takes its distances from a float64 array, so
            each tuple gets its own ``np.float64`` object and two NaNs are never identical, never equal.
        reverse: Whether to sort with ``reverse=True`` (NSGA-III's call).

    Returns:
        The case: the encoded distances, ``reverse``, and the indices in sorted order (see :func:`describe`).
    """
    array = np.array([POOL[c] for c in codes], dtype=np.float64)
    pairs = [(array[i], i) for i in range(len(codes))]
    pairs.sort(reverse=reverse)
    return {"distances": encode(codes), "reverse": reverse, "order": describe([i for _, i in pairs])}


def quantize(x):
    """Map a value in [0, 1] to the pool index of the nearest k / 4090.

    Args:
        x: The value, in [0, 1].

    Returns:
        Its pool index (5 for 0.0, 4095 for 1.0).
    """
    return 5 + int(round(x * 4090))


def sort_cases(rng):
    """Build the sort cases: the journal's example, structured runs, and random NaN-bearing lists.

    Sizes reach past 64 so runs are extended by binary insertion, merged by the powersort policy, and merged with
    galloping; NaN rates from none to all, ties from a small value pool, infinities and signed zeros.

    Args:
        rng: The seeded ``random.Random`` drawing the lists.

    Returns:
        The list of cases (see :func:`sort_case`).
    """
    cases = [sort_case([0, 0], True)]  # QA-56: the borderline front of two (inf, inf) trials, NaN distances
    special = [1, 2, 3, 4, quantize(0.25), quantize(0.5), quantize(1.0)]
    for size in (0, 1, 2, 3, 7, 32, 63, 64, 65, 127, 128, 129, 300, 1000):
        for nan_rate in (0.0, 0.05, 0.3, 1.0):
            for reverse in (True, False):
                codes = [0 if rng.random() < nan_rate else rng.choice(special + [quantize(rng.random())])
                         for _ in range(size)]
                cases.append(sort_case(codes, reverse))
    for _ in range(240):
        size = rng.choice([rng.randint(0, 20), rng.randint(20, 200), rng.randint(200, 800)])
        nan_rate = rng.choice([0.0, 0.01, 0.1, 0.4, 0.9])
        shape = rng.choice(["random", "ascending", "descending", "plateaus", "sawtooth"])
        if shape == "random":
            codes = [quantize(rng.random()) for _ in range(size)]
        elif shape == "ascending":
            codes = sorted(quantize(rng.random()) for _ in range(size))
        elif shape == "descending":
            codes = sorted((quantize(rng.random()) for _ in range(size)), reverse=True)
        elif shape == "plateaus":
            codes = [quantize(rng.randint(0, 4) / 4) for _ in range(size)]
        else:
            codes = [quantize(((i * 7) % 23) / 23) for i in range(size)]
        codes = [0 if rng.random() < nan_rate else c for c in codes]
        cases.append(sort_case(codes, rng.random() < 0.7))
    return cases


def main():
    """Write ``cpython_order.json`` beside this script. Rerunning on the same CPython rewrites an identical file.

    Raises:
        SystemExit: Under a CPython other than 3.12, whose sort may order NaN-bearing lists differently.
    """
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("record with CPython 3.12 (see the module docstring); this is %s" % sys.version.split()[0])
    rng = random.Random(20261112)
    corpus = {
        "python": sys.version.split()[0],
        "set_difference": set_cases(rng),
        "list_sort": sort_cases(rng),
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(corpus, f, separators=(",", ":"))
        f.write("\n")
    print("wrote", OUT, len(corpus["set_difference"]), "set cases,", len(corpus["list_sort"]), "sort cases")


if __name__ == "__main__":
    main()
