"""Load monitor, idle gate and load guard for the OptunaSharp README benchmark on the worker machine (Proxmox host ``worker``).

Usage:
    python -I monitor.py log <file> <interval_s>
        Appends one line per interval until killed: wall-clock time, epoch (the window's end), total CPU busy %, the
        Proxmox guests' CPU use (containers and VMs), the host services' (system.slice) and the other host sessions'
        (user.slice without this process's own session, so without the benchmark, whose runner starts this logger), each
        in % of one CPU, then each logical CPU's busy %.
    python -I monitor.py gate <max_total_pct> <max_guest_pct> <windows> <window_s> <max_wait_s>
        Returns (exit 0) once ``windows`` consecutive windows of ``window_s`` seconds each show the whole machine
        below ``max_total_pct`` % busy AND the guests below ``max_guest_pct`` % of one CPU; prints those windows on one
        line. After ``max_wait_s`` seconds without such a stretch it prints the last windows and exits 1, so the
        runner can abort instead of measuring a loaded machine.
    python -I monitor.py check <file> <start_epoch> <end_epoch> <max_guest_pct> <max_services_pct> <max_others_pct> <interval_s>
        The load guard: judges one process's span from the log ``log`` writes. Every logged window that overlaps the span
        must show the guests, the host services and the other host sessions below their limits (% of one CPU). Waits up
        to three intervals for the window holding the span's end. Prints one line and exits 0 (``clean: ...``), 1
        (``interfered: ...``) or 2 (``unknown: ...``, no window logged: the logger is not running).

Why all three: the gate only checks the moment before a process starts; work that starts in the middle of a timing (a
TraderLab job in CT 801, a build in CT 803, the on-demand LLM in CT 804, a Windows desktop VM, another session's command
on the host) shows only in the log, so the guard judges each process from it once the process has ended, the runner
redoes a process the guard rejects, and the parser drops any that stayed rejected. Guests are read from their cgroups
(``/sys/fs/cgroup/lxc/<id>`` and ``/sys/fs/cgroup/qemu.slice/<id>.scope``), which separates their load from the
benchmark's own on shared CPUs; the benchmark's own load is excluded from "other sessions" through its cgroup, the
session scope the runner was started in (``setsid`` keeps it), which the logger inherits.
"""
import glob
import re
import sys
import time


def read_stat() -> dict:
    """Busy and total jiffies per CPU line of /proc/stat.

    Returns:
        ``{"cpu": (busy, total), "cpu0": (busy, total), ...}``; idle and iowait count as not busy, guest time is
        already inside user time, so only the first eight fields are summed.

    Raises:
        OSError: /proc/stat cannot be read (not Linux).
    """
    cpus = {}
    with open("/proc/stat", encoding="ascii") as f:
        for line in f:
            if not line.startswith("cpu"):
                break
            parts = line.split()
            vals = [int(v) for v in parts[1:9]]
            idle = vals[3] + vals[4]
            total = sum(vals)
            cpus[parts[0]] = (total - idle, total)
    return cpus


def cgroup_usec(patterns) -> int:
    """Summed ``usage_usec`` of every cgroup matching the glob patterns (descendants are included by the kernel).

    Args:
        patterns: Glob patterns (or plain paths) of ``cpu.stat`` files.

    Returns:
        Total CPU time in microseconds; a cgroup that vanished between the glob and the read counts as zero, so a guest
        that stops mid-window under-reports that window rather than failing the logger.
    """
    total = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                with open(path, encoding="ascii") as f:
                    for line in f:
                        if line.startswith("usage_usec"):
                            total += int(line.split()[1])
                            break
            except OSError:
                pass
    return total


def own_cgroup() -> str:
    """This process's cgroup v2 path, e.g. ``/user.slice/user-0.slice/session-7.scope``.

    Returns:
        The path from ``/proc/self/cgroup``'s unified (``0::``) line, or ``""`` when there is none (not Linux, or cgroup
        v1 only), in which case "other sessions" counts the whole of user.slice, the benchmark included, so the guard's
        ``others`` limit would reject every process: the runner only runs on cgroup v2 hosts (Proxmox VE 7+).
    """
    try:
        with open("/proc/self/cgroup", encoding="ascii") as f:
            for line in f:
                if line.startswith("0::"):
                    return line[3:].strip()
    except OSError:
        pass
    return ""


GUESTS = ["/sys/fs/cgroup/lxc/*/cpu.stat", "/sys/fs/cgroup/qemu.slice/*/cpu.stat"]
HOST_SERVICES = ["/sys/fs/cgroup/system.slice/cpu.stat"]
USERS = ["/sys/fs/cgroup/user.slice/cpu.stat"]
# The benchmark's own session scope: subtracted from user.slice only when it lies inside it (a runner started from a
# login session); a runner started any other way is outside user.slice, so nothing of its load is there to subtract.
_OWN = own_cgroup()
OWN = [f"/sys/fs/cgroup{_OWN}/cpu.stat"] if _OWN.startswith("/user.slice/") else []

# One line of the log; ``others`` is optional so logs written before it existed (2026-11-23 10:03 run) still parse.
LINE = re.compile(r"^(\S+) ([\d.]+) total=([\d.]+) guests=([\d.]+) services=([\d.]+)(?: others=([\d.]+))? cpu=")


def window(seconds: float):
    """Measure one window.

    Args:
        seconds: The window's length.

    Returns:
        ``(total_pct, per_cpu_pcts, guest_pct, services_pct, others_pct)`` over the window, the last three in % of one
        CPU (100 = one CPU fully busy).

    Raises:
        OSError: /proc/stat cannot be read (not Linux).
    """
    s0, g0, h0, u0, o0, t0 = (read_stat(), cgroup_usec(GUESTS), cgroup_usec(HOST_SERVICES), cgroup_usec(USERS),
                              cgroup_usec(OWN), time.monotonic())
    time.sleep(seconds)
    s1, g1, h1, u1, o1, t1 = (read_stat(), cgroup_usec(GUESTS), cgroup_usec(HOST_SERVICES), cgroup_usec(USERS),
                              cgroup_usec(OWN), time.monotonic())

    def pct(name):
        busy = s1[name][0] - s0[name][0]
        total = s1[name][1] - s0[name][1]
        return 100.0 * busy / total if total > 0 else 0.0

    cpus = sorted((k for k in s1 if k != "cpu"), key=lambda k: int(k[3:]))
    elapsed = t1 - t0
    # The two user.slice reads and the two own-scope reads are microseconds apart, not atomic: the difference can dip a
    # hair below zero when the benchmark burns CPU between them, which clamps to zero rather than logging negative load.
    others = max(0.0, ((u1 - u0) - (o1 - o0)) / elapsed / 1e4)
    return pct("cpu"), [pct(c) for c in cpus], (g1 - g0) / elapsed / 1e4, (h1 - h0) / elapsed / 1e4, others


def log(path: str, interval: float) -> None:
    """Append a load line every ``interval`` seconds until the process is killed.

    Args:
        path: The log file (appended to, so a restarted logger continues the same file).
        interval: Seconds per line (each line covers exactly that window, ending at the logged epoch).

    Raises:
        OSError: /proc/stat cannot be read or the log cannot be written.
    """
    while True:
        total, per_cpu, guests, services, others = window(interval)
        line = (f"{time.strftime('%H:%M:%S')} {time.time():.1f} total={total:.1f} guests={guests:.1f} "
                f"services={services:.1f} others={others:.1f} cpu={','.join(f'{p:.0f}' for p in per_cpu)}\n")
        with open(path, "a", encoding="ascii") as f:
            f.write(line)


def gate(max_total: float, max_guest: float, windows: int, seconds: float, max_wait: float) -> int:
    """Wait for an idle stretch.

    Args:
        max_total: Highest whole-machine busy % a window may show.
        max_guest: Highest guest CPU use (% of one CPU) a window may show.
        windows: How many consecutive windows must pass.
        seconds: Each window's length.
        max_wait: Give up after this many seconds.

    Returns:
        0 when the stretch was found (its windows are printed), 1 on timeout (the last windows are printed).

    Raises:
        OSError: /proc/stat cannot be read (not Linux).
    """
    start = time.monotonic()
    recent = []
    while True:
        total, _, guests, _, _ = window(seconds)
        recent.append((total, guests))
        recent = recent[-windows:]
        if len(recent) == windows and all(t < max_total and g < max_guest for t, g in recent):
            # Long stretches (the guard's after-a-void gate) print a summary, not one figure per window.
            if windows <= 5:
                cpu_text = " ".join(f"{t:.1f}" for t, _ in recent)
                guest_text = " ".join(f"{g:.1f}" for _, g in recent)
            else:
                cpu_text = f"max {max(t for t, _ in recent):.1f} over {windows}"
                guest_text = f"max {max(g for _, g in recent):.1f} over {windows}"
            print(f"idle: cpu% {cpu_text} | guests% {guest_text} | waited {time.monotonic() - start:.0f}s")
            return 0
        if time.monotonic() - start > max_wait:
            print("timeout: cpu% " + " ".join(f"{t:.1f}" for t, _ in recent[-5:])
                  + " | guests% " + " ".join(f"{g:.1f}" for _, g in recent[-5:]))
            return 1


def read_log(path: str) -> list:
    """The windows of a load log.

    Args:
        path: The file ``log`` writes.

    Returns:
        ``[(clock, epoch, total, guests, services, others), ...]`` in file order; ``others`` is None on lines from before
        it was logged. A missing file reads as no windows (the logger has not written yet).
    """
    out = []
    try:
        with open(path, encoding="ascii", errors="replace") as f:
            for line in f:
                m = LINE.match(line)
                if m:
                    out.append((m.group(1), float(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5)),
                                float(m.group(6)) if m.group(6) is not None else None))
    except FileNotFoundError:
        pass
    return out


def overlapping(samples: list, start: float, end: float, interval: float) -> list:
    """The windows that overlap a span.

    Args:
        samples: ``read_log``'s windows.
        start: The span's start (epoch seconds).
        end: The span's end (epoch seconds).
        interval: The logger's window length.

    Returns:
        The windows ending after ``start`` whose start (taken as their end minus 1.05 intervals, a little early, since a
        window is the sleep plus the reads around it) lies before ``end``: a window that only touches the span's edge
        counts, so a job that ended just before the process started can reject it (the conservative side).
    """
    return [s for s in samples if s[1] > start and s[1] - 1.05 * interval < end]


def check(path: str, start: float, end: float, max_guest: float, max_services: float, max_others: float,
          interval: float) -> int:
    """The load guard for one process's span.

    Args:
        path: The load log.
        start: The process's start (epoch seconds).
        end: The process's end (epoch seconds).
        max_guest: The guests' limit (% of one CPU); a window at or above it rejects the process.
        max_services: The host services' limit (% of one CPU).
        max_others: The other host sessions' limit (% of one CPU); ignored for windows logged without the field.
        interval: The logger's window length (seconds).

    Returns:
        0 (clean), 1 (interfered) or 2 (unknown: no window was logged after the span ended within three intervals, so the
        logger is not running); the verdict is printed on one line.
    """
    deadline = time.monotonic() + 3 * interval + 1
    while True:
        samples = read_log(path)
        # The window holding the span's end must be on disk before the span can be judged whole.
        if samples and samples[-1][1] >= end:
            break
        if time.monotonic() > deadline:
            print(f"unknown: no load window logged after the process ended (last at "
                  f"{samples[-1][0] if samples else 'none'}); is the logger running?")
            return 2
        time.sleep(0.25)
    inside = overlapping(samples, start, end, interval)
    limits = (("guests", 3, max_guest), ("services", 4, max_services), ("others", 5, max_others))
    peaks = []
    over = []
    for name, index, limit in limits:
        values = [s for s in inside if s[index] is not None]
        if not values:
            continue
        worst = max(values, key=lambda s: s[index])
        peaks.append(f"{name} {worst[index]:.1f}")
        if worst[index] >= limit:
            over.append(f"{name} {worst[index]:.1f} at {worst[0]} (limit {limit:g})")
    if over:
        print(f"interfered: {len(inside)} windows | " + ", ".join(over))
        return 1
    print(f"clean: {len(inside)} windows | peak " + " ".join(peaks) + " (% of a CPU)")
    return 0


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "log":
        log(sys.argv[2], float(sys.argv[3]))
    elif mode == "gate":
        sys.exit(gate(float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]),
                      float(sys.argv[6])))
    elif mode == "check":
        sys.exit(check(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6]),
                       float(sys.argv[7]), float(sys.argv[8])))
    else:
        raise SystemExit("mode must be log, gate or check")
