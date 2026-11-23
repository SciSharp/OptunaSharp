"""Run a Python benchmark script after describing (and, on Windows, preparing) the process it runs in.

Usage: python -I pyrun.py <script.py> [script args...]

The C# probes get the same treatment from common/ProbeEnv.cs. Every process prints one ``probe env:`` line first:
the interpreter, the CPUs it may run on (on Linux the ``Cpus_allowed_list`` that ``taskset -c`` set) and, on
Windows only, whether the execution-speed power-throttling opt-out was applied (a hybrid CPU moves a background
console process's busy thread to its efficiency cores after about 2.5 s). The worker-machine run's parser checks
the affinity field against the CPUs the runner meant, so a mis-pinned timing is dropped instead of averaged in.
The script then runs as ``__main__`` with ``sys.argv`` shifted, exactly as ``python script.py args`` would run it,
except that this folder is first on ``sys.path``: ``-I`` (which keeps the user's site-packages and environment out) also
keeps the script's own folder off the path, and every probe imports ``benchlib``, the measurement protocol, from here.
"""
import os
import platform
import runpy
import sys


def _opt_out_windows() -> str:
    """Apply ControlMask=EXECUTION_SPEED, StateMask=0 ("never throttle") through ProcessPowerThrottling (class 4).

    Returns:
        "True" when Windows accepted the setting, "False" when it refused, "n/a" on any other system.
    """
    if sys.platform != "win32":
        return "n/a"
    import ctypes  # noqa: PLC0415 - Windows only
    from ctypes import wintypes  # noqa: PLC0415 - Windows only

    class _PowerThrottlingState(ctypes.Structure):
        """PROCESS_POWER_THROTTLING_STATE: Version 1; bit 0 of the masks is EXECUTION_SPEED."""

        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    state = _PowerThrottlingState(1, 1, 0)
    return str(bool(k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))))


def _affinity() -> str:
    """The CPUs this process may run on, as the kernel lists them.

    Returns:
        On Linux the ``Cpus_allowed_list`` field of ``/proc/self/status`` (for example ``2-3``); elsewhere, or when
        procfs cannot be read, ``unknown`` (the parser then treats the process as unpinned).
    """
    try:
        with open("/proc/self/status", encoding="ascii") as status:
            for line in status:
                if line.startswith("Cpus_allowed_list:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "unknown"


if __name__ == "__main__":
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    print(f"probe env: os={platform.system()} python={platform.python_version()} cpus={cpus} "
          f"affinity={_affinity()} throttle_optout={_opt_out_windows()}", flush=True)
    target = sys.argv[1]
    sys.argv = sys.argv[1:]
    # The harness's own folder, not the current directory: -I's protection against a planted module in the working
    # directory stays, and the probes still find benchlib.py beside them.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    runpy.run_path(target, run_name="__main__")
