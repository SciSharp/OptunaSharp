#!/usr/bin/env python3
"""Open the Python optuna-dashboard on the study OptunaHubDashboard.cs persisted.

Shared by run-optunahub.sh / run-optunahub.bat (and runnable on its own). It:
  * resolves the study's SQLite file from THIS file's location (so it always finds the .db the
    C# example wrote beside it, whatever the working directory),
  * looks up the `optuna-dashboard` console entry point in the SAME interpreter it is run by
    (no reliance on the Scripts/bin dir being on PATH), and
  * starts the server and opens your browser at http://<host>:<port>/.

Usage:
    python examples/optunahub/dashboard.py [storage-url] [--port N] [--host H]

The storage URL defaults to  sqlite:///<this dir>/optunahub-dashboard.db.
Ctrl+C stops the server.

Docs: https://optuna-dashboard.readthedocs.io/en/latest/
"""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from importlib.metadata import entry_points
from pathlib import Path

DEFAULT_DB = Path(__file__).with_name("optunahub-dashboard.db")


def _pop_opt(argv: list[str], name: str, default: str) -> str:
    """Pull `--name value` (or `--name=value`) out of argv, returning its value or the default."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            value = argv[i + 1]
            del argv[i : i + 2]
            return value
        if a.startswith(name + "="):
            value = a.split("=", 1)[1]
            del argv[i]
            return value
    return default


def _resolve_dashboard_main():
    """Return optuna-dashboard's CLI entry point, or None if the package isn't installed."""
    try:
        eps = entry_points(group="console_scripts")          # py3.10+ selectable API
    except TypeError:                                        # very old importlib.metadata
        eps = entry_points().get("console_scripts", [])
    for ep in eps:
        if ep.name == "optuna-dashboard":
            return ep.load()
    return None


def main() -> int:
    argv = sys.argv[1:]
    port = _pop_opt(argv, "--port", "8080")
    host = _pop_opt(argv, "--host", "127.0.0.1")

    # a positional storage url wins; otherwise the study the C# example wrote next to this file.
    storage = next((a for a in argv if not a.startswith("-")), None)
    if storage is None:
        if not DEFAULT_DB.exists():
            sys.stderr.write(
                f"No study database at {DEFAULT_DB}.\n"
                "Run the persisting example first:\n"
                "    dotnet run examples/optunahub/OptunaHubDashboard.cs\n"
            )
            return 0                                          # not an error — nothing to show yet
        storage = "sqlite:///" + DEFAULT_DB.as_posix()

    dashboard_main = _resolve_dashboard_main()
    if dashboard_main is None:
        sys.stderr.write(
            "optuna-dashboard is not installed in this interpreter.\n"
            "    pip install optuna-dashboard\n"
        )
        return 1

    view_url = f"http://{host}:{port}/"
    print(f"  optuna-dashboard  ->  {view_url}")
    print(f"  storage           ->  {storage}")
    print("  (Ctrl+C to stop the dashboard.)")

    # open the browser once the server has had a moment to bind the port.
    def _open_browser() -> None:
        time.sleep(1.5)
        if os.environ.get("OPTUNASHARP_NO_BROWSER") in (None, "", "0", "false", "False"):
            try:
                webbrowser.open(view_url)
            except Exception:
                pass

    threading.Thread(target=_open_browser, daemon=True).start()

    # hand argv to optuna-dashboard's own argparse (argv[0] is the program name) and run — this blocks.
    sys.argv = ["optuna-dashboard", storage, "--port", str(port), "--host", host]
    try:
        dashboard_main()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
