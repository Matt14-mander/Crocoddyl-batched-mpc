"""Local demo launcher. Read existing dependencies; never install or modify them.

Run with the repository's .venv Python. The optional viewer uses the dedicated
Meshcat cache and reads transport dependencies from the venv's base installation.
The solver never receives the viewer/base dependency paths.
"""

import argparse
import functools
import http.server
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "artifacts/go2-z1-stance-meshcat"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("serve", "solve", "export", "assets"))
    parser.add_argument("--port", type=int, default=8009)
    args, remaining = parser.parse_known_args()
    os.chdir(ROOT)
    if args.command == "serve":
        if remaining:
            parser.error(f"Unrecognized arguments: {remaining}")
        if not (ARTIFACT / "index.html").exists():
            parser.error("Run solve and export first")
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ARTIFACT))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler)
        print(f"Meshcat demo: http://127.0.0.1:{args.port}/ (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
        return
    sys.dont_write_bytecode = True
    native = ROOT / ".dev-tools/crocoddyl321-pin400/cmeel.prefix/lib/python3.11/site-packages"
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(native))
    default_args = []
    if args.command == "export":
        sys.path.insert(0, str(ROOT / ".cache/go2-z1-viewer-python"))
        base_site = (
            Path(sys.base_prefix)
            / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages"
        )
        # Append only: the venv's NumPy retains priority over the base installation.
        if base_site.exists() and str(base_site) not in sys.path:
            sys.path.append(str(base_site))
        default_args = [
            "--report",
            str(ARTIFACT / "report.json"),
            "--trace",
            str(ARTIFACT / "trace.npz"),
            "--output",
            str(ARTIFACT / "index.html"),
        ]
    entry = {
        "solve": "go2_z1_stance_demo.py",
        "export": "go2_z1_stance_meshcat.py",
        "assets": "prepare_go2_z1_meshes.py",
    }[args.command]
    sys.argv = [entry, *default_args, *remaining]
    runpy.run_path(str(ROOT / "examples" / entry), run_name="__main__")


if __name__ == "__main__":
    main()
