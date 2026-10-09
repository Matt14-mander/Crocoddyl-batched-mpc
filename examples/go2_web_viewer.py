"""Build a Meshcat gallery from saved actual closed-loop Go2 traces."""

import argparse
import webbrowser
from pathlib import Path

from crocoddyl_batched_mpc.go2_viewer import Go2Trace, export_gallery, export_go2_html


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/go2-viewer"))
    parser.add_argument("--mesh-dir", type=Path)
    parser.add_argument("--open", action="store_true")
    args = parser.parse_args()
    pages = {}
    for i, path in enumerate(args.traces):
        trace = Go2Trace.load(path)
        if trace.name in pages:
            parser.error(f"duplicate gait name: {trace.name}")
        pages[trace.name] = export_go2_html(
            trace, args.output / f"playback_{i}.html", mesh_dir=args.mesh_dir
        )
    gallery = export_gallery(pages, args.output / "index.html")
    print(gallery)
    if args.open:
        webbrowser.open(gallery.as_uri())


if __name__ == "__main__":
    main()
