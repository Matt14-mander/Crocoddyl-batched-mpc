"""Fetch exact official visual meshes into an isolated cache, without installing packages."""

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

COMMIT = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
REPOSITORY = "unitreerobotics/unitree_ros"
ROOT = Path(__file__).resolve().parents[1]


def retrieve(url):
    request = urllib.request.Request(url, headers={"User-Agent": "Go2-Z1-asset-preparation"})
    with urllib.request.urlopen(request, timeout=90) as response:
        return response.read()


def git_blob(data):
    return hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()


def prepare(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    go2 = json.loads(Path(__file__).with_name("go2_z1_visual_meshes.json").read_text())
    arm = json.loads((ROOT / "src/crocoddyl_batched_mpc/assets/go2_arm/SOURCE.json").read_text())
    if go2["commit"] != COMMIT or arm["commit"] != COMMIT:
        raise ValueError("wrong upstream manifest commit")
    blobs = {item["path"]: item["git_blob_sha1"] for item in go2["files"]}
    blobs.update({item["upstream_path"]: item["git_blob_sha1"] for item in arm["meshes"]["files"]})
    blobs.update({item["upstream_path"]: item["git_blob_sha1"] for item in arm["files"]})
    urdf = ET.parse(ROOT / "src/crocoddyl_batched_mpc/assets/go2_arm/go2_z1_fixed_gripper.urdf")
    paths = {
        "robots/" + node.attrib["filename"].removeprefix("package://")
        for node in urdf.findall("./link/visual/geometry/mesh")
    }
    paths.add("LICENSE")

    def fetch(path):
        expected = blobs[path]
        local = directory / path
        data = local.read_bytes() if local.exists() else None
        if data is None or git_blob(data) != expected:
            data = retrieve(f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{path}")
            if git_blob(data) != expected:
                raise ValueError(f"upstream blob mismatch: {path}")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_bytes(data)
        return {
            "path": path,
            "git_blob_sha1": expected,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(fetch, sorted(paths)))
    # Collada images are optional; only resolve tracked, relative texture references.
    textures = set()
    for path in paths:
        if not path.endswith(".dae"):
            continue
        doc = ET.parse(directory / path)
        for node in doc.findall(".//{*}library_images/{*}image/{*}init_from"):
            name = (node.text or "").strip()
            candidate = str(PurePosixPath(path).parent / name)
            if candidate not in blobs:
                raise ValueError(f"unresolved Collada image: {path}: {name}")
            textures.add(candidate)
    records.extend(fetch(path) for path in sorted(textures - paths))
    report = {
        "repository": REPOSITORY,
        "commit": COMMIT,
        "license": "BSD-3-Clause",
        "visual_meshes": len(paths) - 1,
        "files": records,
    }
    (directory / "MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".cache/go2-z1-visual-assets")
    args = parser.parse_args()
    report = prepare(args.output)
    print(
        json.dumps(
            {
                "meshes": report["visual_meshes"],
                "files": len(report["files"]),
                "bytes": sum(f["size_bytes"] for f in report["files"]),
                "output": str(args.output),
            },
            indent=2,
        )
    )
