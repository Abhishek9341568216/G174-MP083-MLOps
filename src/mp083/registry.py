"""Filesystem model versions and an atomic approval pointer."""
import json
import os
import re
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

import joblib
from filelock import FileLock


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as file:
        temporary = Path(file.name)
        try:
            json.dump(data, file, indent=2, allow_nan=False)
            file.write("\n")
        except Exception:
            file.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def register(root, model, metadata):
    version = uuid4().hex
    root = Path(root)
    staging = root / f".{version}"
    staging.mkdir(parents=True)
    try:
        joblib.dump(model, staging / "pipeline.joblib")
        write_json(staging / "metadata.json", {**metadata, "version": version})
        staging.rename(root / version)
    finally:
        if staging.exists():
            for file in staging.iterdir():
                file.unlink()
            staging.rmdir()
    return version


def load_version(root, version=None):
    root = Path(root)
    if version is None:
        version = json.loads((root / "approved.json").read_text(encoding="utf-8"))["version"]
    if not isinstance(version, str) or not re.fullmatch(r"[0-9a-f]{32}", version):
        raise ValueError("Invalid model version")
    directory = root / version
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    if metadata["version"] != version:
        raise ValueError("Model version does not match metadata")
    return joblib.load(directory / "pipeline.joblib"), metadata


def approve(root, version, *, expected=None):
    load_version(root, version)
    pointer = Path(root) / "approved.json"
    with FileLock(pointer.with_suffix(".lock"), timeout=10):
        if expected is not None:
            current = json.loads(pointer.read_text(encoding="utf-8"))["version"]
            if current != expected:
                raise ValueError("Approved model changed during candidate evaluation")
        write_json(pointer, {"version": version})


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["approve", "rollback", "show"])
    parser.add_argument("version", nargs="?")
    parser.add_argument("--models", default="models")
    args = parser.parse_args()
    if args.action == "show":
        _, metadata = load_version(args.models, args.version)
        print(json.dumps({key: value for key, value in metadata.items() if key != "training_keys"}, indent=2))
    else:
        if not args.version:
            parser.error("approval and rollback require a version")
        approve(args.models, args.version)


if __name__ == "__main__":
    main()
