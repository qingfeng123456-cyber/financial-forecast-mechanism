from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".zip", ".log"}
MAX_FILE_BYTES = 90 * 1024 * 1024
REQUIRED_MODEL_FILES = (
    "model_code/c3r/main.py",
    "model_code/c3r/src/c3rmoe/c3r.py",
    "model_code/c3r/src/c3rmoe/pipeline.py",
    "model_code/darc/main.py",
    "model_code/darc/src/darc/council.py",
    "model_code/darc/src/darc/pipeline.py",
    "model_code/care/main.py",
    "model_code/care/src/care/models.py",
    "model_code/care/src/care/pipeline.py",
    "model_code/mosaic/main.py",
    "model_code/mosaic/src/mosaic/models.py",
    "model_code/mosaic/src/mosaic/pipeline.py",
    "model_code/dashv/main.py",
    "model_code/dashv/src/dashv/aggregate.py",
    "model_code/dashv/src/dashv/pipeline.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    errors: list[str] = []
    for relative in REQUIRED_MODEL_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"required model source missing: {relative}")
    python_files = sorted(ROOT.rglob("*.py"))
    json_files = sorted(ROOT.rglob("*.json"))
    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            errors.append(f"python parse: {path.relative_to(ROOT)}: {exc}")
    for path in json_files:
        if path.name == "RELEASE_MANIFEST.json":
            continue
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"json parse: {path.relative_to(ROOT)}: {exc}")
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"excluded artifact committed: {path.relative_to(ROOT)}")
        if path.stat().st_size > MAX_FILE_BYTES:
            errors.append(f"file exceeds 90 MiB: {path.relative_to(ROOT)}")

    manifest_path = ROOT / "RELEASE_MANIFEST.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for item in manifest["files"]:
            path = ROOT / item["path"]
            if not path.exists():
                errors.append(f"manifest file missing: {item['path']}")
            elif sha256(path) != item["sha256"]:
                errors.append(f"manifest hash mismatch: {item['path']}")

    report = {
        "python_files_parsed": len(python_files),
        "json_files_parsed": len(json_files),
        "required_model_files": len(REQUIRED_MODEL_FILES),
        "errors": errors,
        "runtime_checks_performed": False,
    }
    print(json.dumps(report, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
