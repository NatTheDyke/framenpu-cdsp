"""Offline publication checks: paths, secrets, docs, payload provenance, examples."""

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from framenpu_cdsp import Job, Model, __version__
from framenpu_cdsp.provenance import verify_kernel

PUBLIC_DIRS = (
    "src",
    "docs",
    "tools",
    "tests",
    "examples",
    "evidence",
    ".github",
    "handoff-tools",
    "integrations",
)
IGNORED_PARTS = {"__pycache__", ".ruff_cache", ".pytest_cache", "build", "dist"}
BINARY_FILES = {
    "src/framenpu_cdsp/data/runtime.bin",
    "src/framenpu_cdsp/data/dispatch.bin",
    "src/framenpu_cdsp/data/dispatch_patch.bin",
    "src/framenpu_cdsp/data/layout_aarch64.so",
    "src/framenpu_cdsp/data/mercury/teacher_link.bin",
    "src/framenpu_cdsp/data/mercury/teacher_link_loader.fjob",
    "examples/vector_add.fjob",
    "examples/xor.fnpu",
}


HOST_SOURCES = {"layout": "host/fnpu_layout.c"}


def verify_host_native(package, name="layout"):
    """The bundled layout library matches its manifest, and the manifest its sources."""
    record = json.loads((package / f"data/{name}_aarch64.json").read_text())
    lib = (package / f"data/{name}_aarch64.so").read_bytes()
    if record.get("bytes") != len(lib) or record.get("sha256") != hashlib.sha256(lib).hexdigest():
        raise ValueError(f"bundled {name} library does not match {name}_aarch64.json")
    for source, digest in record.get("source_sha256", {}).items():
        if hashlib.sha256((package / source).read_bytes()).hexdigest() != digest:
            raise ValueError(
                f"{source} changed; run python tools/build_host_native.py --lib {name}"
            )
    if list(record.get("source_sha256", {})) != [HOST_SOURCES[name], "host/fnpu_common.h"]:
        raise ValueError(f"{name} library manifest lists unexpected sources")


def verify_bundled_image(package):
    """The bundled Mercury image matches its link record (the record pins the loader kernel)."""
    folder = package / "data/mercury"
    record = json.loads((folder / "link.json").read_text())
    image = (folder / "teacher_link.bin").read_bytes()
    linked = record.get("linked", {})
    if linked.get("sha256") != hashlib.sha256(image).hexdigest() or linked.get(
        "bytes", len(image)
    ) != len(image):
        raise ValueError("bundled Mercury image does not match data/mercury/link.json")


def public_files(root=ROOT):
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], check=True, capture_output=True
        )
        names = result.stdout.decode().split("\0")
        if any(names):
            return [root / name for name in names if name]
    paths = {p for p in root.iterdir() if p.is_file()}
    for directory in PUBLIC_DIRS:
        paths.update(
            p
            for p in (root / directory).rglob("*")
            if p.is_file()
            and not IGNORED_PARTS.intersection(p.relative_to(root).parts)
            and not any(part.endswith(".egg-info") for part in p.parts)
        )
    return sorted(paths)


def check(root=ROOT):
    root = Path(root)
    required = (
        "LICENSE",
        "README.md",
        "CONTRIBUTING.md",
        "SECURITY.md",
        "THIRD_PARTY_NOTICES.md",
        ".github/workflows/ci.yml",
    )
    for name in required:
        if not (root / name).is_file():
            raise ValueError(f"missing publication file: {name}")
    config = tomllib.loads((root / "pyproject.toml").read_text())
    if config["project"]["version"] != __version__ or config["project"]["license"] != "MIT":
        raise ValueError("package version/license metadata mismatch")
    patterns = {
        "private local path": rb"/home/[a-zA-Z0-9_.-]+",
        "SSH capture": rb"steamos" + rb"@frame|Frame authenti" + rb"cation:",
        "private key": rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "access token": rb"\b(?:gh[pousr]_[a-zA-Z0-9]{20,}|github_pat_[a-zA-Z0-9_]{30,})",
        "credential assignment": rb"(?i)(?:password|api_key|access_token)\s*[:=]\s*['\"][^'\"\n]{8,}['\"]",
    }
    files = public_files(root)
    for path in files:
        name = path.relative_to(root).as_posix()
        if name == "HANDOFF.md" or (
            name.startswith("handoff-tools/") and name != "handoff-tools/build_mw.py"
        ):
            raise ValueError(f"local handoff/recovery helper must not be published: {name}")
        if path.suffix in (".mbn", ".dtb", ".img", ".elf", ".o", ".pcap", ".pcapng"):
            raise ValueError(f"research/firmware artifact must not be published: {name}")
        raw = path.read_bytes()
        for label, pattern in patterns.items():
            if re.search(pattern, raw):
                # Do not echo the offending content.
                raise ValueError(f"{label} found in {name}")
        if path.suffix in (".bin", ".fjob", ".fnpu", ".so") and name not in BINARY_FILES:
            raise ValueError(f"unreviewed binary: {name}")
        if path.suffix == ".md":
            for match in re.finditer(r"\[[^\]]+\]\(([^)\s]+)\)", raw.decode()):
                target = match.group(1).split("#")[0]
                if not target or re.match(r"[a-z]+://|mailto:", target):
                    continue
                resolved = (path.parent / target).resolve()
                if not resolved.is_relative_to(root.resolve()) or not resolved.exists():
                    raise ValueError(
                        f"broken/external local documentation link in {name}: {target}"
                    )
    ci = (root / ".github/workflows/ci.yml").read_text()
    for action in ("checkout", "setup-python"):
        if not re.search(r"actions/" + action + r"@[0-9a-f]{40}", ci):
            raise ValueError(f"CI action must use a verified full revision: {action}")
    if "pull_request_target" in ci:
        raise ValueError("untrusted PR code must not run in a privileged workflow")
    verify_kernel(root / "src/framenpu_cdsp")
    verify_host_native(root / "src/framenpu_cdsp", "layout")
    verify_bundled_image(root / "src/framenpu_cdsp")
    Job.load(root / "examples/vector_add.fjob")
    Model.load(root / "examples/xor.fnpu").plan(4)
    print(
        f"Repository checks passed: {len(files)} public files, MIT metadata, local links, "
        "payload source hashes and example containers"
    )


if __name__ == "__main__":
    try:
        check()
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"Repository check failed: {exc}") from exc
