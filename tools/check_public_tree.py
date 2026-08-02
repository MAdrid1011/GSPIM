"""Release gate for the public GSPIM mechanism artifact."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_IGNORES = {
    "/hpca2027-latex-template/", "/GCC.pdf", "/REACT3D.pdf",
    "/#813 - ASPLOS'27 - April cycle.pdf", "/baselines/", "/third_party/",
    "/submodules/", "/out/", "/rtl/verilog/", "/hardware/generated/",
}
LOCAL_ONLY_ROOTS = {
    "baselines", "third_party", "submodules", "out", "rtl/verilog", "hardware/generated",
}
FORBIDDEN_PATHS = {
    ".gitmodules", "benchmark_gspim.py", "render_gspim.py", "run.sh", "train.py",
    "cuda_extensions", "gspim_cuda", "gspim/pim_filter.py", "gspim/ppim_backend.py",
    "gspim/ablation.py", "gspim/profiler.py", "gspim/layouts/fourdgs1k.py",
    "arguments", "configs", "gaussian_renderer", "lpipsPyTorch", "pointops2", "scene", "utils",
    "scripts/n3v2blender.py", "environment.yml",
}
REQUIRED_PATHS = {
    "README.md", "docs/paper-contract.md", "docs/traceability.md", "docs/public-file-map.md", "docs/artifact-scope.md", "docs/workloads.md", "docs/runtime.md",
    "gspim/model.py", "gspim/ppim.py", "gspim/hcf.py", "gspim/dataflow.py", "gspim/renderer.py", "gspim/runtime.py", "gspim/scheduling.py", "gspim/pipeline.py", "gspim/layouts/explicit4d.py", "gspim/layouts/ex4dgs.py", "gspim/layouts/anchored.py",
    "reference_model/pipeline.py", "tests/golden/explicit4d_window.json", "tests/golden/ex4dgs_window.json", "tests/golden/anchored_window.json", "tests/golden/explicit4d_six_frame_sequence.json", "tests/test_ppim.py", "tests/test_hcf.py", "tests/test_dataflow.py", "tests/test_renderer.py", "tests/test_runtime.py", "tests/test_pipeline.py", "tests/test_layouts.py", "tests/test_integrations.py",
    "hardware/src/main/scala/gspim/PPIMDatapath.scala", "hardware/src/main/scala/gspim/Hcf.scala", "hardware/src/main/scala/gspim/LpddrTaskAdapter.scala", "hardware/src/main/scala/gspim/TaskController.scala", "hardware/src/main/scala/gspim/GspimRank.scala", "hardware/src/test/scala/gspim/PpimSpec.scala", "hardware/src/test/scala/gspim/HcfSpec.scala", "hardware/src/test/scala/gspim/LpddrTaskAdapterSpec.scala", "hardware/src/test/scala/gspim/TaskControllerSpec.scala", "hardware/src/test/scala/gspim/GeneratedRtlSpec.scala", "hardware/verify-rtl.sh",
}
PRIVATE_ROOTS = {"hpca2027-latex-template", "GCC.pdf", "REACT3D.pdf", "#813 - ASPLOS'27 - April cycle.pdf"}


def tracked_paths() -> set[str]:
    completed = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True)
    return set(completed.stdout.splitlines())


def tracks_root(tracked: set[str], root: str) -> bool:
    """Return whether a local-only root would enter the public release."""

    return any(path == root or path.startswith(f"{root}/") for path in tracked)


def traceability_paths(text: str) -> set[str]:
    return set(re.findall(r"`((?:docs|gspim|hardware|tests|tools|integrations|reference_model|scripts)/[^` ]+\.(?:md|py|scala|sh|json))`", text))


def mapped_paths(text: str) -> set[str]:
    """Extract the literal public paths owned by the file-level contract."""

    return set(re.findall(r"`((?:\.gitignore|README\.md|THIRD_PARTY_NOTICES\.md|pyproject\.toml|(?:docs|gspim|hardware|tests|tools|integrations|reference_model|scripts)/[^` ]+))`", text))


def public_source_paths(tracked: set[str]) -> set[str]:
    """Files that need a paper/scope entry before a release can be trusted."""

    roots = ("docs/", "gspim/", "hardware/", "tests/", "tools/", "integrations/", "reference_model/", "scripts/")
    suffixes = (".md", ".py", ".scala", ".sh", ".json", ".sbt", ".properties")
    root_files = {".gitignore", "README.md", "THIRD_PARTY_NOTICES.md", "pyproject.toml"}
    return {
        path for path in tracked
        if path in root_files or (path.startswith(roots) and path.endswith(suffixes))
    }


def non_ascii_paths(tracked: set[str]) -> set[str]:
    """Return tracked files that violate the public ASCII-English policy."""

    return {
        path for path in tracked
        if any(byte > 0x7F for byte in (ROOT / path).read_bytes())
    }


def main() -> int:
    failures: list[str] = []
    ignores = (ROOT / ".gitignore").read_text(encoding="ascii")
    tracked = tracked_paths()
    for pattern in sorted(REQUIRED_IGNORES):
        if pattern not in ignores:
            failures.append(f"missing ignore rule: {pattern}")
    private = {path for path in tracked if any(path == root or path.startswith(f"{root}/") for root in PRIVATE_ROOTS)}
    if private:
        failures.append(f"private paper artifact is tracked: {sorted(private)}")
    for root in sorted(LOCAL_ONLY_ROOTS):
        if tracks_root(tracked, root):
            failures.append(f"local-only path is tracked: {root}")
    for path in sorted(FORBIDDEN_PATHS):
        if (ROOT / path).exists() or path in tracked:
            failures.append(f"legacy public path remains: {path}")
    for path in sorted(REQUIRED_PATHS):
        if not (ROOT / path).exists():
            failures.append(f"required mechanism artifact is missing: {path}")
        elif path not in tracked:
            failures.append(f"required mechanism artifact is not tracked: {path}")
    traceability = (ROOT / "docs/traceability.md").read_text(encoding="ascii")
    for mechanism in ("S1", "S2", "S3", "S4", "S5", "PIM_SELECT", "PIM_REORG", "TEMP_ACTIVITY", "ANCHOR_OVERLAP"):
        if mechanism not in traceability and mechanism not in (ROOT / "docs/paper-contract.md").read_text(encoding="ascii"):
            failures.append(f"traceability contract omits {mechanism}")
    for path in sorted(traceability_paths(traceability)):
        if not (ROOT / path).exists():
            failures.append(f"traceability path is missing: {path}")
        elif path not in tracked:
            failures.append(f"traceability path is not tracked: {path}")
    file_map = (ROOT / "docs/public-file-map.md").read_text(encoding="ascii")
    mapped = mapped_paths(file_map)
    public = public_source_paths(tracked)
    for path in sorted(public - mapped):
        failures.append(f"public source is missing a file-map entry: {path}")
    for path in sorted(mapped - public):
        failures.append(f"file-map entry is absent or not a tracked public source: {path}")
    for path in sorted(non_ascii_paths(tracked)):
        failures.append(f"public tracked file is not ASCII English: {path}")
    if failures:
        print("public-tree check failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print("public-tree check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
