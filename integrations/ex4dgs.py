"""Strict boundary adapter for an optional pinned Ex4DGS checkout."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from gspim.layouts.ex4dgs import Ex4DGSLayout

from .manifest import EX4DGS


@dataclass(frozen=True)
class Ex4DGSCheckout:
    root: Path
    commit: str


def verify_checkout(root: Path) -> Ex4DGSCheckout:
    """Validate that an opt-in checkout is the manifest revision, not merely a directory."""

    if not (root / ".git").exists():
        raise FileNotFoundError(f"not an Ex4DGS checkout: {root}")
    completed = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ValueError(f"cannot read Ex4DGS checkout revision: {completed.stderr.strip()}")
    commit = completed.stdout.strip()
    if commit != EX4DGS.commit:
        raise ValueError(f"Ex4DGS checkout revision is {commit}, expected {EX4DGS.commit}")
    return Ex4DGSCheckout(root, commit)


def layout_from_normalized_records(records: Iterable[Mapping[str, object]]) -> Ex4DGSLayout:
    """Load a fully specified normalized export without guessing checkpoint fields."""

    required = ("primitive_id", "is_static", "keyframes", "attribute_bytes", "die", "bank", "block", "slot")
    normalized: list[dict[str, object]] = []
    for record in records:
        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"normalized Ex4DGS record is missing fields: {', '.join(missing)}")
        keyframes: list[dict[str, object]] = []
        for frame in record["keyframes"]:  # type: ignore[index]
            if not isinstance(frame, Mapping) or set(("timestamp", "mean", "covariance", "opacity", "color")) - set(frame):
                raise ValueError("each normalized Ex4DGS keyframe needs timestamp, mean, covariance, opacity, and color")
            mean = tuple(float(value) for value in frame["mean"])  # type: ignore[arg-type]
            if len(mean) != 3:
                raise ValueError("normalized Ex4DGS keyframe mean must have three values")
            keyframes.append({"timestamp": float(frame["timestamp"]), "mean": mean, "covariance": frame["covariance"], "opacity": frame["opacity"], "color": frame["color"]})
        normalized.append({
            "primitive_id": record["primitive_id"], "is_static": record["is_static"], "keyframes": tuple(keyframes),
            "attribute_bytes": record["attribute_bytes"], "die": record["die"], "bank": record["bank"],
            "block": record["block"], "slot": record["slot"],
        })
    return Ex4DGSLayout.from_mappings(normalized)
