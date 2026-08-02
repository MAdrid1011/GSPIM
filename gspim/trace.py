"""Dependency-only task trace with duplicate and missing-edge protection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class TraceNode:
    name: str
    kind: str
    dependencies: tuple[str, ...] = ()


@dataclass
class TaskTrace:
    nodes: dict[str, TraceNode] = field(default_factory=dict)

    def add(self, name: str, kind: str = "task", dependencies: Iterable[str] = ()) -> None:
        dependencies_tuple = tuple(dependencies)
        if name in self.nodes:
            raise ValueError(f"duplicate trace node: {name}")
        if len(dependencies_tuple) != len(set(dependencies_tuple)):
            raise ValueError(f"duplicate trace dependency for {name}")
        self.nodes[name] = TraceNode(name, kind, dependencies_tuple)

    def has_dependency(self, parent: str, child: str) -> bool:
        return child in self.nodes and parent in self.nodes[child].dependencies

    def to_dict(self) -> dict[str, object]:
        return {"nodes": [{"name": node.name, "kind": node.kind, "depends_on": list(node.dependencies)} for node in self.nodes.values()]}
