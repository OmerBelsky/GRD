from __future__ import annotations

from typing import Dict, List, Tuple

try:
    from igraph import Graph  # type: ignore
except ModuleNotFoundError:

    class _VertexSeq:
        def __init__(self, graph: "Graph") -> None:
            self._graph = graph
            self._attrs: Dict[str, List[object]] = {}

        def __setitem__(self, key: str, values: List[object]) -> None:
            if len(values) != self._graph.vcount():
                raise ValueError(f"Attribute '{key}' length must match vertex count")
            self._attrs[key] = list(values)

        def __getitem__(self, key: str) -> List[object]:
            return self._attrs[key]

    class Graph:  # pragma: no cover - used when igraph is unavailable.
        def __init__(self, n: int, edges: List[Tuple[int, int]], directed: bool = True) -> None:
            self._n = n
            self._directed = directed
            self._out: List[List[int]] = [[] for _ in range(n)]
            self._in: List[List[int]] = [[] for _ in range(n)]
            for src, dst in edges:
                self._out[src].append(dst)
                self._in[dst].append(src)
                if not directed:
                    self._out[dst].append(src)
                    self._in[src].append(dst)
            self.vs = _VertexSeq(self)

        def neighbors(self, vid: int, mode: str = "out") -> List[int]:
            if mode == "out":
                return list(self._out[vid])
            if mode == "in":
                return list(self._in[vid])
            raise ValueError(f"Unsupported mode: {mode}")

        def vcount(self) -> int:
            return self._n
