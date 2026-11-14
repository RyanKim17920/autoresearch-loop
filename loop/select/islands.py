"""Islands by mechanism family (D4.3), off until ``split_at`` scored nodes.

Migration is a parent pool: a confirmation joins every other island's frontier. Every
``reset_every`` confirmations the weakest island (by its OWN confirmed wins) restarts
from the global incumbent. State lives under one store key."""

from __future__ import annotations

from typing import Any

from loop.contracts import DESIGN_FAMILY, ROOT_NODE_ID, Node, NodeStatus, WorldSnapshot
from loop.spec import LoopConfig
from loop.select.thompson import utility

STATE_KEY = "select.islands"


class IslandBook:
    def __init__(self, cfg: LoopConfig, store: Any) -> None:
        self.cfg, self.icfg, self.store = cfg, cfg.islands, store
        st = store.get_state(STATE_KEY) or {}
        self.assign: dict[str, str] = dict(st.get("assign", {}))
        self.n_islands = int(st.get("n_islands", 0))
        self.migrants: dict[str, list[str]] = {k: list(v) for k, v in st.get("migrants", {}).items()}
        self.resets: dict[str, dict] = dict(st.get("resets", {}))      # island -> {parent, excluded}
        self.last_reset_at = int(st.get("last_reset_at", 0))

    def _save(self) -> None:
        self.store.set_state(STATE_KEY, {"assign": self.assign, "n_islands": self.n_islands,
                                         "migrants": self.migrants, "resets": self.resets,
                                         "last_reset_at": self.last_reset_at})

    @property
    def split(self) -> bool:
        return self.n_islands > 0

    @property
    def islands(self) -> list[str]:
        return [f"i{k}" for k in range(self.n_islands)] if self.split else ["main"]

    def _counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for n in self.store.nodes():
            if n.id != ROOT_NODE_ID and n.status is not NodeStatus.REJECTED and n.family != DESIGN_FAMILY:
                out[n.family] = out.get(n.family, 0) + 1
        return out

    def maybe_split(self, snapshot: WorldSnapshot) -> None:
        """Greedy balanced family partition once ``split_at`` nodes are scored."""
        if self.split or snapshot.n_scored < self.icfg.split_at:
            return
        counts = self._counts()
        fams = sorted((set(counts) | set(self.cfg.families)) - {""})
        if min(self.icfg.n, len(fams)) < 2:
            return
        self.n_islands = min(self.icfg.n, len(fams))
        for f in sorted(fams, key=lambda f: (-counts.get(f, 0), f)):
            self.assign[f] = self._smallest(counts)
        self._save()

    def _smallest(self, counts: dict[str, int]) -> str:
        load = {i: 0 for i in self.islands}
        for f, i in self.assign.items():
            load[i] += counts.get(f, 0)
        return min(self.islands, key=lambda i: (load[i], i))

    def island_of(self, family: str) -> str:
        """A NEW family joins the smallest island (persisted)."""
        if not self.split:
            return "main"
        if family not in self.assign:
            self.assign[family] = self._smallest(self._counts())
            self._save()
        return self.assign[family]

    def families_of(self, island: str) -> list[str]:
        known = sorted((set(self.cfg.families) | set(self._counts())) - {"", DESIGN_FAMILY})
        return [f for f in known if self.island_of(f) == island]

    def confirmed(self, island: str | None = None) -> list[Node]:
        return [n for n in self.store.nodes(status=[NodeStatus.CONFIRMED])
                if island is None or self.island_of(n.family) == island]

    def frontier(self, island: str, snapshot: WorldSnapshot) -> list[str]:
        """Own confirmed nodes (minus those before a reset) + migrants + the reset parent,
        minus vetoed; root when nothing else."""
        reset = self.resets.get(island) or {"parent": None, "excluded": []}
        ids = [n.id for n in self.confirmed(island) if n.id not in reset["excluded"]]
        ids += self.migrants.get(island, []) + ([reset["parent"]] if reset["parent"] else [])
        ids = [i for i in dict.fromkeys(ids) if not (i in snapshot.nodes and snapshot.nodes[i].vetoed)]
        return ids or [ROOT_NODE_ID]

    def best(self, island: str, snapshot: WorldSnapshot) -> str:
        return max(self.frontier(island, snapshot),
                   key=lambda i: (utility(snapshot, i).mean, i != ROOT_NODE_ID))

    def on_confirmed(self, node: Node, snapshot: WorldSnapshot) -> None:
        self.maybe_split(snapshot)
        if not self.split:
            return
        home = self.island_of(node.family)
        for isl in self.islands:
            if isl != home and node.id not in self.migrants.get(isl, []):
                self.migrants.setdefault(isl, []).append(node.id)
        n_conf = len(self.confirmed())
        if n_conf - self.last_reset_at >= self.icfg.reset_every:
            self.last_reset_at = n_conf
            own = {i: max([utility(snapshot, n.id).mean for n in self.confirmed(i)], default=0.0)
                   for i in self.islands}
            weakest = min(self.islands, key=lambda i: (own[i], i))
            if own[weakest] < utility(snapshot, snapshot.incumbent_id).mean:
                self.resets[weakest] = {"parent": snapshot.incumbent_id,
                                        "excluded": [n.id for n in self.confirmed(weakest)]}
                self.migrants[weakest] = []
        self._save()
