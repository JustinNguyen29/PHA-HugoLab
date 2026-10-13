"""Cell Ontology helpers for scoring predictions against ground truth in CL space.

`load_cl` / `resolve_cl_id` mirror `cell2sentence.ontology_utils` (vendored so this repo does not
import the torch-heavy cell2sentence package). Unlike the C2S helpers, ancestry here uses the full
is_a DAG, not just the primary parent, so multi-parent terms are scored generously and consistently
for every method.
"""
from functools import lru_cache

import networkx as nx
import obonet

CL_ROOT = "CL:0000000"


class CellOntology:
    def __init__(self, obo_path):
        graph = obonet.read_obo(str(obo_path))
        self.name2id, self.syn2id, self.id2name = {}, {}, {}
        isa = nx.DiGraph()  # child -> parent, CL nodes and is_a edges only
        for nid, data in graph.nodes(data=True):
            if not str(nid).startswith("CL:"):
                continue
            isa.add_node(nid)
            name = (data.get("name") or "").strip()
            if name:
                self.id2name[nid] = name
                self.name2id.setdefault(name.lower(), nid)
            for syn in data.get("synonym", []):
                parts = syn.split('"')
                if len(parts) >= 3 and parts[2].strip().split(" ", 1)[0].upper() == "EXACT":
                    self.syn2id.setdefault(parts[1].strip().lower(), nid)
        for u, v, k in graph.edges(keys=True):
            if k == "is_a" and str(u).startswith("CL:") and str(v).startswith("CL:"):
                isa.add_edge(u, v)
        self.isa = isa
        self._undirected = isa.to_undirected(as_view=True)

    def resolve(self, label, overrides=None):
        """Free-text label -> CL id via override, exact name, then exact synonym (case-insensitive)."""
        if overrides and label in overrides:
            return overrides[label]
        key = str(label).strip().lower()
        return self.name2id.get(key) or self.syn2id.get(key)

    def name(self, cid):
        return self.id2name.get(cid, cid)

    @lru_cache(maxsize=None)
    def ancestors(self, cid):
        """All is_a ancestors of cid (excluding cid itself)."""
        if cid not in self.isa:
            return frozenset()
        return frozenset(nx.descendants(self.isa, cid))  # edges point child -> parent

    @lru_cache(maxsize=None)
    def depth(self, cid):
        """Longest is_a path length from cid up to the CL root."""
        if cid not in self.isa or cid == CL_ROOT:
            return 0
        parents = list(self.isa.successors(cid))
        return 1 + max((self.depth(p) for p in parents), default=0)

    @lru_cache(maxsize=None)
    def relation(self, pred, true):
        """How a predicted CL id relates to the true CL id.

        exact      : same term
        ancestor   : prediction is a (coarser) ancestor of the truth  -> correct but less specific
        descendant : prediction is a (finer) descendant of the truth  -> correct and more specific
        wrong      : neither
        none       : prediction missing / unmappable
        """
        if pred is None or pred != pred or pred not in self.isa:
            return "none"
        if pred == true:
            return "exact"
        if pred in self.ancestors(true):
            return "ancestor"
        if true in self.ancestors(pred):
            return "descendant"
        return "wrong"

    @lru_cache(maxsize=None)
    def lca_hops(self, pred, true):
        """Hops between pred and true through their closest common ancestor (0 = exact)."""
        if pred is None or pred != pred or pred not in self.isa or true not in self.isa:
            return None
        if pred == true:
            return 0
        up_p = nx.single_source_shortest_path_length(self.isa, pred)
        up_t = nx.single_source_shortest_path_length(self.isa, true)
        common = set(up_p) & set(up_t)
        return min(up_p[a] + up_t[a] for a in common) if common else None
