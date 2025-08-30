from itertools import combinations

import networkx as nx


class CGMol(nx.Graph):

    def __init__(self, incoming_graph_data=None, **attr):
        super().__init__(incoming_graph_data, **attr)
        self._hyperedges: dict[int, dict[tuple, dict]] = {}

    def add_atom(self, bead_id, type, coords=None, radius=0.5, **extra):
        data = {'type': type, 'x': coords, 'r': radius}
        data.update(extra)
        self.add_node(bead_id, **data)



    def add_bond(self, u, v, order=1, bt='bond',k=1100, r0=1,**extra):
        data = {'order': order, 'k':1100, 'r0':1}
        data.update(extra)
        self.add_edge(u, v, **data)

    def add_hyperedge(self, nodes: tuple, k: int, **attr):
        if len(nodes) != k:
            raise ValueError(f"nodes tuple length must be {k}, got {len(nodes)}")
        self._hyperedges.setdefault(k, {})
        self._hyperedges[k][nodes] = dict(attr)

    def remove_hyperedge(self, nodes: tuple):
        k = len(nodes)
        if k in self._hyperedges and nodes in self._hyperedges[k]:
            del self._hyperedges[k][nodes]

    def iter_hyperedges(self, k: int):
        for nodes, attr in self._hyperedges.get(k, {}).items():
            yield nodes, attr

    def get_hyperedge_attrs(self, nodes: tuple):
        return self._hyperedges.get(len(nodes), {}).get(nodes)

    def add_angle(self, i, j, k, bt='angle', **attr):
        self.add_hyperedge((i, j, k), 3, **attr)
        self._hyperedges[3][(i, j, k)]['bt'] = bt

    def add_dihedral(self, i, j, k, l, bt='dih', **attr):
        self.add_hyperedge((i, j, k, l), 4, **attr)
        self._hyperedges[4][(i, j, k, l)]['bt'] = bt

    def iter_angles(self):
        return self.iter_hyperedges(3)

    def iter_dihedrals(self):
        return self.iter_hyperedges(4)

    def auto_angles(self):
        for j in self.nodes:
            nbrs = list(self.adj[j])
            for i, k in combinations(nbrs, 2):
                tpl = (i, j, k)
                if tpl not in self._hyperedges.get(3, {}):
                    self.add_angle(i, j, k)

    def auto_dihedrals(self):
        for edge in self.edges:
            j, k = edge
            for i in self.adj[j]:
                if i == k: continue
                for l in self.adj[k]:
                    if l in (j, i): continue
                    tpl = (i, j, k, l)
                    if tpl not in self._hyperedges.get(4, {}):
                        self.add_dihedral(i, j, k, l)
