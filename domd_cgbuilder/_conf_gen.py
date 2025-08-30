import networkx as nx
import numba as nb
import numpy as np
from typing import Union,List
import random
import time
from misc.logger import logger
from domd_cgbuilder.cg_mol import CGMol


@nb.jit(nopython=True, nogil=True)
def _cell_w(x, box, ib):
    ret = 0
    n_cell = 1
    for i in range(0, x.shape[0]):
        tmp = x[i] / box[i]
        if tmp < -0.5 or tmp > 0.5:
            return -1
        ret = ret + np.floor((tmp + 0.5) * ib[i]) * n_cell
        n_cell = n_cell * ib[i]
    return ret


@nb.jit(nopython=True, nogil=True)
def _cell_i(ix, ib):
    ret = 0
    n_cell = 1
    for i in range(0, ix.shape[0]):
        tmp = (ix[i] + ib[i]) % ib[i]
        ret = ret + tmp * n_cell
        n_cell = n_cell * ib[i]
    return ret


@nb.jit(nopython=True, nogil=True)
def pbc_dist(r, d):
    return np.sqrt(np.sum(r - d * np.rint(r / d)) ** 2)


@nb.jit(nopython=True, nogil=True)
def pbc(r, d):
    return r - d * np.rint(r / d)


class CellList(object):
    def __init__(self, box, rc):
        self.box = box
        self.rc = rc
        self.cells = {}
        self.cell_map = {}
        self.ib = np.asarray(box / rc, np.int64)
        self.dim = (self.box.shape[0],) * 3
        print("Building cell list...")
        self._build_cell_map()
        print("Build cell list")

    def _w_cell(self, x):
        return _cell_w(x, self.box, self.ib)

    def _i_cell(self, ix):
        return _cell_i(ix, self.ib)

    def _build_cell_map(self):
        for ix in np.ndindex(tuple(self.ib)):
            ic = self._i_cell(np.asarray(ix))
            self.cells[ic] = []
            self.cell_map[ic] = {}
            for j in range(self.box.shape[0] ** 3):
                jc = np.asarray(np.unravel_index(j, self.dim)) - 1
                self.cell_map[ic][j] = self._i_cell(jc + ic)

    def add_x(self, x):
        ic = self._w_cell(x)
        self.cells[ic].append(x)

    def iter_neighbors(self, x):
        ic = self._w_cell(x)
        for jc in self.cell_map[ic]:  # include self-cell
            x_in_jc = self.cells[jc]
            for xj in x_in_jc:
                yield xj


def is_valid(x, cl, r=0.2):
    failed_p = False
    for nei in cl.iter_neighbors(x):
        if pbc_dist(x - nei) < r:
            failed_p = True
            return failed_p
    return failed_p


@nb.jit(nopython=True, nogil=True)
def _step_w(r, d=3):
    step = np.random.random(d) - 0.5
    step = step / np.sum(step ** 2) ** 0.5 * r
    return step

def wcell(r,boxl,ixyz):
    ir = np.floor((r/boxl + 0.5 ) * ixyz)
    return int(ir[0]) + int(ir[1]) * ixyz[0] + int(ir[2]) * ixyz[1] * ixyz[0]

def icell(ix,iy,iz,ib):
    return int((ix+ib[0]) % ib[0] + (iy + ib[1]) % ib[1] * ib[0] + (iz + ib[2]) % ib[2] * ib[0] * ib[1])

def build_box_map(boxl,r_cut):
    ib = np.ceil(boxl/r_cut)
    ib = ib.astype(int)
    bs = boxl/ib
    cell_nei_map = {}
    cell_pos_map = {}
    for iz in range(int(ib[2])):
        for iy in range(int(ib[1])):
            for ix in range(int(ib[0])):
                cell_id = icell(ix,iy,iz,ib)
                cell_pos_map[cell_id] = []
                cell_nei_map[cell_id] = []
                for i in range(-1,2):
                    for j in range(-1,2):
                        for k in range(-1,2):
                            cell_nei_map[cell_id].append(icell(ix+i,iy+j,iz+k,ib))
    logger.info(f'Successfully create {max(cell_pos_map.keys())} cells in the box!')
    return cell_nei_map,cell_pos_map,ib,bs

def dis(r1,r2,box):
    r = np.sum((pbc(r1-r2,box))**2)**0.5
    return r

def check_override(p,cell_id,cell_nei_map,cell_pos_map,check,box):
    for ibox in cell_nei_map[cell_id]:
        for pos in cell_pos_map[ibox]:
            if dis(p,pos,box) <= check:
                return 1
    return 0

def dfs(G:nx.DiGraph,start=0,visited=None):
    if visited is None:
        visited = []
        visited.append(start)
    for nei in G.neighbors(start):
        if nei not in visited:
            visited.append(nei)
            dfs(G,nei,visited)
    return visited


def rand():
    return random.random() - 0.5


def randw(wl):
    r = np.array([rand(), rand(), rand()])
    return r * wl / (r ** 2).sum(axis=0) ** 0.5

def findorder(cg_reactants_list,cg_reactants):
    ti, tj = cg_reactants
    if f'{ti}-{tj}' in cg_reactants_list:
        return (ti,tj)
    elif f'{tj}-{ti}' in cg_reactants_list:
        return (tj,ti)
    else:
        logger.error("conect %s-%s is not deffined in the topology" % (ti,tj))
        raise 'fuck'


def embed_CG_system(system: list[CGMol], box, FFPara, rc=1, rcut=0.5):
    r'''
    Generate arbitary polymer system based on self-avoiding walk, the top of the system is reference to any graph structure representation of molecules. It may contains a problem of Brownian bridge.
    :param systm: the list of CGMol
    :param box: the period boundary condition box
    :param rc: the limit dictance of each bead
    '''
    res = build_box_map(box,rc)
    cell_nei_map = res[0]
    cell_pos_map = res[1]
    ib = res[2]
    bs = res[3]
    check = rcut
    binfo = FFPara[2]
    allbt = [bt for bt in binfo]
    nmol = len(system)
    for i,sub in enumerate(system):
        start = time.time()
        for node in sub.nodes:
            if sub.degree(node) == 1:
                start = node
                break
        path = dfs(sub,start=start)
        head=[]
        vpath = set()
        for node in list(path):
            flag = 1
            neis = list(sub.neighbors(node))
            while flag:
                if len(set(neis).intersection(vpath)) == 0:
                    TP = np.array([rand(), rand(), rand()]) * box
                    CM = TP
                else:
                    for nei in neis:
                        if nei in vpath:
                            TP = sub.nodes[nei]['x']
                            last = nei
                            break
                    order = findorder(allbt, [sub.nodes[node]['type'], sub.nodes[last]['type']])
                    wl = binfo[f'{order[0]}-{order[1]}'][0]
                    if wl <= check:
                        wl = check #* 1.05
                    CM = pbc(randw(wl) + TP, box)
                cell_id = wcell(CM, box, ib)
                if check_override(CM, cell_id, cell_nei_map, cell_pos_map, check, box):
                    continue
                flag = 0
            sub.nodes[node]['x'] = CM
            sub.nodes[node]['v'] = np.array([0,0,0])
            vpath.add(node)
            cell_pos_map[cell_id].append(CM)
            n_ = 0
        gen_time = time.time()
        t = gen_time - start
        logger.info(f'Successfully generate molecules {i+1}/{nmol}.')
    return system


def embed_system(system: list[CGMol], box, rc=1):
    r"""Arbitrary configuration generator based on self-avoiding random walk method
    The system contains a series of molecules, i.e., nx.Graph objects
    Random walk coordinates are generated based on edges from edge_dfs, for each molecule,
    a node with given coordinate is treated as the source node for the dfs transversal.
    """
    cl = CellList(box, rc)
    print(cl)
    for mol in system:
        s_node = None
        for bead in mol.nodes:
            if mol.nodes[bead]['x'] is not None:
                s_node = bead
        if s_node is not None:
            cl.add_x(mol.nodes[s_node]['x'])

    # 1st: Add all possible source node coordinates in the cell-list
    # the source node choice algorithm is as same as below, making sure
    # that even multiple nodes has given position, only the last is chosen

    for mol in system:
        s_node = None
        for bead in mol.nodes:
            if mol.nodes[bead]['x'] is not None:
                s_node = bead
        # the source bead is the bead with given coordinate
        # only one for each molecule, for Brownian bridge is
        # too f**king difficult for generating arbitrary graph:
        # The Brownian bridge can be approximated with a center-bias
        # method followed by a spring relaxation
        # for example, 1-2-3-4-5 with anchored 1, 3, 5, when generate 2 from 1
        # (or from 3), the random vector of 1-2 should take a bias from the position
        # of 3 (i.e. the 1-2 towards 3) to ensure a "not too diverged" 2-3
        for edge in nx.edge_dfs(mol, source=s_node):
            ip, jp = edge
            if mol.nodes[ip]['x'] is None:
                xi = (np.random.random() - 0.5) * (box - 0.2)  # tolerance
                while not is_valid(xi, cl):
                    xi = (np.random.random() - 0.5) * (box - 0.2)
                cl.add_x(xi)
            else:
                # since xi is xj in the precursor edge, this should not be problem
                # even if the node has a given coordinate, it has been overwritten
                # by the precursor generation
                xi = mol.nodes[ip]['x']

            step_r = mol.edges[(ip,jp)]['r0']#mol.nodes[ip]['r'] + mol.nodes[jp]['r']
            print(step_r)
            step_i = _step_w(step_r, d=box.shape[0])
            xj_raw = xi + step_i
            xj_img = pbc(xj_raw, box)
            # whether xj is valid
            while not is_valid(xj_img, cl):
                step_i = _step_w(step_r, d=box.shape[0])
                xj_raw = xi + step_i
                xj_img = pbc(xj_raw, box)
            mol.nodes[jp]['x'] = xj_img
            mol.nodes[jp]['img'] = np.asarray(xj_raw / (box / 2), dtype=np.int64)
            cl.add_x(xj_img)
    return system
