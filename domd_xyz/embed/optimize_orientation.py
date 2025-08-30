import time
from collections import namedtuple

import numba as nb
import numpy as np
import torch
from scipy.optimize import minimize
from torch import optim

from misc.logger import logger


def optimization_by_chunk(chunk_per_d, connections, pos, local_frame_idx, trans, box, rot):
    rot = rot.reshape(-1, 3, 3)
    chunk_len = np.ceil(box / chunk_per_d)
    pos = pos + 0.5 * box
    trans = trans + 0.5 * box
    # pos_cell_idx = pos//chunk_len
    cgpos_cell_idx = trans // chunk_len
    cell_idx_set = set([tuple(i) for i in cgpos_cell_idx])
    idx_to_cid = {s: i for i, s in enumerate(cell_idx_set)}
    cid_to_idx = {i: s for i, s in enumerate(cell_idx_set)}
    rid_to_cid = {i: idx_to_cid[tuple(idx)] for i, idx in enumerate(cgpos_cell_idx)}
    cgpos_cell_cid = np.array([rid_to_cid[i] for i in range(len(trans))])
    cid_set = set(cgpos_cell_cid)
    r1 = pos[local_frame_idx.T[0]]
    r2 = pos[local_frame_idx.T[1]]
    r1_cid = cgpos_cell_cid[connections.T[0]]
    r2_cid = cgpos_cell_cid[connections.T[1]]
    cid_hash = {}
    for cid in cid_set:
        hash_ = (r1_cid == cid) + (r2_cid == cid)
        cid_hash[cid] = hash_
    cid_meta = {}
    for cid in cid_hash:
        hash_ = cid_hash[cid]
        connections_ = connections[hash_]
        in_chunk_res_idx = set()
        for i, j in connections_:
            in_chunk_res_idx.add(i)
            in_chunk_res_idx.add(j)
        in_chunk_res_idx = np.sort(list(in_chunk_res_idx))
        local_to_global = {i: gid for i, gid in enumerate(in_chunk_res_idx)}
        global_to_local = {gid: i for i, gid in enumerate(in_chunk_res_idx)}
        in_chunk_connections = np.array([[global_to_local[i], global_to_local[j]] for i, j in connections_])
        cid_meta[cid] = {'connections_': in_chunk_connections, 'local_frame_idx': local_frame_idx[hash_],
                         'n_residue': len(in_chunk_res_idx),
                         'rot': rot[in_chunk_res_idx].ravel(), 'local_to_global': local_to_global,
                         'connections': connections_}
    return cid_set, cid_meta, cid_hash


@nb.jit(nopython=True, nogil=True)
def pbc(r, d):
    return r - d * np.rint(r / d)


def pbc_torch(r, d):
    return r - d * torch.floor(r / d + 0.5)


def rot_cons(rot0):
    rot = rot0.reshape(-1, 3, 3)
    a = np.sum((np.einsum('ikj,ikl->ijl', rot, rot) - np.eye(3)) ** 2)
    return a


def rot_cons_jac(rot0):
    rot = rot0.reshape(-1, 3, 3)
    b = np.einsum('ikj,ikl->ijl', rot, rot) - np.eye(3)
    a = 4 * np.einsum('ijk, ikl->ijl', rot, b).ravel()
    return np.nan_to_num(a, nan=0)  # 4 * np.einsum('ijk, ikl->ijl', rot, b).ravel()


cons = ({'type': 'eq', 'fun': rot_cons, 'jac': rot_cons_jac, },)
# cons = ({'type': 'ineq', 'fun': rot_cons, },)

Meta = namedtuple("Meta", "bonds trans_v local_x atom_pos atom_res_id box")


def optimize_res_orientation(n_residue, meta, chunk_per_d=1):
    r0 = (np.random.normal(0, 0.01, (
            n_residue * 9)))  # np.tile(np.eye(3), (n_residue, 1, 1)).ravel()# + (np.random.normal(0,0.01,(n_residue * 9)) )
    # r0 = np.concatenate((r0,r0),axis=0)
    connections = meta.bonds
    # print(connections,meta.local_x,meta.atom_pos)
    trans = meta.trans_v
    trans_torch = torch.tensor(trans)
    # trans = trans.astype(np.float16)
    local_frame_idx = meta.local_x
    box = meta.box  # * n_residue
    # box = box.astype(np.float16)
    pos = meta.atom_pos
    pos_torch = torch.tensor(meta.atom_pos)
    box_torch = torch.tensor(meta.box)
    # self.post = pos_torch
    # self.boxt = box_torch
    ##self.trant = trans_torch
    if n_residue == 1:
        # if 1:
        return np.array([np.eye(3), ] * n_residue)
    if len(connections) == 0:
        logger.warning('molecules with connections in CG without in aa, check your SMARTS')
        return np.array([np.eye(3), ] * n_residue)
    # numba free
    # @nb.jit(nopython=True, nogil=True)
    # device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda:0')

    def _loss_jac_torch(rot0):
        start = time.time()
        rot = torch.tensor(rot0, requires_grad=True, device=device)
        ri = pbc_torch(
            (torch.einsum('ijk,ik->ij', rot[connections.T[0]], pos_torch[local_frame_idx.T[0]].to(device)) +
             trans_torch[
                 connections.T[0]].to(device)),
            box_torch.to(device)
        )
        rj = pbc_torch(
            (torch.einsum('ijk,ik->ij', rot[connections.T[1]], pos_torch[local_frame_idx.T[1]].to(device)) +
             trans_torch[
                 connections.T[1]].to(device)),
            box_torch.to(device)
        )
        rij = pbc_torch(rj - ri, box_torch.to(device))
        s = torch.sum(rij ** 2)
        s.backward()
        # print(rot)
        # print('JacobiTime',time.time()-start)
        return (rot.grad.detach().cpu().numpy().ravel())

    @nb.njit
    def matrix_multiply(matrices, vectors):
        N = matrices.shape[0]
        results = np.empty((N, 3))  # Initialize an array to hold the results
        for i in range(N):
            results[i] = matrices[i] @ vectors[i]  # Matrix-vector multiplication
        return results

    @nb.jit(nopython=True, nogil=True)
    def _loss_jac(rot):
        ri = (matrix_multiply(rot[connections.T[0]], pos[local_frame_idx.T[0]])).reshape(-1, 3) + trans[
            connections.T[0]]
        rj = (matrix_multiply(rot[connections.T[1]], pos[local_frame_idx.T[1]])).reshape(-1, 3) + trans[
            connections.T[1]]
        rij = pbc(rj - ri, box)
        r0j = pos[local_frame_idx.T[1]]
        grad = np.zeros_like(rot)
        g1_grad = np.zeros_like(rot)
        for i in range(len(rot)):
            g1 = np.sum(
                2 * (rij.reshape(-1, 1, 3) * pos[local_frame_idx.T[1]].reshape(-1, 3, 1))[connections.T[1] == i],
                axis=0)
            g2 = np.sum(
                2 * (rij.reshape(-1, 1, 3) * pos[local_frame_idx.T[0]].reshape(-1, 3, 1))[connections.T[0] == i],
                axis=0)
            # print(g1.shape)
            if g1.shape[0] == 0:
                g1 = np.zeros((3, 3))
            if g2.shape[0] == 0:
                g2 = np.zeros((3, 3))
            # print(g1.shape)
            # rint(g2.shape)
            grad[i] = ((g1 - g2).T)
            g1_grad[i] = g1
            # print(i)
        return grad.ravel()

    def _loss_jac_ana(rot):
        # print('loss_jac')
        rot = rot.reshape(-1, 3, 3)
        ri = (np.einsum('ijk, ipk->ipj', rot[connections_.T[0]], pos[local_frame_idx.T[0]].reshape(-1, 1, 3)) + trans[
            connections.T[0]].reshape(-1, 1, 3))
        rj = (np.einsum('ijk, ipk->ipj', rot[connections_.T[1]], pos[local_frame_idx.T[1]].reshape(-1, 1, 3)) + trans[
            connections.T[1]].reshape(-1, 1, 3))
        rij = pbc(rj - ri, box)
        g1 = np.zeros_like(rot).reshape(-1, 1, 3, 3)
        g2 = np.zeros_like(rot).reshape(-1, 1, 3, 3)
        np.add.at(g1, connections_.T[1],
                  2 * np.einsum('ijk,ijl->ijkl', rij, pos[local_frame_idx.T[1]].reshape(-1, 1, 3)))
        np.add.at(g2, connections_.T[0],
                  2 * np.einsum('ijk,ijl->ijkl', rij, pos[local_frame_idx.T[0]].reshape(-1, 1, 3)))
        a = g1 - g2
        return a.ravel()

    def _loss(rot):
        rot = rot.reshape(n_residue, 3, 3)
        ri = (np.einsum('ijk, ipk->ipj', rot[connections_.T[0]], pos[local_frame_idx.T[0]].reshape(-1, 1, 3)) + trans[
            connections.T[0]].reshape(-1, 1, 3))
        rj = (np.einsum('ijk, ipk->ipj', rot[connections_.T[1]], pos[local_frame_idx.T[1]].reshape(-1, 1, 3)) + trans[
            connections.T[1]].reshape(-1, 1, 3))
        rij = pbc(rj - ri, box)
        s = np.sum(rij ** 2)
        jac = _loss_jac_ana(rot)
        return s, jac

    def lfn(rot, pos, trans, box):
        ri = torch.einsum('ijk,ik->ij', rot[connections.T[0]], pos[local_frame_idx.T[0]]) + trans[connections.T[0]]
        rj = torch.einsum('ijk,ik->ij', rot[connections.T[1]], pos[local_frame_idx.T[1]]) + trans[connections.T[1]]
        rij = pbc_torch(rj - ri, box)
        s = torch.sum(rij ** 2)
        return s

    def Rot_cons(rot, eye):
        return torch.sum((torch.det(rot) - 1) ** 2)

    def torch_opt(rot, lam, rho, pos_torch, trans_torch, box_torch, maxiter=500):
        device = torch.device(1)
        rot = torch.tensor(rot.reshape(n_residue, 3, 3), requires_grad=True, device=device)
        lam = torch.tensor(lam, device=device)
        rho = torch.tensor(rho, device=device)
        done = False
        pos_torch = pos_torch.to(device)
        trans_torch = trans_torch.to(device)
        box_torch = box_torch.to(device)
        eye = torch.eye(3).to(device)
        inner_loop = 100
        outer_loop = maxiter
        for i in range(outer_loop):
            optimizerR = optim.Adam([rot], lr=1e-6)
            rot_ = rot.clone().detach()
            for j in range(inner_loop):
                optimizerR.zero_grad()
                if done:
                    break
                f = lfn(rot, pos_torch, trans_torch, box_torch)
                p = lam * Rot_cons(rot, eye) + 0.5 * rho * Rot_cons(rot, eye) ** 2
                loss = f + p
                loss.backward()
                optimizerR.step()
                logger.info(
                    f'{torch.abs(rot.grad).sum().item():.6f} cons loss: {Rot_cons(rot, eye).item()}, lam: {lam.item()}')
            constraint_val = Rot_cons(rot, eye).item()
            with torch.no_grad():
                lam += rho * constraint_val
            if lam > 1e9:
                rho *= 1.0001
            else:
                rho *= 1.5
            if torch.mean(torch.abs(rot.grad)) < 1e-2 and torch.mean(
                    torch.abs(rot_ - rot)) < 1e-4 and constraint_val < 5e-7:
                done = True
            if i % 2 == 0:
                logger.info(f'rot loss: {torch.mean(torch.abs(rot_ - rot))}, cons loss: {constraint_val}, lam: {lam}')
            if done:
                break
        return rot.detach().cpu().numpy().ravel(), lam, rho, done

    maxiter, max_count, count = 500, 5, 0
    if chunk_per_d <= 1:
        maxiter, max_count, count = 500, 5, 0
        connections_ = connections
        res = minimize(_loss, r0, constraints=cons, options={'maxiter': maxiter, 'disp': True}, jac=True,
                       method='trust-constr')
        while (not res.success) and (count < max_count):
            maxiter *= 2
            logger.warning(f"Minimization failed, try increasing maxiter to {maxiter}")
            r0 = res.x
            res = minimize(_loss, r0, constraints=cons, options={'maxiter': maxiter, 'disp': True}, jac=True,
                           method='trust-constr')
            count += 1
        if count >= max_count:
            logger.error("Orientation minimization failed, use the lasted rotation matrix instead.")
            return res.x.reshape(-1, 3, 3)  # np.array([np.eye(3), ] * n_residue)
        return res.x.reshape(-1, 3, 3)
    elif chunk_per_d > 1:
        logger.info(f"Optimizing orentation by chunk with {chunk_per_d} chunks in each dimension.")
        cid_set, cid_meta, cid_hash = optimization_by_chunk(chunk_per_d, connections, pos, local_frame_idx, trans, box,
                                                            r0)
        maxiter, max_count, count = 100, 5, 0
        rot_ = np.array([np.eye(3), ] * n_residue)
        for i, cid in enumerate(cid_set):
            logger.info(f"Optimizing orentation chunk {i + 1}/{len(cid_set)}.")
            # print(cid)
            r0 = cid_meta[cid]['rot']
            connections = cid_meta[cid]['connections']
            connections_ = cid_meta[cid]['connections_']
            local_frame_idx = cid_meta[cid]['local_frame_idx']
            local_to_global = cid_meta[cid]['local_to_global']
            # print(connections)
            n_residue = cid_meta[cid]['n_residue']
            res = minimize(_loss, r0, constraints=cons, options={'maxiter': maxiter, 'disp': True}, jac=True,
                           method='trust-constr')
            while (not res.success) and (count < max_count):
                maxiter *= 2
                logger.warning(f"Chunk {i + 1} Minimization failed, try increasing maxiter to {maxiter}")
                r0 = res.x
                # res = minimize(_loss, r0, constraints=cons, options={'maxiter': maxiter, 'disp': True}, jac=False, method='COBYLA')
                res = minimize(_loss, r0, constraints=cons, options={'maxiter': maxiter, 'disp': True}, jac=True,
                               method='trust-constr')
                count += 1
            if count >= max_count:
                for ii, r0_ in enumerate(res.x.reshape(-1, 3, 3)):
                    rot_[local_to_global[ii]] = r0_
                continue
            for ii, r0_ in enumerate(res.x.reshape(-1, 3, 3)):
                rot_[local_to_global[ii]] = r0_
        return rot_
