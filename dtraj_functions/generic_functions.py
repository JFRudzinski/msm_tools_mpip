import unittest
import warnings
import numpy as np
import pyemma
import mdtraj as md

def feat_CN_per_mol( X, n_frames_tot, n_frames_p_traj, n_traj, n_feat_p_mol, n_mol ):
    X = np.reshape(X,(n_frames_tot,n_feat_p_mol,n_mol),order='F')
    Xre = []
    for i in range(n_traj):
        for j in range(n_mol):
            Xre.append(X[i*n_frames_p_traj:(i+1)*n_frames_p_traj,:,j])
    return Xre

def calc_chunkwise( func, traj_list, top_file, chunk_size=1, dim=1, stride=1, skip=0 ):
# This function computes some observable from an md traj in trunks, as to not use too much memory
# It assumes that the passed in function has no additional input (e.g., use a lambda function)
# and that the output is to be (non-weighted) averaged over chuncks
    count = 0
    for i in range(len(traj_list)):
        for chunk in md.iterload(traj_list[i], chunk=chunk_size, top=top_file, stride=stride, skip=skip):
            func_ret_tmp = func(chunk)
            if (count==0):
                func_ret = np.array(func_ret_tmp)
            else:
                for j in range(dim):
                    func_ret[j] += np.array(func_ret_tmp)[j]
            count += 1
    for i in range(dim):
        func_ret[i] /= (1.0*count)
    return func_ret, count

def calc_chunkwise_noavg( func, traj_list, top_file, chunk_size=1, dim=1, stride=1, skip=0 ):
# This function computes some observable from an md traj in trunks, as to not use too much memory
# It assumes that the passed in function has no additional input (e.g., use a lambda function)
# and that the output is to be (non-weighted) averaged over chuncks
    count = 0
    for i in range(len(traj_list)):
        for chunk in md.iterload(traj_list[i], chunk=chunk_size, top=top_file, stride=stride, skip=skip):
            func_ret_tmp = func(chunk)
            if (count==0):
                func_ret = np.array(func_ret_tmp)
            else:
                if (dim==1):
                    func_ret = np.concatenate((func_ret,np.array(func_ret_tmp)),axis=0)
                else: # this is not yet tested!!
                    for j in range(dim):
                        func_ret[j] = np.concatenate((func_ret[j],np.array(func_ret_tmp)[j]),axis=0)
            count += 1
    return func_ret, count
