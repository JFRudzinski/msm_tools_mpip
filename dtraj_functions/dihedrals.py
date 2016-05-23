import unittest
import warnings
import numpy as np
import pyemma
import mdtraj as md


def calc_inst_ABAB_dih_PS( mdtraj, (N_dih) ): 
    # we expect mdtraj has dims (nmol*natoms/mol)
    n_mol = mdtraj.n_residues
    n_atms_p_mol = mdtraj.n_atoms / mdtraj.n_residues
    # define the dimensions of the feature
    feature_vec = np.zeros(shape=(mdtraj.n_frames,N_dih,n_mol),dtype=np.float32)
    for fr in range(mdtraj.n_frames):
        sel_pre = 'name '
        sel_type = [ 'A', 'B' ]
        sel1_nr = 1
        sel2_nr = 1
        sel3_nr = 2
        sel4_nr = 2
        for i in range(N_dih):
            #print i
            # define the selections
            sel1 = sel_pre+sel_type[i%2]+str(sel1_nr)
            sel2 = sel_pre+sel_type[(i+1)%2]+str(sel2_nr)
            sel3 = sel_pre+sel_type[i%2]+str(sel3_nr)
            sel4 = sel_pre+sel_type[(i+1)%2]+str(sel4_nr)
            dih_ind = mdtraj.topology.select(sel1+' or '+sel2+' or '+sel3+' or '+sel4)
            dih_ind = np.reshape(dih_ind,(dih_ind.shape[0]/4,4))
            feature_vec[fr,i] = md.compute_dihedrals(mdtraj[fr], dih_ind, periodic=True, opt=True)[0][:]
            # get the new selections
            sel1_nr = sel2_nr
            sel2_nr = sel3_nr
            sel3_nr = sel4_nr
            if (i%2==0):
                sel4_nr += 1
#        if ( fr % 200 == 0):
#            print fr
    return feature_vec

def calc_inst_XXXX_dih_PS( mdtraj, (N_dih, part_type) ):
    # we expect mdtraj has dims (nmol*natoms/mol)
    n_mol = mdtraj.n_residues
    n_atms_p_mol = mdtraj.n_atoms / mdtraj.n_residues
    # define the dimensions of the feature
    feature_vec = np.zeros(shape=(mdtraj.n_frames,N_dih,n_mol),dtype=np.float32)
    for fr in range(mdtraj.n_frames):
        sel_pre = 'name '
        for i in range(N_dih):
            # define the selections
            sel1 = sel_pre+part_type+str(i+1)
            sel2 = sel_pre+part_type+str(i+2)
            sel3 = sel_pre+part_type+str(i+3)
            sel4 = sel_pre+part_type+str(i+4)
            dih_ind = mdtraj.topology.select(sel1+' or '+sel2+' or '+sel3+' or '+sel4)
            dih_ind = np.reshape(dih_ind,(dih_ind.shape[0]/4,4))
            feature_vec[fr,i] = md.compute_dihedrals(mdtraj[fr], dih_ind, periodic=True, opt=True)[0][:]
#        if ( fr % 200 == 0):
#            print fr
    return feature_vec


def calc_wghtd_dih_pop( Xre_dih, bins, weights ): 
# Note: this function treats the dihedral as symmetric about zero and only considers abs(dih_val)
    # define the dimensions of the feature
    feature_vec = np.zeros(shape=(len(Xre_dih),Xre_dih[0].shape[0],len(weights)),dtype=np.float32)
    weight_grids = np.zeros(len(weights))
    # get some grid info
    N_grids = bins.shape[0]
    dbin = bins[1] - bins[0]
    for traj in range(len(Xre_dih)):
        for fr in range(Xre_dih[traj].shape[0]):
            # get the grids corresponding to abs(dih)
            grids = ((np.abs(Xre_dih[traj][fr,:]) - bins[0])/dbin).astype(int)
            # and the corresponding weights
            for i in range(len(weights)):
                weight_grids[i] = np.sum( np.array(weights)[i][grids] )
            feature_vec[traj,fr] = weight_grids
#        if ( traj % 200 == 0):
#            print traj
    return feature_vec

def calc_wghtd_dih_pop_full( Xre_dih, bins, weights ):
    # define the dimensions of the feature
    feature_vec = np.zeros(shape=(len(Xre_dih),Xre_dih[0].shape[0],len(weights)),dtype=np.float32)
    weight_grids = np.zeros(len(weights))
    # get some grid info
    N_grids = bins.shape[0]
    dbin = bins[1] - bins[0]
    for traj in range(len(Xre_dih)):
        for fr in range(Xre_dih[traj].shape[0]):
            # get the grids corresponding to abs(dih)
            grids = ((Xre_dih[traj][fr,:] - bins[0])/dbin).astype(int)
            # and the corresponding weights
            for i in range(len(weights)):
                weight_grids[i] = np.sum( np.array(weights)[i][grids] )
            feature_vec[traj,fr] = weight_grids
#        if ( traj % 200 == 0):
#            print traj
    return feature_vec

def calc_wghtd_dih_pop_full_noavg( Xre_dih, bins, weights ):
    # define the dimensions of the feature
    feature_vec = np.zeros(shape=(len(Xre_dih),Xre_dih[0].shape[0],len(weights),Xre_dih[0].shape[1]),dtype=np.float32)
    weight_grids = np.zeros(shape=(len(weights),Xre_dih[0].shape[1]))
    # get some grid info
    N_grids = bins.shape[0]
    dbin = bins[1] - bins[0]
    for traj in range(len(Xre_dih)):
        for fr in range(Xre_dih[traj].shape[0]):
            # get the grids corresponding to abs(dih)
            grids = ((Xre_dih[traj][fr,:] - bins[0])/dbin).astype(int)
            # and the corresponding weights
            for i in range(len(weights)):
                weight_grids[i] = np.sum( np.array(weights)[i][grids] )
                feature_vec[traj,fr] = weight_grids[i]
#        if ( traj % 200 == 0):
#            print traj
    return feature_vec
