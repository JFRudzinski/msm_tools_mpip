import unittest
import warnings
import numpy as np
import pyemma
import mdtraj as md


def get_CN_rcut( r, gr, Nspac, Npts ):
    # This function is an easy way to find the minimum of a function (in particular the rdf)
    # We look for function values which are lower than their neighbors
    # N is the number of grids on each side to check, five seems to work well for 0.005 spacing
    # Always check the result! This is not gauranteed to work.
    return r[np.where( np.r_[True, gr[Nspac:] < gr[:-Nspac]] & np.r_[gr[:-Nspac] < gr[Nspac:], True] == True )[0][0:Npts]]

def get_gr_max( r, gr, Nspac, Npts ):
    # This function is an attempt to return the first solvation shell maximum of an rdf
    # We look for function values which are lower than their neighbors
    # N is the number of grids on each side to check, five seems to work well for 0.005 spacing
    # Always check the result! This is not gauranteed to work.
    return r[np.where( np.r_[True, gr[Nspac:] > gr[:-Nspac]] & np.r_[gr[:-Nspac] > gr[Nspac:], True] == True )[0][0:Npts]]

def get_loc_extr( r, gr, Nspac, Npts, flag_max ):
    # This function is an easy way to find the minimum of a function (in particular the rdf)
    # We look for function values which are lower than their neighbors
    # N is the number of grids on each side to check, five seems to work well for 0.005 spacing
    # Always check the result! This is not gauranteed to work.
    if ( flag_max ):
        return r[np.where( np.r_[True, gr[Nspac:] > gr[:-Nspac]] & np.r_[gr[:-Nspac] > gr[Nspac:], True] == True )[0][0:Npts]]
    else:
        return r[np.where( np.r_[True, gr[Nspac:] < gr[:-Nspac]] & np.r_[gr[:-Nspac] < gr[Nspac:], True] == True )[0][0:Npts]]


def get_excl_pair_list( top, sel_1, sel_2, n_excl, n_sites_p_mol ):
    import numpy as np
    # get the full pair list
    pairs = top.select_pairs( sel_1, sel_2 )
    # get the exclusions based only on index
    excl_list = np.where( np.abs( pairs[:,0] - pairs[:,1] ) < n_excl )[0]
    # remove excluded if they are on different molecules
    excl_excl = np.where( (pairs // n_sites_p_mol)[:,0] != (pairs // n_sites_p_mol)[:,1] )[0]
    #ind_list = []
    #for i in range(excl_excl.shape[0]): # slow but I can't figure out the python way
    #    if ( np.array(np.where( excl_list == excl_excl[i] )).shape[1] != 0 ):
    #        ind_list.append( np.where( excl_list == excl_excl[i])[0][0] )
    #excl_list = np.delete( excl_list, ind_list, axis=0)
    for i in range(excl_excl.shape[0]):
        if ( np.array(np.where( excl_list == excl_excl[i] )).shape[1] != 0 ):
            excl_list = [value for value in excl_list if value != excl_excl[i]]
    # now remove the exclusions
    pairs = np.delete(pairs, excl_list, axis=0)
    return pairs

# **Do the pair selection with get_excl_pair_list()**
# sel1 is the group of interest and should match that from the pair selection
def calc_coord_num_faster( mdtraj, (sel1, pairs, rcut) ): 
    # we expect mdtraj has dims (nmol*natoms/mol)
    n_mol = mdtraj.n_residues
    n_atms_p_mol = mdtraj.n_atoms / mdtraj.n_residues
    # get the indices of the particles
    part_ind = mdtraj.topology.select(sel1)
    part_ind = part_ind.tolist()
    # define the dimensions of the feature
    feature_vec = np.zeros(shape=(mdtraj.n_frames,len(part_ind)),dtype=np.float32)
    for fr in range(mdtraj.n_frames):
        # get the distances once for each frame
        dist = md.compute_distances(mdtraj[fr], pairs, periodic=True, opt=True)
        coord = np.where(dist < rcut)[1]
        neigh = reshape(pairs[coord],(pairs[coord].shape[0]*2))
        val = np.bincount(neigh)
        pind = np.nonzero(val)[0]
        ind = [part_ind.index(i) for i in pind]
        feature_vec[fr,ind] = val[pind]
        if ( fr % 200 == 0):
            print fr
    return feature_vec

# **Do the pair selection with get_excl_pair_list()**
# sel is the group of interest and should correspond to all sites in the pair selection, i.e., sel1&sel2
def calc_wghtd_CN( mdtraj, (sel, pairs, rcut, r, weights) ): 
    # we expect mdtraj has dims (nmol*natoms/mol)
    n_mol = mdtraj.n_residues
    n_atms_p_mol = mdtraj.n_atoms / mdtraj.n_residues
    # get the indices of the particles
    part_ind = mdtraj.topology.select(sel)
    part_ind = part_ind.tolist()
    # define the dimensions of the feature
    feature_vec = np.zeros(shape=(mdtraj.n_frames,len(part_ind)),dtype=np.float32)
    # get some grid info
    N_grids = r.shape[0]
    dr = r[1] - r[0]
    for fr in range(mdtraj.n_frames):
        # get the distances once for each frame
        dist = md.compute_distances(mdtraj[fr], pairs, periodic=True, opt=True)
        # select indices of pairs within the cutoff
        coord = np.where(dist < rcut)[1]
        # get the grids corresponding to these distances
        grids = ((dist[0,coord] - r[0])/dr).astype(int)
        # and the corresponding weights
        weight_grids = np.array(weights)[grids]
        # store the identifier of each pair separately (i.e., each part has it's own features)
        set1 = pairs[coord][:,0]
        set2 = pairs[coord][:,1]
        # now sum up the weights for each particle
        ugroup1 = np.unique(set1)
        ugroup2 = np.unique(set2)
        sums1 = []
        sums2 = []
        for group in ugroup1:
            sums1.append(weight_grids[set1 == group].sum())
        for group in ugroup2:
            sums2.append(weight_grids[set2 == group].sum())
        sums_comb = np.concatenate((np.array(sums1),np.array(sums2)))
        ugroup_comb = np.concatenate((ugroup1,ugroup2))
        ugroup = np.unique(ugroup_comb)
        sums = []
        for group in ugroup:
            sums.append(sums_comb[ugroup_comb == group].sum())
        # finally, store the feature
        ind = [part_ind.index(i) for i in ugroup]
        feature_vec[fr,ind] = sums
#        if ( fr % mdtraj.n_frames/5 == 0):
#            print 'On frame '+str(fr)+' of '+str(mdtraj.n_frames)+' for current traj...'
    return feature_vec

# **Do the pair selection with get_excl_pair_list()**
# sel is the group of interest and should correspond to all sites in the pair selection, i.e., sel1&sel2
def calc_wghtd_MT_cont( mdtraj, topfile, Xre_coor_all, (pairs_excl_sel_1, r_sel_1, rcut_sel_1, weight_sel_1), (pairs_excl_sel_2, r_sel_2, rcut_sel_2, weight_sel_2)):
    # we expect mdtraj has dims (nmol*natoms/mol)
    n_mol = mdtraj.n_residues
    n_atms_p_mol = mdtraj.n_atoms / mdtraj.n_residues
    # define the dimensions of the feature
    feat_vec = np.zeros(shape=(mdtraj.n_frames,mdtraj.n_atoms),dtype=np.float32)
    # get some grid info
    N_grids_sel_1 = r_sel_1.shape[0]
    dr_sel_1 = r_sel_1[1] - r_sel_1[0]
    N_grids_sel_2 = r_sel_2.shape[0]
    dr_sel_2 = r_sel_2[1] - r_sel_2[0]
    for fr in range(mdtraj.n_frames):
        # get the distances between each set of pairs
        dist_sel_1 = md.compute_distances(mdtraj[fr], pairs_excl_sel_1, periodic=True, opt=True)
        dist_sel_2 = md.compute_distances(mdtraj[fr], pairs_excl_sel_2, periodic=True, opt=True)
        # select indices of pairs within the cutoff
        dist_ind_cut_sel_1 = np.where(dist_sel_1 < rcut_sel_1)[1]
        dist_ind_cut_sel_2 = np.where(dist_sel_2 < rcut_sel_2)[1]
        # get the grids corresponding to these distances
        dr_sel_1 = r_sel_1[1] - r_sel_1[0]
        grids_sel_1 = ((dist_sel_1[0,dist_ind_cut_sel_1] - r_sel_1[0])/dr_sel_1).astype(int)
        dr_sel_2 = r_sel_2[1] - r_sel_2[0]
        grids_sel_2 = ((dist_sel_2[0,dist_ind_cut_sel_2] - r_sel_2[0])/dr_sel_2).astype(int)
        # and the corresponding weights
        weight_grids_sel_1 = np.array(weight_sel_1)[grids_sel_1]
        weight_grids_sel_2 = np.array(weight_sel_2)[grids_sel_2]      
        # for each pair within the cutoff, calculate the force vector
        # sel 1
        set1_sel_1 = pairs_excl_sel_1[dist_ind_cut_sel_1][:,0]
        set2_sel_1 = pairs_excl_sel_1[dist_ind_cut_sel_1][:,1]
        coor_p1_sel_1 = Xre_coor_all[0][set1_sel_1]
        coor_p2_sel_1 = Xre_coor_all[0][set2_sel_1]
        f_unit_sel_1 = coor_p1_sel_1 - coor_p2_sel_1
        f_unit_sel_1 /= np.linalg.norm(f_unit_sel_1,keepdims=True,axis=1)
        # sel 2
        set1_sel_2 = pairs_excl_sel_2[dist_ind_cut_sel_2][:,0]
        set2_sel_2 = pairs_excl_sel_2[dist_ind_cut_sel_2][:,1]
        coor_p1_sel_2 = Xre_coor_all[0][set1_sel_2]
        coor_p2_sel_2 = Xre_coor_all[0][set2_sel_2]
        f_unit_sel_2 = coor_p1_sel_2 - coor_p2_sel_2
        f_unit_sel_2 /= np.linalg.norm(f_unit_sel_2,keepdims=True,axis=1)
        # get the unique set of particle numbers from each search
        ugroup1_sel_1 = np.unique(set1_sel_1)
        ugroup2_sel_1 = np.unique(set2_sel_1)
        ugroup1_sel_2 = np.unique(set1_sel_2)
        ugroup2_sel_2 = np.unique(set2_sel_2)
        ugroup_comb = np.concatenate((ugroup1_sel_1,ugroup2_sel_1,ugroup1_sel_2,ugroup2_sel_2))
        ugroup = np.unique(ugroup_comb) # unique group of all particle indices participating in a pair 
        # Now, loop through pairs of pairs and calculate the contribution to MT
        for group in ugroup: # loop over the relevant particle indices
            # 1 - set1_sel_1
            for i in (np.where(set1_sel_1==group)[0]): 
                w1 = np.sqrt(weight_grids_sel_1[i])
                # set1_sel_2
                for j in (np.where(set1_sel_2==group)[0]):
                    if (set2_sel_1[i] == set2_sel_2[j]): continue # only consider distinct pairs
                    w2 = np.sqrt(weight_grids_sel_2[j])
                    costhet = np.dot(f_unit_sel_1[i],f_unit_sel_2[j]) # don't forget to change the sign of the force vector
                    feat_vec[fr,group] += w1*w2*costhet
            # set2_sel_2
            for j in (np.where(set2_sel_2==group)[0]):
                if (set2_sel_1[i] == set1_sel_2[j]): continue # only consider distinct pairs
                w2 = np.sqrt(weight_grids_sel_2[j])
                costhet = np.dot(f_unit_sel_1[i],-1.0*f_unit_sel_2[j]) # don't forget to change the sign of the force vector
                feat_vec[fr,group] += w1*w2*costhet
            # 1 - set2_sel_1
            for i in (np.where(set2_sel_1==group)[0]):
                w1 = np.sqrt(weight_grids_sel_1[i])
                # set1_sel_2
                for j in (np.where(set1_sel_2==group)[0]):
                    if (set1_sel_1[i] == set2_sel_2[j]): continue # only consider distinct pairs
                    w2 = np.sqrt(weight_grids_sel_2[j])
                    costhet = np.dot(-1.0*f_unit_sel_1[i],f_unit_sel_2[j]) # don't forget to change the sign of the force vector
                    feat_vec[fr,group] += w1*w2*costhet
                # set2_sel_2
                for j in (np.where(set2_sel_2==group)[0]):
                    if (set1_sel_1[i] == set1_sel_2[j]): continue # only consider distinct pairs
                    w2 = np.sqrt(weight_grids_sel_2[j])
                    costhet = np.dot(-1.0*f_unit_sel_1[i],-1.0*f_unit_sel_2[j]) # don't forget to change the sign of the force vector
                    feat_vec[fr,group] += w1*w2*costhet
    
    return feat_vec
