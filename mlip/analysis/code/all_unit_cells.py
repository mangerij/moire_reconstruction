import numpy as np

from ase.neighborlist import NewPrimitiveNeighborList
from ase.geometry import get_distances

class AllUnitCells:
    def __init__(self, atoms, r_nl=1.6, uc_center_type='Ti'):
        self.r_nl = r_nl
        self._set_uc_center_type(uc_center_type, atoms)
        self.nn = self._find_uc(atoms)

    def _set_uc_center_type(self, uc_center_type, atoms):
        if len(atoms.symbols.species()) != 3:
            raise Exception('This function only works for perovskites with 3 elements')
        for element in atoms.symbols.species():
            if element != 'O' and element != uc_center_type:
                self.uc_other_type = element
        self.uc_center_type = uc_center_type
        if 'Ti' not in atoms.symbols.species():
            raise Exception('Function only works for Ti B cation')
        if self.uc_center_type == 'Ti':
            self.number_nn_oxygens = 6
            self.number_nn = 15
        else:
            self.number_nn_oxygens = 12
            self.number_nn = 21

    def find_uc_indices_for_atom(self, atoms, center_idxs, r_nl=1.8):
        nl = NewPrimitiveNeighborList([r_nl]*len(atoms), bothways=True, self_interaction=False)

        nl.update(atoms.pbc, atoms.get_cell(), atoms.positions)
        all_found_idxs_other = []
        all_found_idxs_o = []
        all_found_idxs_center = []
        for atom_idx in center_idxs:
            indices, offsets = nl.get_neighbors(atom_idx)
            d_lens_temp = []
            
            for i, offset in zip(indices, offsets):
                d, d_len = get_distances(atoms[atom_idx].position, atoms[i].position, cell=atoms.cell, pbc=atoms.pbc)
                d_lens_temp.append(d_len[0][0])
            found_idxs = []
            for i in np.argsort(d_lens_temp):
                idx = indices[i]
                if atoms[idx].symbol == self.uc_other_type:
                    found_idxs.append(idx)
            all_found_idxs_other.append(found_idxs)
            if len(found_idxs) < 8:
                raise Exception('Too few other')
            elif len(found_idxs) > 8:
                raise Exception('Too many other')
            found_idxs = []
            for i in np.argsort(d_lens_temp):
                idx = indices[i]
                if atoms[idx].symbol == 'O':
                    found_idxs.append(idx)
            all_found_idxs_o.append(found_idxs)
            if len(found_idxs) < self.number_nn_oxygens:
                raise Exception('Too few oxygen')
            elif len(found_idxs) > self.number_nn_oxygens:
                raise Exception('Too many oxygen')
            all_found_idxs_center.append([atom_idx])
        return all_found_idxs_center, all_found_idxs_o, all_found_idxs_other

    def _find_uc(self, atoms):
        # Mask out interface
        mask_interface = [] 
        mask_interface += atoms.info['all_layers_idx'][atoms.info['n_lower']-1]
        mask_interface += atoms.info['all_layers_idx'][atoms.info['n_lower']]
        # Mask out surfance
        mask_interface += atoms.info['all_layers_idx'][0]
        mask_interface += atoms.info['all_layers_idx'][-1]

        num_center = len(atoms[[True if ((a.symbol == self.uc_center_type) and (i not in mask_interface)) else False for i, a in enumerate(atoms)]])
        center_atom_idxs = [a.index for i, a in enumerate(atoms) if ((a.symbol == self.uc_center_type) and (i not in mask_interface))]
        center_idxs, o_idxs, other_idxs = self.find_uc_indices_for_atom(atoms, center_atom_idxs, r_nl=self.r_nl)

        uc_nn = np.full((num_center,self.number_nn),np.nan)

        for i, (ti_idx, o_idx, sr_idx) in enumerate(zip(center_idxs,o_idxs,other_idxs)):
            uc_nn[i,0] = ti_idx[0]
            for j in range(len(o_idx)):
                uc_nn[i,1+j] = o_idx[j]
            for k in range(len(sr_idx)):
                uc_nn[i,2+j+k] = sr_idx[k]

        uc_nn = uc_nn.astype(int)
        return uc_nn
    
    def get_uc_for_atom_idx(self, idx):
        uc_list = self.nn[self.nn[:,0] == idx]
        return uc_list[0]
    
    def split_list_in_tisro(self, uc_list):
        uc_center = uc_list[[0]]
        uc_o = uc_list[1:self.number_nn_oxygens+1]
        uc_other  = uc_list[self.number_nn_oxygens+1:]
        return uc_center, uc_other, uc_o 