import numpy as np

from .nn_calculator import NNCalculator
from .nn_functions import rotate, get_layer_atom_type
from .all_unit_cells import AllUnitCells

class StrainCalculator(NNCalculator):
    def __init__(self, final, initial, a, atom_types=['Sr','Ti'], n_nearest_neighbors='four'):
        super().__init__(final, initial, a, atom_types, n_nearest_neighbors=n_nearest_neighbors)

    def calculate_strains(self):
        self.get_all_data()
        self.strain = self._get_strains_data(self.final, self.nn_data, self.nn_direction, self.ref_pos, self.a)
        return self.strain
    
    def get_strains(self):
        if not hasattr(self, 'strain'):
            self.calculate_strains()
        return self.strain

    def _get_strains_data(self, atoms, nn_data, nn_direction, ref_pos, a):
        strain = np.full((nn_data.shape[0],2,2),np.nan)
        self.u = np.full((nn_data.shape[0],2),np.nan)
        for i, row in enumerate(nn_data):
            atom_idx = int(row[0])

            nn_list = row[1:5]
            second_nn_list = row[5:]
            mask_10 = (nn_direction[i,:4] == 0) & (nn_direction[i,4:8] == -1)
            atom_nn_idx_10 = nn_list[mask_10][0]
            ref_mask_10 = atom_nn_idx_10 == nn_data[:,0]

            mask_01 = (nn_direction[i,:4] == 1) & (nn_direction[i,4:8] == 1)
            atom_nn_idx_01 = nn_list[mask_01][0]
            ref_mask_01 = atom_nn_idx_01 == nn_data[:,0]

            mask11 = (nn_direction[i,8:12] == -1) & (nn_direction[i,12:] == 1)
            atom_nn_idx_11 = second_nn_list[mask11][0]
            ref_mask_11 = atom_nn_idx_11 == nn_data[:,0]

            u00 = atoms.positions[atom_idx,:2] - ref_pos[i,:2]
            u10 = (atoms.positions[atom_nn_idx_10,:2] - ref_pos[ref_mask_10,:2])[0]
            u01 = (atoms.positions[atom_nn_idx_01,:2] - ref_pos[ref_mask_01,:2])[0]
            u11 = (atoms.positions[atom_nn_idx_11,:2] - ref_pos[ref_mask_11,:2])[0]
            self.u[i][0] = u00[0]
            self.u[i][1] = u00[1]

            u00 = rotate(u00,45) 
            u10 = rotate(u10,45) 
            u01 = rotate(u01,45) 
            u11 = rotate(u11,45) 

            layer_atom_type = get_layer_atom_type(self.final, atom_idx)
            if layer_atom_type == 'oti':
                denom = a/np.sqrt(2)
            elif layer_atom_type in ['sr','ti','osr']:
                denom = a

            exx = 1/(2*denom)*((u11[0]-u01[0])+(u10[0]-u00[0]))*100
            exy = 1/(2*denom)*((u11[0]-u10[0])+(u01[0]-u00[0]))*100

            eyy = 1/(2*denom)*((u11[1]-u10[1])+(u01[1]-u00[1]))*100
            eyx = 1/(2*denom)*((u11[1]-u01[1])+(u10[1]-u00[1]))*100

            strain[i,0,0] = exx
            strain[i,0,1] = exy
            strain[i,1,0] = eyx
            strain[i,1,1] = eyy
        return strain
    
    def calculate_uc_average_strain(self, layer_indices, include_os=True):
        self.layer_indices = layer_indices
        self.get_strains()

        all_strain_data = []
        all_ti_idxs = []
        for layer_index in layer_indices:
            layer_mask = np.array([True if i in self.final.info['all_layers_idx'][layer_index] else False for i in range(len(self.final))])
            if self.final[layer_mask].symbols.count('Ti') != 0:
                ti_mask = np.array([True if a.symbol == 'Ti' else False for a in self.final])
                layer_type = 'Ti'
            else:
                ti_mask = np.array([True if a.symbol == 'Sr' else False for a in self.final])
                layer_type = 'Sr'
            mask  = layer_mask & ti_mask
            ti_idxs = np.arange(len(self.final))[mask]
            strain_data = np.full((len(ti_idxs),2,2),np.nan)

            auc = AllUnitCells(self.initial, uc_center_type=layer_type)
            for i_data, ti_idx in enumerate(ti_idxs):
                uc_idxs = auc.get_uc_for_atom_idx(ti_idx)
            
                idxs = []
                for idx in uc_idxs:
                    atom_data_idx = np.argwhere(self.nn_data[:,0] == idx)[0][0]
                    if include_os:
                        idxs.append(atom_data_idx)
                    elif self.final[atom_data_idx].symbol != 'O':
                        idxs.append(atom_data_idx)
                strain_data[i_data] = self.strain[idxs].mean(axis=0)

            all_strain_data.append(strain_data)
            all_ti_idxs.append(ti_idxs)
        strain = np.array(all_strain_data)
        ti_idxs = np.array(all_ti_idxs)
        return ti_idxs, strain
    
    def get_ti_strain_data(self, ti_idxs):
        self.get_strains()
        strain_data_ti = []
        for ti_idx in ti_idxs:
            nn_ti_idx = self.get_nn_idx_from_atom_idx(ti_idx)
            strain_data_ti.append(self.strain[nn_ti_idx])
        return np.array(strain_data_ti)