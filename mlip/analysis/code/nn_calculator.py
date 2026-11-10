import numpy as np

from ase.neighborlist import NewPrimitiveNeighborList
from ase.geometry import get_distances

from .nn_functions import fix_pbc, get_layer_atom_type

class NNCalculator:
    def __init__(self, final, initial, a, atom_types=['Ti','Sr','O'], n_nearest_neighbors='four'):
        if isinstance(n_nearest_neighbors,str):
            n_nearest_neighbors = n_nearest_neighbors.lower()
        if n_nearest_neighbors not in ['four','eight']:
            raise Exception("n_nearest_neighbors needs to be either 'four' or 'eight'")
        
        self.orig_initial = initial.copy()
        self.orig_final = final.copy()
        self.a = a
        self.a_moire = self.orig_initial.cell[0][0]
        self.atom_types = atom_types
        self.initial = initial.copy()
        self.final = final.copy()
        self.n_nearest_neighbors = n_nearest_neighbors

    def get_nn_idx_from_atom_idx(self, atom_idxs):
        matches = []
        for idx in atom_idxs:
            matches.append(np.argwhere(idx == self.nn_data[:,0])[0][0])
        matches = np.array(matches)
        return matches

    def get_atom_idx_from_nn_idx(self, atom_idxs):
        return self.nn_data[atom_idxs,0]

    def get_layer_idxs_from_atom_idx(self, idxs):
        all_layer_idxs = []
        for idx in idxs:
            found = False
            for i, layer_idxs in enumerate(self.final.info['all_layers_idx']):
                if idx in layer_idxs:
                    all_layer_idxs.append(i)
                    found = True
                    break
            if found is False:
                raise Exception('Index was not found')
        all_layer_idxs = np.array(all_layer_idxs)
        return all_layer_idxs

    def _get_nn_data(self, atoms, r_nl=3.0):
        nl = NewPrimitiveNeighborList([r_nl]*len(atoms), bothways=True, self_interaction=False)
        nl.update(atoms.pbc, atoms.get_cell(), atoms.positions)

        atom_idxs = []
        all_nn_indices = []
        for atom_idx in range(len(atoms)):
            if not atoms[atom_idx].symbol in self.atom_types:
                continue
            indices, offsets = nl.get_neighbors(atom_idx)
            atom_type = atoms[atom_idx].symbol
            
            mask = [True if a.symbol == atom_type else False for a in atoms[indices]] # keep only same element
            indices, offsets = indices[mask], offsets[mask]
            
            z_mask = abs(atoms.positions[indices,2] - atoms.positions[atom_idx,2]) < 1.5
            indices, offsets = indices[z_mask], offsets[z_mask]

            ds = []
            for i, offset in zip(indices, offsets):
                d, d_len = get_distances(atoms[atom_idx].position, atoms[i].position, cell=atoms.cell, pbc=atoms.pbc)
                ds.append(d_len[0][0])
            ds = np.array(ds)

            n_lowest_values = 4
            nn_indices = np.argsort(ds)
            second_nn_indices = nn_indices.copy()
            nn_indices = nn_indices[:n_lowest_values]
            ds_nn = ds[nn_indices]
            nn_atom_indices = indices[nn_indices]
            if abs(ds_nn.max() - ds_nn.min()) > 1.8:
                raise Exception('Distance seems too large. Need to double check')

            second_nn_indices = second_nn_indices[~np.isin(second_nn_indices,nn_indices)]
            if len(second_nn_indices) < 4:
                raise Exception('Too few second nearest neighbors found. Need to up r_nl')
            second_nn_indices = second_nn_indices[:n_lowest_values]
            ds_second_nn = ds[second_nn_indices]
            second_nn_atom_indices = indices[second_nn_indices]
            if abs(ds_second_nn.max() - ds_second_nn.min()) > 2.0:
                raise Exception('Distance seems too large. Need to double check')

            atom_idxs.append(atom_idx)
            all_nn_indices.append(np.hstack([nn_atom_indices,second_nn_atom_indices]))
            
        nn_data = np.vstack([atom_idxs,np.array(all_nn_indices).T]).T
        return nn_data

    def _get_position_data(self, atoms):
        position_data = np.full((self.nn_data.shape[0],3),np.nan)
        for i, row in enumerate(self.nn_data):
            atom_idx = row[0]
            nn_list = row[1:]
            atom_position = atoms[atom_idx].position
            position_data[i,:] = atom_position
        return position_data

    def _get_distance_data(self, atoms, nn_data):
        distance_data = np.full((nn_data.shape[0],8,3),np.nan)
        distance_len_data = np.full((nn_data.shape[0],8),np.nan)
        for i, row in enumerate(nn_data):
            atom_idx = row[0]
            nn_list = row[1:]
            atom_position = atoms[atom_idx].position
            d, d_len = get_distances(atom_position, atoms[nn_list].positions, cell=atoms.cell, pbc=atoms.pbc)
            d, d_len = d[0], d_len[0]
            distance_data[i, :, :] = d
            distance_len_data[i,:] = d_len
        return distance_data, distance_len_data

    def _get_rot_distance_data(self, atoms, nn_data):
        atoms_rotated = atoms.copy()
        atoms_rotated.rotate(45,'z',rotate_cell=True)

        rot_distance_data = np.full((nn_data.shape[0],8,3),np.nan)
        rot_distance_len_data = np.full((nn_data.shape[0],8),np.nan)
        for i, row in enumerate(nn_data):
            atom_idx = row[0]
            nn_list = row[1:]
            atom_position = atoms_rotated[atom_idx].position
            d, d_len = get_distances(atom_position, atoms_rotated[nn_list].positions, cell=atoms_rotated.cell, pbc=atoms_rotated.pbc)
            d, d_len = d[0], d_len[0]
            rot_distance_data[i, :, :] = d
            rot_distance_len_data[i,:] = d_len
        return rot_distance_data, rot_distance_len_data

    def _get_nn_direction_data(self, nn_data):
        nn_direction = np.full((nn_data.shape[0],16),-99999)
        for i, row in enumerate(nn_data):
            ### Nearest neighbor block ###
            # assuming it is aligned to xy, one 
            # column gives xy direction and second
            # column gives positive / negative
            atom_idx = row[0]

            layer_atom_type = get_layer_atom_type(self.final, atom_idx)
            # Ti and Sr needs to use rotated distance data to align with xy
            if layer_atom_type in ['ti','sr']:
                d = self.rot_distance_data_init[i,:4,:]
            # Oxygen in Sr layer is aligned to xy with rotation
            elif layer_atom_type == 'osr':
                d = self.rot_distance_data_init[i,:4,:]
            elif layer_atom_type == 'oti': # Oxygen in Ti layer is aligned to xy without rotation
                d = self.distance_data_init[i,:4,:]
            xy_dir = np.argmax(abs(d),axis=1)
            pos_dir = []

            for j in range(len(xy_dir)):
                if d[j, xy_dir[j]] < 0:
                    pos_dir.append(-1)
                else:
                    pos_dir.append(1)
            nn_direction[i,:4] = xy_dir
            nn_direction[i,4:8] = pos_dir
            ### Second nearest neighbor block ###
            # assuming NOT aligned to xy (since nn are)
            # so first column gives x-direction
            #   second column gives y-direction
            if layer_atom_type in ['ti','sr']:
                d = self.rot_distance_data_init[i,4:,:]
            # Oxygen in Sr layer is aligned to xy with rotation
            elif layer_atom_type == 'osr':
                d = self.rot_distance_data_init[i,4:,:]
            elif layer_atom_type == 'oti': # Oxygen in Ti layer is aligned to xy without rotation
                d = self.distance_data_init[i,4:,:]
            
            nn_direction[i,8:12][d[:,0] < 0] = -1
            nn_direction[i,8:12][d[:,0] > 0] = 1
            nn_direction[i,12:][d[:,1] < 0] = -1
            nn_direction[i,12:][d[:,1] > 0] = 1
        return nn_direction

    def _get_n_nearest_neighbors(self, nn_list):
        if self.n_nearest_neighbors == 'four':
            idx_nn = nn_list[:4]
        elif self.n_nearest_neighbors == 'eight':
            idx_nn = nn_list
        return idx_nn
    
    def _get_ref_pos_data(self, atoms, nn_data):
        a_moire = atoms.cell[0][0]
        ref_pos = np.full(shape=(nn_data.shape[0],3),fill_value=np.nan)

        for i, row in enumerate(nn_data):
            atom_idx = row[0]
            nn_list = row[1:]
            idx_nn = self._get_n_nearest_neighbors(nn_list)

            p0 = atoms.positions[atom_idx, :]
            p1 = atoms.positions[idx_nn, :]
            p1 = fix_pbc(p0,p1,a_moire)

            ref_pos[i, 0] = p1[:,0].mean()
            ref_pos[i, 1] = p1[:,1].mean()
            ref_pos[i, 2] = p1[:,2].mean()
        return ref_pos
    
    def get_all_data(self):
        if hasattr(self, 'nn_data'):
            return

        self.nn_data = self._get_nn_data(self.initial)
        self.position_data = self._get_position_data(self.final)
        self.distance_data, self.distance_len_data = self._get_distance_data(self.final, self.nn_data)
        self.rot_distance_data, self.rot_distance_len_data = self._get_rot_distance_data(self.final, self.nn_data)
        self.distance_data_init, self.distance_len_data_init = self._get_distance_data(self.initial, self.nn_data)
        self.rot_distance_data_init, self.rot_distance_len_data_init = self._get_rot_distance_data(self.initial, self.nn_data)
        self.nn_direction = self._get_nn_direction_data(self.nn_data)
        self.ref_pos = self._get_ref_pos_data(self.final, self.nn_data)

    def set_nn_data(self, other):
        self.nn_data                    = other.nn_data
        self.position_data              = other.position_data
        self.distance_data              = other.distance_data
        self.distance_len_data          = other.distance_len_data
        self.rot_distance_data          = other.rot_distance_data
        self.rot_distance_len_data      = other.rot_distance_len_data
        self.distance_data_init         = other.distance_data_init
        self.distance_len_data_init     = other.distance_len_data_init
        self.rot_distance_data_init     = other.rot_distance_data_init
        self.rot_distance_len_data_init = other.rot_distance_len_data_init
        self.nn_direction               = other.nn_direction
        self.ref_pos                    = other.ref_pos