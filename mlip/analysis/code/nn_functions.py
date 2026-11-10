import numpy as np

def rotate(coordinates, degrees):
    theta = np.deg2rad(degrees)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
    
    return (rot_matrix @ coordinates.T).T

def fix_pbc(p0, p1, a_moire):
    p0 = p0.copy()
    p1 = p1.copy()
    u = p1 - p0

    p1[u[:,0] >  a_moire/2,0] -= a_moire
    p1[u[:,0] < -a_moire/2,0] += a_moire
    p1[u[:,1] >  a_moire/2,1] -= a_moire
    p1[u[:,1] < -a_moire/2,1] += a_moire
    return p1

def get_layer_atom_type(atoms, atom_idx):
    ''' Some logic depends on the atom type and which layer it is in
        The 4 options and corresponding output are:
            Ti : ti
            Sr : sr
            O in Ti layer : oti
            O in Sr layer : osr
    '''
    layer_index = [i for i, layer_idxs in enumerate(atoms.info['all_layers_idx']) if atom_idx in layer_idxs][0]

    if atoms[atom_idx].symbol == 'Ti':
        return 'ti'
    elif atoms[atom_idx].symbol == 'Sr':
        return 'sr'
    elif atoms[atoms.info['all_layers_idx'][layer_index]].symbols.count('Ti') != 0:
        return 'oti'
    else: 
        return 'osr'

def get_layer_type(atoms):
    if atoms.symbols.count('Ti') != 0:
        return 'ti'
    else:
        return 'sr'

def repeat_data(atoms, pos, data, rep):
    if (abs(atoms.cell[0][1]) > 1e-3) or (abs(atoms.cell[1][0]) > 1e-3):
        raise Exception('Repeat function assumes the twisted unit cell is')
    a_moire_x = atoms.cell[0][0]
    a_moire_y = atoms.cell[1][1]

    new_shape = [1]*len(data.shape)
    new_shape[0] = rep[0]*rep[1]

    data = np.tile(data, new_shape)
    i_rep = rep[0]
    j_rep = rep[1]
    new_pos = []
    for i in range(i_rep):
        for j in range(j_rep):
            pos_temp = pos.copy()
            pos_temp[:,0] += a_moire_x*i
            pos_temp[:,1] += a_moire_y*j
            new_pos.append(pos_temp)
    new_pos = np.vstack(new_pos)
    return new_pos, data
