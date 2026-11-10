import numpy as np

### Help functions to set data lost when saving as .cif files ### 

def _get_levels(atoms):
    pos = atoms.get_positions()
    levels = np.unique(pos[:,2])
    return levels

def set_layer_idxs(final, initial):
    ''' Find layer that each atom belongs to and save in atoms.info['all_layers_idx'] '''
    pos = initial.get_positions()
    levels = _get_levels(initial)

    idxs = []
    for level in levels:
        mask = np.argwhere(np.isclose(pos[:,2], level)).flatten()
        idxs.append(list(mask))
    
    final.info['all_layers_idx'] = idxs
    initial.info['all_layers_idx'] = idxs

def set_number_lower_layers(final, initial):
    ''' Find number of layers below interface 
        and save in atoms.info['n_lower'] '''
    levels = _get_levels(initial)

    for i in range(len(levels)-1):
        if levels[i+1] - levels[i] > 2:
            final.info['n_lower'] = i+1
            initial.info['n_lower'] = i+1
            break