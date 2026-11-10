from code.strain_calculator import StrainCalculator
from code.interlayer_distance_calculator import InterlayerDistanceCalculator

from code.atom_utils import set_layer_idxs, set_number_lower_layers

from ase.io import read
import numpy as np
import sys

angle = sys.argv[1]

final = read(f'{angle}deg_sim.cif')
initial = read(f'{angle}deg_rigid.cif')
bulk = read('bulk.cif')
a = bulk.cell[0][0]

# Set atom info used for later functions
set_layer_idxs(final, initial)
set_number_lower_layers(final, initial)

# Calculate strain
sc = StrainCalculator(final, initial, a, ['Sr','Ti','O'])
sc.get_strains()
ti_idxs, uc_strain = sc.calculate_uc_average_strain([4,7])

# Calculate interlayer distance
idc = InterlayerDistanceCalculator(final, 3)
x_il,y_il,z_il = idc.get_interlayer_distance([4,7])
data_interlayer = np.array([x_il,y_il,z_il]).T
np.savetxt(f'{angle}deg_interlayer.csv',data_interlayer)

# Format data in array
all_data = []
for layer_idx, ti_idx, strain in zip(sc.layer_indices, ti_idxs, uc_strain):
    nn_idxs = sc.get_nn_idx_from_atom_idx(ti_idx)
    ti_pos = sc.position_data[nn_idxs,:]
    u = sc.u[nn_idxs,:]
    
    df_layer_idx    = np.ones((ti_pos.shape[0],1))*layer_idx
    df_ti_pos       = ti_pos
    df_strain       = strain.reshape(ti_pos.shape[0],4)
    df_u            = u
    all_data.append(np.hstack([df_layer_idx,
                                df_ti_pos, 
                                df_strain, 
                                df_u]))
for layer_num in [5,6]:
    pos_u = sc.position_data[[True if a.symbol!= 'O' and i in final.info['all_layers_idx'][layer_num] else False for i,a in enumerate(final)]]
    u = sc.u[[True if a.symbol!= 'O' and i in final.info['all_layers_idx'][layer_num] else False for i,a in enumerate(final)]]
    df_layer_idx    = np.ones((pos_u.shape[0],1))*layer_num
    df_ti_pos       = pos_u
    df_strain       = np.ones((pos_u.shape[0],4))*np.nan
    df_u            = u
    all_data.append(np.hstack([df_layer_idx,
                                df_ti_pos, 
                                df_strain, 
                                df_u]))

# Save to CSV file
all_data = np.vstack(all_data)
np.savetxt(f'{angle}deg_strain.csv',all_data)