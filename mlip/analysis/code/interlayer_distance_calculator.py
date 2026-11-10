import numpy as np

class InterlayerDistanceCalculator:
    def __init__(self, atoms, cutoff):
        self.atoms = atoms
        self.cutoff = cutoff
        self.layer_indices = None

    def calculate_interlayer_distance(self, layer_indices):
        af = self.atoms.copy()
 
        if abs(af.cell[0][0] - af.cell[1][1]) > 1e-5:
            raise Exception('Function assumes that Moire unit cell is square and aligned to xy')

        top_idxs = np.array(af.info['all_layers_idx'][layer_indices[1]])
        bot_idxs = np.array(af.info['all_layers_idx'][layer_indices[0]])

        mask_out_oxygen = [True if a.symbol in ['Sr','Ti'] else False for a in af[top_idxs]]
        top_idxs = top_idxs[mask_out_oxygen]
        mask_out_oxygen = [True if a.symbol in ['Sr','Ti'] else False for a in af[bot_idxs]]
        bot_idxs = bot_idxs[mask_out_oxygen]

        x = []
        y = []
        z_d = []

        for i in top_idxs:
            distances = af.get_distances(i, bot_idxs, mic=True, vector=True)
            xy_bool = np.sqrt(distances[:,0]**2+distances[:,1]**2) < self.cutoff

            # raise error if no atoms found within cutoff
            if len(distances[xy_bool,2]) == 0: 
                raise Exception('No atoms found within cutoff')
            d_local_mean = abs(distances[xy_bool,2].mean())
            x.append(af[i].position[0])
            y.append(af[i].position[1])
            z_d.append(d_local_mean)
        self.x = np.array(x)
        self.y = np.array(y)
        self.z_d = np.array(z_d)
        self.layer_indices = layer_indices

    def get_interlayer_distance(self, layer_indices):
        if hasattr(self, 'x') and hasattr(self, 'y') and hasattr(self, 'z_d') and self.layer_indices == layer_indices:
            return self.x, self.y, self.z_d
        self.calculate_interlayer_distance(layer_indices)
        return self.x, self.y, self.z_d