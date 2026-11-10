import numpy as np
import sys

from ase.optimize import QuasiNewton
from ase.io import read

from mace.calculators import MACECalculator

angle = sys.argv[1]
fmax = 0.03
device = 'cuda'
model_path = 'MACE_stagetwo_compiled.model'

atoms = read(f'{angle}deg_rigid.cif')
calc = MACECalculator(model_paths=model_path, 
                      device=device)

atoms.calc = calc

qn = QuasiNewton(atoms, trajectory=f'{angle}deg_relaxed.traj')
qn.run(fmax=fmax)