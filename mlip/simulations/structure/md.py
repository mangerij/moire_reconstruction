import sys

from ase.io import read
from ase import units
from ase.md import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from mace.calculators import MACECalculator

angle = sys.argv[1]
timestep = 1.0
friction = 0.01
temperature = 300

device = 'cuda'
model_path = 'MACE_stagetwo_compiled.model'

calc = MACECalculator(model_paths=model_path, device=device)

atoms = read(f'{angle}deg_relaxed.traj')

atoms.calc = calc
MaxwellBoltzmannDistribution(atoms, temperature_K=temperature)

dyn = Langevin(
            atoms,
            timestep=timestep * units.fs,
            temperature_K=temperature,
            friction=friction / units.fs,
            trajectory=f'md_{angle}.traj',
            logfile=f'md_{angle}.log',
            loginterval=10,
            )
dyn.run(50000)