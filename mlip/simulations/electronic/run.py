from ase.calculators.aims import Aims
from ase.io import read
import yaml
import sys

input_file = sys.argv[1]

atoms = read('geometry.in')
with open(input_file, 'r') as f:
    aims_params = yaml.safe_load(f)

calc = Aims(**aims_params)
atoms.calc = calc
energy = atoms.get_potential_energy()