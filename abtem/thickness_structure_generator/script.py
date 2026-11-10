"""
Generate a thickness series of twisted STO bilayer interfaces.

Starting from rigid and relaxed interface structures, this script extends the
interface by adding repeated rigid STO layers above and below the interface region.
The resulting structures are written to ASE trajectory files.
"""

from pathlib import Path

from ase.io.trajectory import Trajectory
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor


# -----------------------------
# Parameters
# -----------------------------

lattice_parameter = 3.896899738  # A, cubic STO lattice parameter - This parameter is used to calculate the interlayer distance

interlayer_vacuum = lattice_parameter / 4 #vacuume in A used to space the top and bottom layers added - corresponds to half the distance between the TiO2 and SrO layers
final_vacuum = 2.0  #vacuume in A at entrance and exit of the sample - included to make sure the whole potential is included

layers_series = [0,1,2,3,4,5,6,7,8,9,10,11,12,13] #A series of layers to be added on both sides of the interface - 1 layer corresponds to 1 SrO-TiO2 block

top_layer_z_min = 20.0      # A, atoms above this z-position define the top STO block
bottom_layer_z_max = 4.0    # A, atoms below this z-position define the bottom STO block

# -----------------------------
# Input/output directories
# -----------------------------

script_dir = Path(__file__).resolve().parent

input_dir = script_dir / "input_structures"
output_dir = script_dir / "output_structures"

output_dir.mkdir(parents=True, exist_ok=True)

relaxed_interface_file = input_dir / "md_SrO-TiO2_4.98.cif"
rigid_interface_file = input_dir / "rigid_SrO-TiO2_4.98.cif"

output_files = {
    "rigid": output_dir / "rigid_4.98_thickness_series_02to12nm.traj",
    "md": output_dir / "md_4.98_thickness_series_02to12nm.traj",
}

# -----------------------------
# Helper functions
# -----------------------------

def read_cif_as_ase_atoms(filepath, vacuum=2.0):
    """Read a CIF file and return it as an ASE Atoms object."""
    structure = Structure.from_file(filepath)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.center(vacuum=vacuum, axis=2)
    return atoms


def add_layers(
    sample,
    layers,
    top_layer,
    bottom_layer,
    lattice_parameter,
    vacuum=2.0,
):
    """
    Extend an interface structure by adding repeated STO layers above and below.

    Parameters
    ----------
    sample : ase.Atoms
        Interface structure to be extended.
    layers : int
        Number of repeated STO blocks added on each side.
    top_layer : ase.Atoms
        STO block added above the interface.
    bottom_layer : ase.Atoms
        STO block added below the interface.
    lattice_parameter : float
        STO lattice parameter in A.
    vacuum : float
        Vacuum added along z after stitching.
    """
    if layers == 0:
        atoms = sample.copy()
        atoms.center(vacuum=vacuum, axis=2)
        return atoms

    top = top_layer.repeat((1, 1, layers))
    bottom = bottom_layer.repeat((1, 1, layers))

    z_max_sample = sample.positions[:, 2].max()
    z_min_sample = sample.positions[:, 2].min()

    z_min_top = top.positions[:, 2].min()
    z_max_bottom = bottom.positions[:, 2].max()

    top.translate((0, 0, z_max_sample + lattice_parameter / 2 - z_min_top))
    bottom.translate((0, 0, z_min_sample - lattice_parameter / 2 - z_max_bottom))

    atoms = sample + top + bottom
    atoms.center(vacuum=vacuum, axis=2)

    return atoms


# -----------------------------
# Load structures
# -----------------------------

relaxed_interface = read_cif_as_ase_atoms(relaxed_interface_file, vacuum=final_vacuum)
rigid_interface = read_cif_as_ase_atoms(rigid_interface_file, vacuum=final_vacuum)


# -----------------------------
# Extract STO building blocks
# -----------------------------

# The rigid interface is used to define the repeating STO blocks.
# These blocks are then reused to stich an extention onto the original interfaces

top_mask = rigid_interface.positions[:, 2] > top_layer_z_min
bottom_mask = rigid_interface.positions[:, 2] < bottom_layer_z_max

top_layer = rigid_interface[top_mask].copy()
bottom_layer = rigid_interface[bottom_mask].copy()

top_layer.center(vacuum=interlayer_vacuum, axis=2)
bottom_layer.center(vacuum=interlayer_vacuum, axis=2)


# -----------------------------
# Generate and save thickness series
# -----------------------------

samples = {
    "rigid": rigid_interface,
    "md": relaxed_interface,
}

for sample_name, sample in samples.items():
    traj_filename = output_files[sample_name]

    with Trajectory(traj_filename, "w") as traj:
        for layers in layers_series:
            print(f"Sample: {sample_name}, layers added per side: {layers}")

            atoms = add_layers(
                sample=sample.copy(),
                layers=layers,
                top_layer=top_layer,
                bottom_layer=bottom_layer,
                lattice_parameter=lattice_parameter,
                vacuum=final_vacuum,
            )

            atoms.info["sample"] = sample_name
            atoms.info["added_layers_per_side"] = layers

            traj.write(atoms)

    print(f"Saved trajectory: {traj_filename}")