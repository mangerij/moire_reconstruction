"""
Simulate 4D-STEM data for all structures in a thickness-series trajectory.

The input trajectory should contain one structure for each thickness.
For each structure, the script saves:

1. Raw 4D-STEM measurement as .zarr
2. Center-of-mass image as .png and .npy
3. Bright-field image as .png and .npy
"""

from pathlib import Path

import abtem
import matplotlib.pyplot as plt
import numpy as np
from ase.io.trajectory import Trajectory

script_dir = Path(__file__).resolve().parent

# -----------------------------
# User input
# -----------------------------

structure_file = script_dir / "input_structures" / "md_4.98_thickness_series_02to12nm.traj"


# -----------------------------
# Output folders
# -----------------------------

output_abtem_dir = script_dir / "output_abtem/"
output_image_dir = script_dir / "output_images/"
output_npy_dir = script_dir / "output_npy/"

output_abtem_dir.mkdir(parents=True, exist_ok=True)
output_image_dir.mkdir(parents=True, exist_ok=True)
output_npy_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# abTEM settings
# -----------------------------

abtem.config.set({"device": "gpu"})
abtem.config.set({"dask.chunk-size-gpu": "2048 MB"})


# -----------------------------
# Microscope settings
# -----------------------------

beam_energy = 300e3  # eV
convergence_angle = 2  # mrad
spherical_aberration = 1590  # A
probe_sampling = 0.5  # A


# -----------------------------
# Potential settings
# -----------------------------

potential_sampling = 0.05  # A
slice_thickness = 1.0  # A
projection_type = "infinite"


# -----------------------------
# Scan and detector settings
# -----------------------------

z_vacuum = 2.0  # A

scan_start = [0, 0]
scan_end = [1, 1]
fractional_scan = True

max_collection_angle = 3  # mrad
pixelated_detector = abtem.PixelatedDetector(max_angle=max_collection_angle)

bf_inner_angle = 0  # mrad
bf_outer_angle = 2  # mrad


# -----------------------------
# Load trajectory
# -----------------------------

if not structure_file.exists():
    raise FileNotFoundError(f"Missing input file: {structure_file}")

trajectory = Trajectory(structure_file)
structure_label = structure_file.stem

print(f"Loaded trajectory: {structure_file}")
print(f"Number of structures: {len(trajectory)}")


# -----------------------------
# Simulate each thickness
# -----------------------------

for structure_index, sample in enumerate(trajectory):
    sample = sample.copy()

    # Add vacuum along z to avoid interaction between periodic images.
    sample.center(vacuum=z_vacuum, axis=2)

    # Estimate sample thickness from the simulation cell.
    sample_height_nm = (sample.cell[2, 2] - 2 * z_vacuum) / 10

    calculation_label = (
        f"{structure_label}_index_{structure_index:02d}_"
        f"{sample_height_nm:.1f}nm"
    )

    print(f"\nSimulating {calculation_label}")

    # The interface is assumed to be halfway through the structure.
    entrance_z = sample.positions[:, 2].min()
    exit_z = sample.positions[:, 2].max()
    interface_z = 0.5 * (entrance_z + exit_z)

    print(f"Interface position: {interface_z:.2f} A")

    # Define the electron probe.
    probe = abtem.Probe(
        energy=beam_energy,
        semiangle_cutoff=convergence_angle,
        Cs=spherical_aberration,
        sampling=probe_sampling,
    )

    # Focus the probe at the interface.
    probe.aberrations.defocus = interface_z

    # Calculate the electrostatic potential.
    potential = abtem.Potential(
        sample,
        sampling=potential_sampling,
        parametrization="lobato",
        slice_thickness=slice_thickness,
        projection=projection_type,
    )

    # Scan the full x-y simulation cell.
    scan = abtem.GridScan(
        start=scan_start,
        end=scan_end,
        sampling=probe_sampling,
        potential=potential,
        fractional=fractional_scan,
    )

    probe.grid.match(potential)

    # Run the 4D-STEM simulation.
    measurement_4d = probe.scan(
        potential,
        scan=scan,
        detectors=pixelated_detector,
        lazy=True,
    )

    zarr_filename = output_abtem_dir / f"{calculation_label}_4dstem.zarr"

    print("Saving raw 4D-STEM measurement")
    measurement_4d.to_zarr(zarr_filename, compute=True)

    print("Loading raw 4D-STEM measurement")
    measurement_4d = abtem.array.from_zarr(zarr_filename)

    # Calculate center-of-mass image.
    print("Calculating center-of-mass image")
    center_of_mass = measurement_4d.center_of_mass()

    com_npy_filename = output_npy_dir / f"{calculation_label}_center_of_mass.npy"
    com_image_filename = output_image_dir / f"{calculation_label}_center_of_mass.png"

    np.save(com_npy_filename, center_of_mass.array)

    center_of_mass.show(cbar=True, figsize=(10, 10))
    plt.savefig(com_image_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {com_npy_filename}")
    print(f"Saved: {com_image_filename}")

    # Calculate bright-field image by integrating the central diffraction disk.
    print("Calculating bright-field image")
    bright_field = measurement_4d.integrate_radial(
        inner=bf_inner_angle,
        outer=bf_outer_angle,
    )

    bf_array = np.array(bright_field.array)

    bf_npy_filename = output_npy_dir / f"{calculation_label}_bright_field.npy"
    bf_image_filename = output_image_dir / f"{calculation_label}_bright_field.png"

    np.save(bf_npy_filename, bf_array)

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(bf_array, cmap="gray", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig(bf_image_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {bf_npy_filename}")
    print(f"Saved: {bf_image_filename}")