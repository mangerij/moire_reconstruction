"""
Simulate HAADF and LAADF STEM images for all structures in a thickness-series trajectory.

The input trajectory should contain one structure for each thickness.
The script saves one image and one NumPy array for each detector and thickness.
"""

from pathlib import Path

import abtem
import matplotlib.pyplot as plt
import numpy as np
from ase.io.trajectory import Trajectory


# -----------------------------
# User inputs
# -----------------------------

script_dir = Path(__file__).resolve().parent

structure_file = script_dir / "input_structures" / "md_thickness_series_02_to_20nm.traj"

label = "STO_5deg_relaxed"


# -----------------------------
# Output folders
# -----------------------------

output_image_dir = script_dir / "output_images"
output_npy_dir = script_dir / "output_npy"

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
convergence_angle = 30  # mrad
spherical_aberration = 1590  # A
probe_sampling = 0.5  # A


# -----------------------------
# Simulation settings
# -----------------------------

potential_sampling = 0.03  # A
slice_thickness = 1.0  # A
projection_type = "infinite"

z_vacuum = 2.0  # A

electron_dose = 4201478  # e / A^2
random_seed = 100


# -----------------------------
# Image post-processing settings
# -----------------------------

interpolation_sampling = 0.1  # A / pixel
gaussian_filter_width = 0.5
crop_extent = (100, 100)  # A
tile_factor = (2, 2)

scale_bar_length_nm = 2


# -----------------------------
# Detectors
# -----------------------------

detectors = {
    "haadf": abtem.AnnularDetector(inner=64, outer=200),
    "laadf": abtem.AnnularDetector(inner=32, outer=64),
}


# -----------------------------
# Small plotting helper
# -----------------------------

def add_scalebar(ax, image_shape, length_nm=2, pixel_size_A=0.1):
    """Add a white scale bar to an image."""
    length_px = (length_nm * 10) / pixel_size_A

    height, width = image_shape
    x0 = width - length_px - 25
    x1 = width - 25
    y = 25

    ax.plot([x0, x1], [y, y], color="white", lw=3, solid_capstyle="butt")
    ax.text(
        (x0 + x1) / 2,
        y + 70,
        f"{length_nm} nm",
        color="white",
        ha="center",
        va="top",
        fontsize=8,
    )


# -----------------------------
# Load trajectory
# -----------------------------

if not structure_file.exists():
    raise FileNotFoundError(f"Missing input file: {structure_file}")

trajectory = Trajectory(structure_file)

print(f"Loaded trajectory: {structure_file}")
print(f"Number of structures: {len(trajectory)}")


# -----------------------------
# Simulate each structure in the trajectory
# -----------------------------

for structure_index, sample in enumerate(trajectory):
    sample = sample.copy()

    # Add vacuum along z to avoid interaction between periodic images.
    sample.center(vacuum=z_vacuum, axis=2)

    # Estimate sample thickness from the simulation cell.
    sample_height_nm = (sample.cell[2, 2] - 2 * z_vacuum) / 10

    thickness_label = f"{sample_height_nm:.1f}nm"
    calculation_label = f"{label}_index_{structure_index:02d}_{thickness_label}"

    print(f"\nSimulating {calculation_label}")

    # The interface is assumed to be halfway through the structure.
    entrance_z = sample.positions[:, 2].min()
    exit_z = sample.positions[:, 2].max()
    interface_z = 0.5 * (entrance_z + exit_z)

    # Define probe and focus it at the interface.
    probe = abtem.Probe(
        energy=beam_energy,
        semiangle_cutoff=convergence_angle,
        Cs=spherical_aberration,
        sampling=probe_sampling,
    )

    probe.aberrations.defocus = interface_z

    # Calculate electrostatic potential.
    potential = abtem.Potential(
        sample,
        sampling=potential_sampling,
        parametrization="lobato",
        slice_thickness=slice_thickness,
        projection=projection_type,
    )

    # Scan the full x-y cell.
    scan = abtem.GridScan(
        start=[0, 0],
        end=[1, 1],
        sampling=probe_sampling,
        potential=potential,
        fractional=True,
    )

    probe.grid.match(potential)

    # Run HAADF and LAADF simulations.
    measurements = probe.scan(
        potential,
        scan=scan,
        detectors=list(detectors.values()),
    ).compute()

    measurements = abtem.stack(measurements, list(detectors.keys()))

    # Post-process simulated images.
    measurements = measurements.interpolate(sampling=interpolation_sampling)
    measurements = measurements.gaussian_filter(gaussian_filter_width)
    measurements = measurements.tile(tile_factor)
    measurements = measurements.poisson_noise(
        dose_per_area=electron_dose,
        seed=random_seed,
    )

    # Save one image and one NumPy array per detector.
    for detector_index, detector_name in enumerate(detectors.keys()):
        image = measurements[detector_index].crop(extent=crop_extent)
        array = np.array(image.array)

        image_filename = output_image_dir / f"{calculation_label}_{detector_name}.png"
        npy_filename = output_npy_dir / f"{calculation_label}_{detector_name}.npy"

        np.save(npy_filename, array)

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(array, cmap="gray", origin="lower")
        ax.set_xticks([])
        ax.set_yticks([])

        add_scalebar(
            ax,
            image_shape=array.shape,
            length_nm=scale_bar_length_nm,
            pixel_size_A=interpolation_sampling,
        )

        plt.savefig(image_filename, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved image: {image_filename}")
        print(f"Saved array: {npy_filename}")