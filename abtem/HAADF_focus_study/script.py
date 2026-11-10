"""
Simulate LAADF and HAADF STEM images from an STO interface structure.

User inputs:
1. structure_file
2. defocus_relative_to_interface
3. electron_dose
"""

from pathlib import Path

import abtem
import matplotlib.pyplot as plt
import numpy as np
from ase.io import read


# -----------------------------
# User inputs
# -----------------------------

label = '4.98_rigid_0defocus_32319edose'
structure_file = Path("input_structures/20nmtotal_rigid_SrO-TiO2_4.98.traj")
defocus_relative_to_interface = 0.0  # Å
electron_dose = 32319 # e / Å^2


# -----------------------------
# Output directories
# -----------------------------

output_abtem_dir = Path("output/abtem/")
output_image_dir = Path("output/images/")
output_npy_dir = Path("output/npy/")

output_image_dir.mkdir(parents=True, exist_ok=True)
output_npy_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# abTEM settings
# -----------------------------

abtem.config.set({"device": "gpu"})
abtem.config.set({"dask.chunk-size-gpu": "2048 MB"})


# -----------------------------
# Microscope parameters
# -----------------------------

beam_energy = 300e3  # eV
convergence_angle = 30  # mrad
spherical_aberration = 1590  # Å
probe_sampling = 0.5  # Å


# -----------------------------
# Potential parameters
# -----------------------------

potential_sampling = 0.03  # Å
slice_thickness = 1.0  # Å
projection_type = "infinite"


# -----------------------------
# Post-processing parameters
# -----------------------------

z_vacuum = 4.0  # Å

interpolation_sampling = 0.1  # Å / pixel
gaussian_filter_width = 0.5
crop_extent = (100, 100)  # Å

tile_factor = (3, 3)
random_seed = 100

scale_bar_length_nm = 2


# -----------------------------
# Detectors
# -----------------------------

detectors = {
    "haadf": abtem.AnnularDetector(inner=64, outer=200),
    "laadf": abtem.AnnularDetector(inner=32, outer=64),
}


# -----------------------------
# Helper functions
# -----------------------------

def add_scalebar(ax, image_shape, length_nm=2, pixel_size_A=0.1):
    """Add a white scale bar to an image."""
    length_px = (length_nm * 10) / pixel_size_A  # nm -> Å -> pixels

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


def save_image(array, filename, pixel_size_A, scale_bar_length_nm=2):
    """Save a grayscale image with no ticks and a scale bar."""
    fig, ax = plt.subplots(figsize=(10, 10))

    ax.imshow(array, cmap="gray", origin="lower")
    ax.set_xticks([])
    ax.set_yticks([])

    add_scalebar(
        ax,
        image_shape=array.shape,
        length_nm=scale_bar_length_nm,
        pixel_size_A=pixel_size_A,
    )

    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Load structure
# -----------------------------

sample = read(structure_file, index=0)
sample.center(vacuum=z_vacuum, axis=2)

structure_name = structure_file.stem


# -----------------------------
# Define interface and defocus
# -----------------------------

entrance_z = sample.positions[:, 2].min()
exit_z = sample.positions[:, 2].max()

interface_z = (entrance_z + exit_z)/2
defocus = interface_z + defocus_relative_to_interface

print(f"Structure file: {structure_file}")
print(f"Interface position: {interface_z:.2f} Å")
print(f"Relative defocus: {defocus_relative_to_interface:.2f} Å")
print(f"Absolute defocus: {defocus:.2f} Å")
print(f"Electron dose: {electron_dose:.2e} e/Å²")


# -----------------------------
# Set up STEM simulation
# -----------------------------

probe = abtem.Probe(
    energy=beam_energy,
    semiangle_cutoff=convergence_angle,
    Cs=spherical_aberration,
    sampling=probe_sampling,
)

probe.aberrations.defocus = defocus

potential = abtem.Potential(
    sample,
    sampling=potential_sampling,
    parametrization="lobato",
    slice_thickness=slice_thickness,
    projection=projection_type,
)

scan = abtem.GridScan(
    start=[0, 0],
    end=[1, 1],
    sampling=probe_sampling,
    potential=potential,
    fractional=True,
)

probe.grid.match(potential)


# -----------------------------
# Run simulation
# -----------------------------

measurements = probe.scan(
    potential,
    scan=scan,
    detectors=detectors,
)
abtem_filename = output_abtem_dir / f'{label}.zarr'
measurements.to_zarr(url = abtem_filename, compute = True) #calculating to disk
measurements = abtem.from_zarr(abtem_filename) #reading back in measurement

measurements = abtem.stack(
    measurements,
    list(detectors.keys()),
).compute() #stacking measurement with detector labels



# -----------------------------
# Post-process simulation
# -----------------------------

measurements = measurements.interpolate(sampling=interpolation_sampling)
measurements = measurements.gaussian_filter(gaussian_filter_width)
measurements = measurements.tile(tile_factor)
measurements = measurements.poisson_noise(
    dose_per_area=electron_dose,
    seed=100,
)


# -----------------------------
# Save HAADF and LAADF images
# -----------------------------

for detector_index, detector_name in enumerate(detectors.keys()):
    image = measurements[detector_index].crop(extent=crop_extent)
    array = np.array(image.array)

    image_filename = output_image_dir / f"{label}_{detector_name}.png"
    

    npy_filename = output_npy_dir /  f"{label}_{detector_name}.npy"
    

    save_image(
        array=array,
        filename=image_filename,
        pixel_size_A=interpolation_sampling,
        scale_bar_length_nm=scale_bar_length_nm,
    )

    np.save(npy_filename, array)

    print(f"Saved image: {image_filename}")
    print(f"Saved array: {npy_filename}")