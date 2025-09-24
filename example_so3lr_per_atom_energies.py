import numpy as np
from ase import Atoms
from mlff.md.calculator_sparse import mlffCalculatorSparse
from ase.io import read

def make_ase_calculator_with_per_atom_components(path,
    lr_cutoff=12.,
    dispersion_energy_lr_cutoff_damping=2.,
    calculate_stress=False,
    dtype=np.float32
    ):
    """
    Modified version of your make_ase_calculator function that enables per-atom energy components.
    """

    # The key change: set output_per_atom_energy_components=True
    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=path,
        lr_cutoff=lr_cutoff,
        dispersion_energy_lr_cutoff_damping=dispersion_energy_lr_cutoff_damping,
        from_file=False,
        calculate_stress=calculate_stress,
        dtype=dtype,
        output_per_atom_energy_components=True  # <-- This is the new parameter
    )

    return calc

def modified_so3lr_workflow_example():
    """
    Example of how to modify your so3lr get_ase_fep.py workflow.
    """

    model_path = '/home/hamza/github/sqm_sf/so3lr/so3lr/params'
    # In your get_ase_fep.py, modify the make_ase_calculator call:

    # OLD:
    # calc = make_ase_calculator(model_path,
    #     calculate_stress=False,
    #     dtype=np.float32
    # )

    # NEW:
    calc = mlffCalculatorSparse.create_from_ckpt_dir(
        ckpt_dir=model_path,
        calculate_stress=False,
        dtype=np.float32,
        output_per_atom_energy_components=True  # Enable per-atom components
    )
    system = read('/home/hamza/github/sqm_sf/datasets/fep-benchmark/cdk8/ligands/ligands1.xyz')
    system.calc = calc

    # Get total energy (same as before)
    total_energy = system.get_potential_energy()
    
    # NEW: Get per-atom energy components
    per_atom_components = calc.get_per_atom_energy_components()
    
    print(f"Total Energy: {total_energy} eV")
    for component, values in per_atom_components.items():
        print(f"{component.capitalize()} per-atom energies: {values} eV")

if __name__ == "__main__":
    print("=== MLFF Per-Atom Energy Components for so3lr Integration ===\n")
    modified_so3lr_workflow_example()

