from rdkit import Chem
import os
import pandas as pd
from pathlib import Path
import ase
import ase.io
from rdkit.Chem import AllChem
from dscribe.descriptors import SOAP
from sklearn.decomposition import PCA
import random
import numpy as np

def replace_atoms(mol, old_atom_num, new_atom_num):
    mol = Chem.RWMol(mol)
    indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == old_atom_num]

    for idx in indices:
        mol.GetAtomWithIdx(idx).SetAtomicNum(new_atom_num)

    return mol, indices


def rdkit_mol_to_ase_atoms(rdkit_mol: Chem.Mol, conformer_idx) -> ase.Atoms:
    """Convert an RDKit molecule to an ASE Atoms object.

    Args:
        rdkit_mol: RDKit molecule object.

    Returns:
        ASE Atoms object.
    """
    ase_atoms = ase.Atoms(
        numbers=[
            atom.GetAtomicNum() for atom in rdkit_mol.GetAtoms()
        ],
        positions=rdkit_mol.GetConformer(conformer_idx).GetPositions()
    )
    return ase_atoms

def smiles_to_structures(smiles_list, output_dir, num_conformers=10):
    atoms_list = []
    atom_index_list = []
    for smile in smiles_list:
        try:
            # Convert SMILES to molecule
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                print(f"Invalid SMILES string: {smile}")
                return

            # Add explicit hydrogens
            mol = Chem.AddHs(mol)

            # Replace Si with C so that the UFF force field relaxed the correct structure
            modified_mol, si_indices = replace_atoms(mol, 14, 6)  # Si to C

            # Generate conformers
            AllChem.EmbedMultipleConfs(modified_mol, randomSeed=11, numConfs=num_conformers)

            # Optimize conformers using UFF
            conformer_energies = []
            for conf_id in range(modified_mol.GetNumConformers()):
                AllChem.UFFOptimizeMolecule(modified_mol, confId=conf_id)
                energy = AllChem.UFFGetMoleculeForceField(modified_mol, confId=conf_id).CalcEnergy()
                conformer_energies.append((conf_id, energy))

            # Find the lowest energy conformer
            lowest_energy_conformer = min(conformer_energies, key=lambda x: x[1])[0]
            modified_mol.SetProp("_Name",
                                 f"Lowest energy conformer (energy={conformer_energies[lowest_energy_conformer][1]:.4f})")

            # Replace C back to Si at the original indices
            for idx in si_indices:
                modified_mol.GetAtomWithIdx(idx).SetAtomicNum(14)  # C back to Si

            # Create a xyz filename from the SMILES string
            final_xyz_file = os.path.join(output_dir, f"{smile}.pdb")

            # Write the lowest energy conformer to a xyz file
            with open(final_xyz_file, 'w') as f:
                f.write(Chem.MolToPDBBlock(modified_mol, confId=lowest_energy_conformer))

            # Get ase atoms object and append to list
            atoms = ase.io.read(final_xyz_file)
            for atom in atoms:
                if atom.symbol == 'Si':
                    atom.symbol = 'C'
            atoms_list.append(atoms)

            # Append index of center atom (Si) to list to track the center atom, which was encoded in the SMILES as Si.
            assert len(si_indices) == 1, f"Expected 1 Si atom, got {len(si_indices)}"
            atom_index_list.append(si_indices)

        except Exception as e:
            print(f"Failed to convert SMILES: {smile}")
            raise UserWarning(f"Error: {e}")

    return atoms_list, atom_index_list

def get_dft_atoms(smiles_list, dft_dir):
    dft_atoms = []
    dft_atoms_index = []
    for smile in smiles_list:
        dft_filepath = Path(dft_dir, f'{smile}.pdb')
        atoms = ase.io.read(dft_filepath)
        dft_atoms.append(atoms)
        # Replace Si placeholder atom with C and record the index of this atom
        assert len([1 for atom in atoms.symbols if atom == 'Si']) == 1, 'Expected 1 Si atom only per molecule.'
        for idx, atom in enumerate(atoms):
            if atom.symbol == 'Si':
                atom.symbol = 'C'
                dft_atoms_index.append([idx])
                break

    return dft_atoms, dft_atoms_index


if __name__ == '__main__':

    features_dir = Path('..', 'data')
    uff_dir = Path('..', 'data', 'uff_geometries')
    dft_dir = Path('..', 'data', 'dft_geometries')

    input_csv = Path(features_dir, 'fukui.csv')
    output_csv = Path(features_dir, 'fukui_soap_pca.csv')



    # Set random seeds for deterministic results. Probably not needed but just in case.
    random.seed(42)
    np.random.seed(42)

    # Make output directory
    uff_dir = Path(uff_dir).resolve()
    uff_dir.mkdir(exist_ok=True, parents=True)
    print('Output UFF directory:', uff_dir)

    # Read data
    input_csv = Path(input_csv).resolve()
    df = pd.read_csv(input_csv)
    smiles_list = df['smiles'].tolist()

    # Generate 3D structures from SMILES
    uff_atoms, uff_atoms_index = smiles_to_structures(smiles_list, output_dir=uff_dir, num_conformers=10)

    # Read in DFT structures from .pdb files
    dft_atoms, dft_atoms_index = get_dft_atoms(smiles_list, dft_dir)

    # Calculate SOAP features for uff and dft structures
    species = ['H', 'C', 'N', 'F', 'Cl', 'Br', 'O', 'S']
    soap = SOAP(
        species=species,
        periodic=False,
        r_cut=10.0,
        sigma=0.1,
        n_max=3,
        l_max=7,
    )
    # UFF structures
    uff_soap_features = soap.create(uff_atoms, centers=uff_atoms_index)
    uff_soap_features = uff_soap_features.reshape(len(uff_soap_features), -1)
    uff_soap_labels = [f'uffsoap_{i}' for i in range(uff_soap_features.shape[1])]
    uff_soap_df = pd.DataFrame(uff_soap_features, columns=uff_soap_labels)
    # DFT structures
    dft_soap_features = soap.create(dft_atoms, centers=dft_atoms_index)
    dft_soap_features = dft_soap_features.reshape(len(dft_soap_features), -1)
    dft_soap_labels = [f'dftsoap_{i}' for i in range(dft_soap_features.shape[1])]
    dft_soap_df = pd.DataFrame(dft_soap_features, columns=dft_soap_labels)
    # Append SOAP features to df
    df = pd.concat([df, uff_soap_df, dft_soap_df], axis=1)

    # Do PCA of SOAP features and append to df
    pca = PCA(whiten=True)
    # UFF
    uff_principal_components = pca.fit_transform(df[uff_soap_labels].to_numpy())
    uff_df_pca = pd.DataFrame(uff_principal_components, columns=[f'uffpca_{i}' for i in range(uff_principal_components.shape[1])])
    # DFT
    dft_principal_components = pca.fit_transform(df[dft_soap_labels].to_numpy())
    dft_df_pca = pd.DataFrame(dft_principal_components, columns=[f'dftpca_{i}' for i in range(dft_principal_components.shape[1])])
    df = pd.concat([df, uff_df_pca, dft_df_pca], axis=1)

    # Save the dataframe
    df.to_csv(output_csv, index=False)

    print('Done!')


    # # %% Check with old csv
    old_csv = '/Users/timosommer/PhD/projects/others/Manting_Fukui_indices/data/OLD_fukui_soap_pca.csv'
    df_old = pd.read_csv(old_csv)

    # print different columns:
    print(f'Different columns: {sorted(set(df.columns) - set(df_old.columns))}')
    pd.testing.assert_frame_equal(df, df_old)
    print('Dataframes are equal!')

