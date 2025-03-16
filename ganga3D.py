#this is my first work. error might occur. thanks for using it. :)

import sys
print(f"Python path: {sys.executable}")
import os
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from matchms import Spectrum, calculate_scores
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy

# IR rules for conformer feature extraction (unused now, kept for future)
IR_RULES = {
    3400: {"group": "[O]-[H]", "dihedral": 180},
    1720: {"group": "[C]=[O]", "dihedral": 180},
    1600: {"group": "[C]=[C]", "dihedral": 180},
    1200: {"group": "[C]-[O]", "dihedral": 120}
}

def ms_to_smiles(ms_file, library_mgf="C:/Users/is/Desktop/work/pubchem_subset.mgf"):
    """Convert MS data to a SMILES string by matching against a library."""
    # Load and clean MS data
    ms_data = pd.read_csv(ms_file, header=None)
    mz = pd.to_numeric(ms_data[0], errors='coerce')
    intensities = pd.to_numeric(ms_data[1], errors='coerce')
    valid_mask = ~np.isnan(mz) & ~np.isnan(intensities)
    if not valid_mask.any():
        print("Error: No valid m/z or intensity data.")
        return None
    mz_clean = mz[valid_mask].to_numpy()
    intensities_clean = intensities[valid_mask].to_numpy()
    sorted_indices = np.argsort(mz_clean)
    mz_sorted = mz_clean[sorted_indices]
    intensities_sorted = intensities_clean[sorted_indices]
    print(f"Raw m/z: {mz[:10]}")
    print(f"Cleaned m/z: {mz_clean[:10]}")
    print(f"Sorted m/z: {mz_sorted[:10]}")
    user_spectrum = Spectrum(mz=mz_sorted, intensities=intensities_sorted / intensities_sorted.max())
    print(f"User spectrum m/z: {user_spectrum.mz[:10]}")

    # Check for library file
    if not os.path.exists(library_mgf):
        print(f"Warning: Library MGF not found at '{library_mgf}'. Defaulting to aspirin SMILES.")
        return "CC(=O)OC1=CC=CC=C1C(=O)O"
    library_spectra = list(load_from_mgf(library_mgf))
    print(f"Library spectra loaded: {len(library_spectra)}")
    if len(library_spectra) < 100:
        print(f"Warning: Small library ({len(library_spectra)} entries)—results may be limited.")
    
    # Perform spectral matching
    cosine = CosineGreedy(tolerance=1.0)
    scores = calculate_scores([user_spectrum], library_spectra, cosine)
    score_pairs = [(i, score[1]['score']) for i, score in enumerate(scores.scores) if score[1] is not None and 'score' in score[1]]
    print(f"Score values generated: {len(score_pairs)}")
    
    # Fallback to aspirin if no good match
    if not score_pairs or max(score[1] for _, score in score_pairs) < 0.5:
        print(f"Warning: No match above 0.5 in '{library_mgf}'. Defaulting to aspirin SMILES.")
        return "CC(=O)OC1=CC=CC=C1C(=O)O"
    
    # Get best match
    best_match_idx, best_score = max(score_pairs, key=lambda x: x[1])
    best_match = library_spectra[best_match_idx]
    print(f"Best match metadata: {best_match.metadata}")
    print(f"Best match score: {best_score}")
    smiles = best_match.metadata.get("smiles", None) or best_match.metadata.get("computed_smiles", None)
    
    # Handle missing SMILES
    if not smiles:
        name = best_match.metadata.get("compound_name", None)
        if name and "aspirin" in name.lower():
            print("Warning: No SMILES in metadata, but compound is aspirin-like.")
            smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"
        else:
            print("Error: No SMILES found in library match.")
            return None
    print(f"Predicted SMILES: {smiles}")
    return smiles

def generate_3d_from_ms_ir(ms_file, ir_file, output_dir):
    """Generate a 3D SDF file from MS and optional IR data."""
    if not os.path.exists(ms_file):
        print(f"Error: MS file '{ms_file}' not found.")
        return
    
    smiles = ms_to_smiles(ms_file)
    if not smiles:
        return
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error: Invalid SMILES '{smiles}'")
        return
    mol = Chem.AddHs(mol)
    print(f"Generated molecule with {mol.GetNumAtoms()} atoms from SMILES: {smiles}")

    if ir_file and os.path.exists(ir_file):
        ir_data = pd.read_csv(ir_file, header=None)
        ir_peaks = ir_data[0].tolist()
        AllChem.EmbedMultipleConfs(mol, numConfs=50, randomSeed=42)
        energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
        if not energies or mol.GetNumConformers() == 0:
            print("Error: No conformers generated")
            return
        # Filter out failed optimizations (energy = -1)
        valid_energies = [(i, e[1]) for i, e in enumerate(energies) if e[1] != -1]
        if not valid_energies:
            print("Error: All conformers failed optimization")
            return
        best_conf_id, best_energy = min(valid_energies, key=lambda x: x[1])
        print(f"Generated {mol.GetNumConformers()} conformers, selected {best_conf_id} with energy {best_energy}")
    else:
        success = AllChem.EmbedMolecule(mol, randomSeed=42)
        if success == -1:
            print("Error: Single conformer embedding failed")
            return
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        best_conf_id = 0
        print("Generated single conformer without IR data")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(ms_file))[0]
    sdf_file = os.path.join(output_dir, f"{base_name}.sdf")
    writer = Chem.SDWriter(sdf_file)
    writer.write(mol, confId=int(best_conf_id))
    writer.close()
    print(f"Generated '{sdf_file}'")

# User inputs
ms_file = input("Enter MS data CSV file path (e.g., C:/Users/is/taxol_ms.csv): ")
ir_file = input("Enter FTIR data CSV file path (optional, press Enter to skip): ")
output_dir = input("Enter output directory (e.g., F:/paris): ")

# Execute
generate_3d_from_ms_ir(ms_file, ir_file, output_dir)