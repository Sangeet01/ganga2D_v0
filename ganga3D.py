

#this is my first work. error might occur. thanks for using it. :)

import sys
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image
from scipy.signal import find_peaks
from rdkit import Chem
from rdkit.Chem import AllChem
from matchms import Spectrum, calculate_scores
from matchms.importing import load_from_mgf
from matchms.similarity import CosineGreedy

def process_file(file_path, is_ms=False):
    """Process CSV or image file to extract m/z or IR data."""
    if not os.path.exists(file_path):
        return None, None
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.csv':
        data = pd.read_csv(file_path, header=None)
        x = pd.to_numeric(data[0], errors='coerce')
        y = pd.to_numeric(data[1], errors='coerce')
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        if not valid_mask.any():
            print(f"Error: No valid data in '{file_path}'")
            return None, None
        x_clean = x[valid_mask].to_numpy()
        y_clean = y[valid_mask].to_numpy()
        if is_ms:
            sorted_indices = np.argsort(x_clean)
            return x_clean[sorted_indices].astype(float), y_clean[sorted_indices]
        return x_clean.astype(float), y_clean
    
    else:  # Image file
        img = Image.open(file_path).convert('L')
        img_array = np.array(img)
        denoised = cv2.GaussianBlur(img_array, (5, 5), 0)
        intensity_profile = np.mean(denoised, axis=0)
        peaks, _ = find_peaks(intensity_profile, height=np.max(intensity_profile) * 0.1, distance=5)
        x_values = np.arange(len(intensity_profile), dtype=float)
        if is_ms:
            return x_values[peaks], intensity_profile[peaks]
        return x_values, intensity_profile

def generate_fallback_smiles(mz, intensities):
    """Generate a fallback SMILES based on MS data."""
    max_mz = mz[np.argmax(intensities)]
    carbon_count = int(max_mz / 14)
    if carbon_count < 1:
        carbon_count = 1
    smiles = "C" * carbon_count
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        return Chem.MolToSmiles(mol)
    return "C"

def ms_to_smiles(ms_file, library_mgf="C:/Users/is/Desktop/work/pubchem_subset.mgf"):
    """Convert MS data (CSV or image) to SMILES."""
    mz, intensities = process_file(ms_file, is_ms=True)
    if mz is None or intensities is None:
        print("Error: Failed to process MS file.")
        return None
    
    user_spectrum = Spectrum(mz=mz, intensities=intensities / intensities.max())
    if not os.path.exists(library_mgf):
        print(f"Warning: Library MGF not found at '{library_mgf}'. Generating fallback SMILES.")
        return generate_fallback_smiles(mz, intensities)
    
    library_spectra = list(load_from_mgf(library_mgf))
    cosine = CosineGreedy(tolerance=1.0)
    scores = calculate_scores([user_spectrum], library_spectra, cosine)
    score_pairs = [(i, score[1]['score']) for i, score in enumerate(scores.scores) if score[1] is not None and 'score' in score[1]]
    
    if not score_pairs or max(score[1] for _, score in score_pairs) < 0.5:
        print("Warning: No match above 0.5. Generating fallback SMILES.")
        return generate_fallback_smiles(mz, intensities)
    
    best_match_idx, _ = max(score_pairs, key=lambda x: x[1])
    best_match = library_spectra[best_match_idx]
    smiles = best_match.metadata.get("smiles", None) or best_match.metadata.get("computed_smiles", None)
    if not smiles:
        print("Warning: No SMILES in library match. Generating fallback SMILES.")
        return generate_fallback_smiles(mz, intensities)
    print(f"Predicted SMILES: {smiles}")
    return smiles

def generate_3d_from_ms_ir(ms_file, ir_file, output_dir):
    """Generate 3D SDF and save SMILES in a .txt file."""
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
    
    if ir_file and os.path.exists(ir_file):
        ir_wavenumbers, _ = process_file(ir_file)
        if ir_wavenumbers is None:
            print("Warning: Failed to process IR file, proceeding without it.")
        else:
            AllChem.EmbedMultipleConfs(mol, numConfs=50, randomSeed=42)
            energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
            valid_energies = [(i, e[1]) for i, e in enumerate(energies) if e[1] != -1]
            if not valid_energies:
                print("Error: All conformers failed optimization")
                return
            best_conf_id, _ = min(valid_energies, key=lambda x: x[1])
    else:
        success = AllChem.EmbedMolecule(mol, randomSeed=42)
        if success == -1:
            print("Error: Embedding failed")
            return
        AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
        best_conf_id = 0

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(ms_file))[0]
    sdf_file = os.path.join(output_dir, f"{base_name}.sdf")
    txt_file = os.path.join(output_dir, f"{base_name}_smiles.txt")
    
    # Save SDF
    writer = Chem.SDWriter(sdf_file)
    writer.write(mol, confId=int(best_conf_id))
    writer.close()
    
    # Save SMILES to .txt
    with open(txt_file, 'w') as f:
        f.write(smiles)
    
    print(f"Generated '{sdf_file}' and saved SMILES to '{txt_file}'")

# User inputs
ms_file = input("Enter MS data file path (CSV or image, e.g., C:/Users/is/taxol_ms.csv): ")
ir_file = input("Enter FTIR data file path (CSV or image, optional, press Enter to skip): ")
output_dir = input("Enter output directory (e.g., F:/paris): ")

# Execute
generate_3d_from_ms_ir(ms_file, ir_file, output_dir)
