import sys
import os
import pandas as pd
import numpy as np
import cv2
import requests
import time
from PIL import Image
from scipy.signal import find_peaks, savgol_filter
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors as Descriptors
from requests.exceptions import RequestException

# API keys (optional)
API_KEYS = {
    "pubchem": None,
    "zinc": None,
    "chembl": None,
    "massbank": None,
    "hmdb": None
}

def preprocess_image(image_path):
    """Preprocess MS/NMR image: denoise, enhance contrast, normalize."""
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Denoising with Gaussian blur
    denoised = cv2.GaussianBlur(img_array, (5, 5), 0)

    # Contrast enhancement with CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised.astype(np.uint8))

    # Normalization
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    return normalized

def extract_peaks(image_data, is_ms=False, is_nmr=False, is_13c_nmr=False):
    """Extract peaks from preprocessed image data."""
    # Compute intensity profile (mean along vertical axis)
    intensity_profile = np.mean(image_data, axis=0)

    # Smooth with Savitzky-Golay filter
    smoothed = savgol_filter(intensity_profile, window_length=11, polyorder=2)

    # Peak detection
    height_threshold = np.max(smoothed) * (0.1 if is_ms else 0.05)
    peaks, properties = find_peaks(smoothed, height=height_threshold, distance=5, prominence=0.05 * np.max(smoothed))

    # Scale x-axis based on data type
    x_values = np.arange(len(smoothed), dtype=float)
    if is_ms:
        x_scaled = x_values  # Assume m/z range is image width (calibration needed for real data)
    elif is_nmr:
        x_scaled = (x_values / len(x_values)) * 12  # 0-12 ppm for 1H NMR
    elif is_13c_nmr:
        x_scaled = (x_values / len(x_values)) * 200  # 0-200 ppm for 13C NMR
    else:
        x_scaled = x_values

    return x_scaled[peaks], smoothed[peaks]

def process_file(file_path, is_ms=False, is_nmr=False, is_13c_nmr=False):
    """Process file (CSV or image) with preprocessing for images."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_path}' not found.")
    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        data = pd.read_csv(file_path, header=None)
        x = pd.to_numeric(data[0], errors='coerce')
        y = pd.to_numeric(data[1], errors='coerce')
        valid_mask = ~np.isnan(x) & ~np.isnan(y)
        if not valid_mask.any():
            raise ValueError(f"No valid data in '{file_path}'")
        x_clean = x[valid_mask].to_numpy()
        y_clean = y[valid_mask].to_numpy()
        if is_ms:
            sorted_indices = np.argsort(x_clean)
            return x_clean[sorted_indices].astype(float), y_clean[sorted_indices]
        if is_nmr or is_13c_nmr:
            peaks, _ = find_peaks(y_clean, height=np.max(y_clean) * 0.05, distance=5)
            return x_clean[peaks].astype(float), y_clean[peaks]
        return x_clean.astype(float), y_clean
    else:
        img_data = preprocess_image(file_path)
        return extract_peaks(img_data, is_ms, is_nmr, is_13c_nmr)

def predict_fragments(smiles, mz, intensities):
    """Predict MS fragments with natural product-specific rules."""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [(m, i) for m, i in zip(mz, intensities)]

    fragments = []
    mol_wt = Descriptors.CalcExactMolWt(mol)
    for bond in mol.GetBonds():
        begin_atom = bond.GetBeginAtom().GetAtomicNum()
        end_atom = bond.GetEndAtom().GetAtomicNum()
        if begin_atom in [7, 8] or end_atom in [7, 8]:
            fragments.append((mol_wt / 2, max(intensities) * 0.5))
        if bond.IsInRing() and begin_atom == 6 and end_atom == 6:
            fragments.append((mol_wt - 68, max(intensities) * 0.3))
    if "O" in smiles and "C" in smiles:
        fragments.append((mol_wt - 162, max(intensities) * 0.4))
    if not fragments:
        fragments = [(m, i) for m, i in zip(mz, intensities)]
    return sorted(fragments, key=lambda x: x[1], reverse=True)[:5]

def api_request(url, method="get", params=None, headers=None, json=None, retries=3, delay=2):
    """Generic API request handler with retries and rate limiting."""
    for attempt in range(retries):
        try:
            if method == "get":
                response = requests.get(url, params=params, headers=headers, timeout=30)
            elif method == "post":
                response = requests.post(url, json=json, headers=headers, timeout=30)
            response.raise_for_status()
            time.sleep(delay)
            return response.json()
        except RequestException as e:
            if attempt < retries - 1:
                print(f"API request failed: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print(f"API request failed after {retries} attempts: {e}")
                return None

def pubchem_library_match(mz, intensities):
    url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/mass/json"
    max_mz = mz[np.argmax(intensities)]
    params = {"mass": f"{max_mz-1},{max_mz+1}", "max_records": 1}
    headers = {"API-Key": API_KEYS["pubchem"]} if API_KEYS["pubchem"] else None
    result = api_request(url, params=params, headers=headers)
    if result and "Compounds" in result and result["Compounds"]:
        cid = result["Compounds"][0]["CID"]
        smiles_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/CanonicalSMILES/json"
        smiles_result = api_request(smiles_url, headers=headers)
        if smiles_result and "PropertyTable" in smiles_result and "Properties" in smiles_result["PropertyTable"]:
            return smiles_result["PropertyTable"]["Properties"][0]["CanonicalSMILES"], []
    print("No PubChem match found.")
    return None, []

def zinc_library_match(mz, intensities):
    url = "https://zinc20.docking.org/substances.json"
    max_mz = mz[np.argmax(intensities)]
    params = {"mw_range": f"{max_mz-1}-{max_mz+1}", "count": 1}
    headers = {"API-Key": API_KEYS["zinc"]} if API_KEYS["zinc"] else None
    result = api_request(url, params=params, headers=headers)
    if result and isinstance(result, list) and "smiles" in result[0]:
        return result[0]["smiles"], []
    print("No ZINC match found.")
    return None, []

def chembl_library_match(mz, intensities):
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
    max_mz = mz[np.argmax(intensities)]
    params = {"molecule_properties__mw_freebase__gte": max_mz-1, "molecule_properties__mw_freebase__lte": max_mz+1, "limit": 1}
    headers = {"API-Key": API_KEYS["chembl"]} if API_KEYS["chembl"] else None
    result = api_request(url, params=params, headers=headers)
    if result and "molecules" in result and result["molecules"]:
        return result["molecules"][0]["molecule_structures"]["canonical_smiles"], []
    print("No ChEMBL match found.")
    return None, []

def massbank_library_match(mz, intensities):
    url = "https://massbank.eu/rest/search"
    peaks = [{"mz": float(m), "intensity": float(i)} for m, i in zip(mz, intensities)]
    payload = {"peaks": peaks, "tolerance": 0.1}
    headers = {"Content-Type": "application/json"}
    if API_KEYS["massbank"]:
        headers["API-Key"] = API_KEYS["massbank"]
    result = api_request(url, method="post", json=payload, headers=headers)
    if result and "hits" in result and result["hits"]:
        top_hit = result["hits"][0]
        if "smiles" in top_hit:
            fragments = [(float(f["mz"]), float(f["intensity"])) for f in top_hit.get("fragments", [])]
            return top_hit["smiles"], fragments
    print("No MassBank match found.")
    return None, []

def hmdb_library_match(mz, intensities):
    url = "https://hmdb.ca/api/metabolites"
    max_mz = mz[np.argmax(intensities)]
    params = {"mass": f"{max_mz-1}:{max_mz+1}", "output": "json"}
    headers = {"API-Key": API_KEYS["hmdb"]} if API_KEYS["hmdb"] else None
    result = api_request(url, params=params, headers=headers)
    if result and "metabolites" in result and result["metabolites"]:
        return result["metabolites"][0]["smiles"], []
    print("No HMDB match found.")
    return None, []

def library_match(mz, intensities):
    """Try multiple libraries in sequence."""
    for func in [pubchem_library_match, zinc_library_match, chembl_library_match, massbank_library_match, hmdb_library_match]:
        smiles, fragments = func(mz, intensities)
        if smiles:
            return smiles, fragments
    return None, []

def sirius_like_2d_inference(mz, intensities, nmr_shifts=None, nmr_intensities=None, c13_shifts=None):
    """SIRIUS-like 2D structure inference using MS and NMR data."""
    # Step 1: Library match
    smiles, lib_fragments = library_match(mz, intensities)
    if smiles:
        try:
            mol = Chem.MolFromSmiles(smiles)
            Chem.SanitizeMol(mol)
            mol_formula = Descriptors.CalcMolFormula(mol)
            return smiles, lib_fragments if lib_fragments else predict_fragments(smiles, mz, intensities), mol_formula, mol
        except Exception as e:
            print(f"Library SMILES error: {e}")

    # Step 2: Molecular formula prediction (SIRIUS-like)
    max_mz = mz[np.argmax(intensities)]
    carbon_est = int(max_mz / 12)  # Approximate carbon count (SIRIUS uses isotopic patterns, simplified here)
    hydrogen_est = carbon_est * 2  # Rough H/C ratio
    oxygen_est, nitrogen_est = 0, 0

    if nmr_shifts is not None and nmr_intensities is not None:
        hetero_peaks = sum(1 for s in nmr_shifts if s > 8)
        nitrogen_est = hetero_peaks if any(9 <= s <= 12 for s in nmr_shifts) else 0
        oxygen_est = hetero_peaks - nitrogen_est
    if c13_shifts is not None:
        oxygen_est += sum(1 for s in c13_shifts if 160 <= s <= 200)  # Carbonyl groups

    # Adjust for mass
    mol_mass_est = carbon_est * 12 + hydrogen_est + oxygen_est * 16 + nitrogen_est * 14
    if abs(mol_mass_est - max_mz) > 2:
        hydrogen_est += int((max_mz - mol_mass_est) / 1.0078)  # Fine-tune with hydrogen

    mol_formula = f"C{carbon_est}H{hydrogen_est}"
    if oxygen_est:
        mol_formula += f"O{oxygen_est}"
    if nitrogen_est:
        mol_formula += f"N{nitrogen_est}"

    # Step 3: Structure generation with NMR constraints
    smiles = "C" * carbon_est
    if oxygen_est:
        smiles += "O" * oxygen_est
    if nitrogen_est:
        smiles += "N" * nitrogen_est

    if nmr_shifts:
        aromatic_peaks = sum(1 for s in nmr_shifts if 6 <= s <= 8)
        if aromatic_peaks >= 4:
            smiles = "c1ccccc1" + "C" * (carbon_est - 6) + "O" * oxygen_est + "N" * nitrogen_est
        elif aromatic_peaks >= 2:
            smiles = "c1ccc(c(c1))" + "C" * (carbon_est - 5) + "O" * oxygen_est + "N" * nitrogen_est

    if c13_shifts and oxygen_est:
        carbonyls = sum(1 for s in c13_shifts if 160 <= s <= 200)
        if carbonyls:
            smiles += "[C=O]" * min(carbonyls, oxygen_est)

    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError("Invalid SMILES generated")
        Chem.SanitizeMol(mol)
        mol_formula = Descriptors.CalcMolFormula(mol)
        return smiles, predict_fragments(smiles, mz, intensities), mol_formula, mol
    except Exception as e:
        print(f"SIRIUS-like inference error: {e}")
        smiles = "c1ccccc1"  # Fallback
        mol = Chem.MolFromSmiles(smiles)
        mol_formula = Descriptors.CalcMolFormula(mol)
        return smiles, predict_fragments(smiles, mz, intensities), mol_formula, mol

def generate_3d_structure(mol, nmr_shifts=None, c13_shifts=None):
    """Generate and optimize 3D structure from 2D molecule."""
    mol = Chem.AddHs(mol)
    try:
        if nmr_shifts or c13_shifts:
            AllChem.EmbedMultipleConfs(mol, numConfs=100, randomSeed=42, pruneRmsThresh=0.5)
            energies = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)
            valid_energies = [(i, e[1]) for i, e in enumerate(energies) if e[0] == 0 and e[1] != -1]
            if not valid_energies:
                raise RuntimeError("All conformers failed optimization")
            best_conf_id, _ = min(valid_energies, key=lambda x: x[1])
        else:
            success = AllChem.EmbedMolecule(mol, randomSeed=42, maxAttempts=10)
            if success == -1:
                raise RuntimeError("Embedding failed")
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
            best_conf_id = 0
        return mol, best_conf_id
    except Exception as e:
        print(f"3D generation error: {e}")
        return None, None

def process_ms_to_3d(ms_file, nmr_file=None, c13_nmr_file=None, output_dir="output"):
    """Pipeline: Process MS/NMR, generate 2D with SIRIUS-like inference, then 3D."""
    try:
        mz, intensities = process_file(ms_file, is_ms=True)
    except Exception as e:
        print(f"MS processing error: {e}")
        return

    nmr_shifts, nmr_intensities = None, None
    c13_shifts = None
    if nmr_file:
        try:
            nmr_shifts, nmr_intensities = process_file(nmr_file, is_nmr=True)
        except Exception as e:
            print(f"1H NMR processing error: {e}")
    if c13_nmr_file:
        try:
            c13_shifts, _ = process_file(c13_nmr_file, is_13c_nmr=True)
        except Exception as e:
            print(f"13C NMR processing error: {e}")

    # Step 1: Generate 2D structure with SIRIUS-like inference
    smiles, fragments, mol_formula, mol_2d = sirius_like_2d_inference(mz, intensities, nmr_shifts, nmr_intensities, c13_shifts)
    if not mol_2d:
        print("Error: Failed to generate 2D structure.")
        return

    # Step 2: Generate and optimize 3D structure
    mol_3d, best_conf_id = generate_3d_structure(mol_2d, nmr_shifts, c13_shifts)
    if not mol_3d:
        print("Error: Failed to generate 3D structure.")
        return

    # Save results
    try:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(ms_file))[0]
        sdf_file = os.path.join(output_dir, f"{base_name}.sdf")
        txt_file = os.path.join(output_dir, f"{base_name}_smiles.txt")

        writer = Chem.SDWriter(sdf_file)
        writer.write(mol_3d, confId=int(best_conf_id))
        writer.close()

        with open(txt_file, 'w') as f:
            f.write(f"SMILES: {smiles}\n")
            f.write(f"Molecular Formula: {mol_formula}\n")
            f.write("Mass Fragments (m/z, intensity):\n")
            for frag_mz, frag_intensity in fragments:
                f.write(f"{frag_mz:.2f}, {frag_intensity:.2f}\n")
        print(f"Generated '{sdf_file}' and saved to '{txt_file}'")
    except Exception as e:
        print(f"Output saving error: {e}")

# User inputs
ms_file = input("Enter MS data file path (CSV or image, e.g., test 1 2025-03-21 120305.png): ")
nmr_file = input("Enter 1H NMR data file path (CSV or image, optional, press Enter to skip): ") or None
c13_nmr_file = input("Enter 13C NMR data file path (CSV or image, optional, press Enter to skip): ") or None
output_dir = input("Enter output directory (e.g., F:/paris): ")

# Execute
process_ms_to_3d(ms_file, nmr_file, c13_nmr_file, output_dir)
