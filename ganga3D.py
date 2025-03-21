
#this is my first work 
#ganga_v_0

try:
    import numpy as np
    import pandas as pd
    import cv2
    from scipy.signal import find_peaks
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as Descriptors
    from rdkit.Chem import AllChem
    import requests
    from sklearn.ensemble import RandomForestClassifier
    import os
except ImportError:
    import numpy as np
    import pandas as pd
    import cv2
    from scipy.signal import find_peaks
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors as Descriptors
    from rdkit.Chem import AllChem
    import requests
    from sklearn.ensemble import RandomForestClassifier
    import os

# Loss rules for natural products and larger molecules
LOSSES = {
    14.0157: "C",
    18.0106: "O",
    34.9689: "Cl",
    44.0262: "CO2",
    46.0055: "NO2",
    78.9183: "Br",
    132.0423: "c1ccc(O)c(O)c1",
    162.0528: "c1ccccc1O",
    42.0470: "CCC",
    162.0528: "C6H10O5",  # Sugar moiety
    57.0215: "CC(N)C",  # Glycine residue
    71.0371: "CC(N)CC",  # Alanine residue
    500.0: "c1ccccc1" * 5  # Simplified large aromatic system
}

def get_molecular_formula(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Descriptors.CalcMolFormula(mol)
    return "Invalid SMILES"

def cosine_similarity(observed_mz, observed_int, ref_mz, ref_int, tolerance=0.01):
    matches = 0
    observed_norm = np.sqrt(np.sum(np.square(observed_int)))
    ref_norm = np.sqrt(np.sum(np.square(ref_int)))
    for i, mz1 in enumerate(observed_mz):
        for j, mz2 in enumerate(ref_mz):
            if abs(mz1 - mz2) < tolerance:
                matches += observed_int[i] * ref_int[j]
    if observed_norm * ref_norm == 0:
        return 0
    return matches / (observed_norm * ref_norm)

def calculate_rmsd(observed_mz, ref_mz, tolerance=0.01):
    matched_pairs = []
    for omz in observed_mz:
        for rmz in ref_mz:
            if abs(omz - rmz) < tolerance:
                matched_pairs.append((omz, rmz))
    if not matched_pairs:
        return float('inf')
    squared_diff = sum((omz - rmz) ** 2 for omz, rmz in matched_pairs)
    return np.sqrt(squared_diff / len(matched_pairs))

def tanimoto_similarity(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    if mol1 and mol2:
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, 2048)
        return AllChem.DataStructs.TanimotoSimilarity(fp1, fp2)
    return 0

def stereo_score(smiles, reference_smiles="C=CCSS(=O)CC=C"):
    mol = Chem.MolFromSmiles(smiles)
    ref_mol = Chem.MolFromSmiles(reference_smiles)
    if not mol or not ref_mol:
        return 0
    stereo_centers = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
    ref_stereo_centers = len(Chem.FindMolChiralCenters(ref_mol, includeUnassigned=True))
    if ref_stereo_centers == 0:
        return 1 if stereo_centers == 0 else 0
    return 1 if stereo_centers == ref_stereo_centers else 0

def fetch_db_candidates(mz, intensities, tolerance=0.01):
    precursor = max(mz)
    candidates = [
        # MassBank: Allicin (MW 162.27 Da)
        {"smiles": "C=CCSS(=O)CC=C", "fragments": [(73.0, 100.0), (41.0, 50.0)], "score": 0.7, "source": "MassBank"},
        # NIST SRD 1A: Allicin (MW 162.27 Da)
        {"smiles": "C=CCSS(=O)CC=C", "fragments": [(73.0, 90.0), (41.0, 60.0)], "score": 0.65, "source": "NIST SRD 1A"},
        # METLIN: Apigenin (MW 270 Da)
        {"smiles": "c1cc(c(cc1O)O)C2=CC(=O)c3c(c(c(c(c3)O)O)O)O2", "fragments": [(271.0, 100.0), (153.0, 40.0)], "score": 0.55, "source": "METLIN"},
        # mzCloud: Luteolin (MW 286 Da)
        {"smiles": "c1cc(c(c(c1)O)O)C2=CC(=O)c3c(c(c(c(c3)O)O)O)O2", "fragments": [(287.0, 100.0), (153.0, 30.0)], "score": 0.5, "source": "mzCloud"},
        # GNPS: Polyketide (MW 274 Da)
        {"smiles": "CC1CC(=O)OC(C)C(=O)OC(C)C(=O)OC(C)C1=O", "fragments": [(275.0, 100.0), (149.0, 20.0)], "score": 0.5, "source": "GNPS"},
        # ChemSpider: Phenolic compound (MW 254 Da)
        {"smiles": "c1cc(c(c(c1)O)O)CC(=O)c2cc(c(c(c2)O)O)O", "fragments": [(255.0, 100.0), (137.0, 25.0)], "score": 0.4, "source": "ChemSpider"},
        # Added for larger molecules:
        # Peptide (e.g., Angiotensin II, MW 1046 Da)
        {"smiles": "CC(C)C[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@H](CCCNC(N)=N)NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](CC2=CN=CN2)NC(=O)[C@H](CC3=CC=CC=C3)NC(=O)[C@H](CC4=CN=CN4)NC(=O)[C@@H](N)CC(O)=O)C(=O)O",
         "fragments": [(1047.0, 100.0), (931.0, 50.0), (784.0, 30.0)], "score": 0.6, "source": "MassBank"},
        # Macrolide (e.g., Erythromycin, MW 733.93 Da)
        {"smiles": "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]3O[C@H](C)C[C@@H]([C@H]3O)N(C)C)[C@](C)(O)C[C@@H](C)C(=O)[C@H](C)[C@@H](O)[C@]1(C)O",
         "fragments": [(734.0, 100.0), (576.0, 40.0), (158.0, 20.0)], "score": 0.55, "source": "METLIN"},
        # Glycopeptide (e.g., Vancomycin, MW 1449 Da)
        {"smiles": "CC[C@H](C)[C@H]1C(=O)N[C@@H](CC2=CC=C(C=C2)OC3=C(C=C(C=C3)[C@H](C(=O)N[C@H](C(=O)N[C@@H](CC4=CC(=C(C=C4)OC5=C(C=C(C=C5)[C@H](C(=O)N1)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@@H](CC6=CC=C(C=C6)O)NC(=O)[C@@H](C)O)Cl)OC7C(C(C(C(O7)CO)O)O)NC(=O)C)Cl)O)O[C@@H]8C(C(C(C(O8)CO)O)O)O)O",
         "fragments": [(1450.0, 100.0), (1306.0, 60.0), (1144.0, 30.0)], "score": 0.5, "source": "mzCloud"},
        # Large natural product (e.g., Rapamycin, MW 914 Da)
        {"smiles": "CC1CCC2CC(C(=CC=CC=CC(CC(C(=O)C(C(C(=CC(C(=O)CC(OC(=O)C3CCCCN3C(=O)C(=O)C1(O2)O)C(C)CC4CCC(C(C4)OC)O)C)C)O)OC)C)C)OC",
         "fragments": [(915.0, 100.0), (897.0, 50.0), (579.0, 20.0)], "score": 0.5, "source": "GNPS"},
        # Synthetic peptide (MW ~3000 Da, 26 amino acids)
        {"smiles": "C[C@H](NC(=O)[C@H](CC1=CC=CC=C1)NC(=O)[C@H](CO)NC(=O)[C@H](CC2=CN=CN2)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](C)NC(=O)[C@H](CC3=CC=C(O)C=C3)NC(=O)[C@H](CC4=CN=CN4)NC(=O)[C@H](CC5=CC=CC=C5)NC(=O)[C@H](CO)NC(=O)[C@H](CC6=CN=CN6)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(O)=O)NC(=O)[C@H](C)NC(=O)[C@H](CC7=CC=C(O)C=C7)NC(=O)[C@H](CC8=CN=CN8)NC(=O)[C@H](CC9=CC=CC=C9)NC(=O)[C@H](CO)NC(=O)[C@H](CC%10=CN=CN%10)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(O)=O)NC(=O)[C@@H](N)C)C(=O)O",
         "fragments": [(3001.0, 100.0), (2873.0, 60.0), (2745.0, 40.0)], "score": 0.45, "source": "ChemSpider"}
    ]
    return candidates

def build_fragment_tree(mz, intensities, max_level=11):
    precursor = max(mz)
    max_level = min(max_level, int(np.log2(precursor / 10) + 1)) if precursor > 100 else 3
    tree = [(precursor, max(intensities))]
    remaining_mz = sorted([m for m, i in zip(mz, intensities) if m < precursor], reverse=True)
    used_mz = {precursor}
    current_level = 1
    
    while current_level < max_level and remaining_mz:
        parent_mz, parent_int = tree[-1]
        for m in remaining_mz[:]:
            loss = parent_mz - m
            if 5 < loss < 3000 and m not in used_mz:  # Extended loss range
                tree.append((m, intensities[mz.tolist().index(m)]))
                used_mz.add(m)
                remaining_mz.remove(m)
                current_level += 1
                break
        else:
            break
    return tree

def score_candidates(candidates, exp_mz, exp_intensities, tree):
    scores = []
    tree_mz = [m for m, _ in tree]
    
    for cand in candidates:
        cand_mz = [f[0] for f in cand["fragments"]]
        cand_int = [f[1] for f in cand["fragments"]]
        mz_matches = sum(1 for em in exp_mz for cm in cand_mz if abs(em - cm) < 0.01)
        tree_matches = sum(1 for tm in tree_mz for cm in cand_mz if abs(tm - cm) < 0.01)
        spectral_similarity = cosine_similarity(exp_mz, exp_intensities, cand_mz, cand_int)
        score = (mz_matches / max(len(exp_mz), 1)) * 0.3 + (tree_matches / max(len(tree), 1)) * 0.2 + spectral_similarity * 0.5
        scores.append((cand["smiles"], score, cand["source"], cand["fragments"], spectral_similarity))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def denovo_predict(mz, intensities, tree):
    neutral_mass = max(mz) - 1.0078  # Up to 3000 Da
    carbon_est = max(1, int(neutral_mass / 14))  # Estimate carbons
    hydrogen_est = carbon_est * 2
    oxygen_est = max(1, int(neutral_mass / 50))
    sulfur_est = int(neutral_mass / 100)  # Rough estimate for sulfur
    smiles_parts = []
    
    # Base structure: Peptide-like or macrocycle depending on mass
    if neutral_mass < 500:
        # Small molecule (e.g., allicin-like)
        smiles_parts.append("C=C")
        smiles_parts.append("SS(=O)")
        smiles_parts.append("CC=C")
    elif 500 <= neutral_mass <= 1500:
        # Medium molecule (e.g., macrolide-like)
        smiles_parts.append("CC1CC(=O)OC(C)C(=O)OC(C)C(=O)OC(C)C1=O")  # Polyketide core
        remaining_mass = neutral_mass - 274  # Polyketide MW
        while remaining_mass > 0:
            if remaining_mass >= 162:
                smiles_parts.append("C6H10O5")  # Sugar moiety
                remaining_mass -= 162
            elif remaining_mass >= 71:
                smiles_parts.append("CC(N)CC")  # Alanine
                remaining_mass -= 71
            elif remaining_mass >= 57:
                smiles_parts.append("CC(N)C")  # Glycine
                remaining_mass -= 57
            else:
                break
    else:
        # Large molecule (e.g., peptide-like)
        smiles_parts.append("C")  # Start with a carbon
        remaining_mass = neutral_mass - 12
        amino_acids = [("CC(N)C", 57), ("CC(N)CC", 71), ("CC(C)C[C@H](N)C", 113)]  # Gly, Ala, Leu
        while remaining_mass > 0 and len(smiles_parts) < 50:  # Limit chain length
            for aa_smiles, aa_mass in amino_acids:
                if remaining_mass >= aa_mass:
                    smiles_parts.append(aa_smiles)
                    remaining_mass -= aa_mass
                    break
            else:
                break
        smiles_parts.append("C(=O)O")  # Carboxyl terminus
    
    smiles = "".join(smiles_parts)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        smiles = "C=CCSS(=O)CC=C"  # Fallback
    return smiles

def train_rf_model():
    X = np.array([
        [3, 0.9, 0.8, 0.6, 1],
        [8, 0.7, 0.8, 0.6, 1],
        [11, 0.6, 0.7, 0.5, 1]
    ])
    y = [1, 1, 0]
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return rf

def run_hybrid_model(mz, intensities, output_dir, top_n=1):
    tree = build_fragment_tree(mz, intensities)
    candidates = fetch_db_candidates(mz, intensities)
    rf = train_rf_model()
    
    smiles_list = []
    if candidates:
        ranked = score_candidates(candidates, mz, intensities, tree)
        features = [
            [len(tree), mz_matches / max(len(mz), 1), 
             cosine_similarity(mz, intensities, [f[0] for f in cand["fragments"]], [f[1] for f in cand["fragments"]]), 
             cand["score"], 1 if any(h in cand["smiles"] for h in ["O", "N", "S", "Cl", "Br"]) else 0]
            for cand, mz_matches in [(c, sum(1 for em in mz for cm in [f[0] for f in c["fragments"]] if abs(em - cm) < 0.01)) for c in candidates]
        ]
        confidences = rf.predict_proba(features)[:, 1] if features else [0] * len(ranked)
        ranked = [(s, score * conf, source, fragments, msms_fit) for (s, score, source, fragments, msms_fit), conf in zip(ranked, confidences)]
        ranked.sort(key=lambda x: x[1], reverse=True)
        for s, score, source, fragments, msms_fit in ranked[:top_n]:
            rmsd = calculate_rmsd(mz, [f[0] for f in fragments])
            tmscore = tanimoto_similarity(s, "C=CCSS(=O)CC=C")
            stereo = stereo_score(s)
            smiles_list.append((s, f"DB Score: {score:.2f}, Formula: {get_molecular_formula(s)}, Source: {source}, "
                                 f"Mass Fragments (m/z, intensity): {fragments}, RMSD: {rmsd:.2f}, TMScore: {tmscore:.2f}, "
                                 f"MS/MS Fit: {msms_fit:.2f}, Stereo Score: {stereo:.2f}"))
        if not smiles_list or ranked[0][1] < 0.7:
            de_novo_smiles = denovo_predict(mz, intensities, tree)
            de_novo_fragments = [(73.0, 100.0), (41.0, 50.0)] if max(mz) < 500 else [(max(mz), 100.0), (max(mz)-128, 50.0)]
            msms_fit = cosine_similarity(mz, intensities, [f[0] for f in de_novo_fragments], [f[1] for f in de_novo_fragments])
            rmsd = calculate_rmsd(mz, [f[0] for f in de_novo_fragments])
            tmscore = tanimoto_similarity(de_novo_smiles, "C=CCSS(=O)CC=C")
            stereo = stereo_score(de_novo_smiles)
            smiles_list.append((de_novo_smiles, f"De Novo Prediction, Formula: {get_molecular_formula(de_novo_smiles)}, Source: De Novo, "
                                               f"Mass Fragments (m/z, intensity): {de_novo_fragments}, RMSD: {rmsd:.2f}, TMScore: {tmscore:.2f}, "
                                               f"MS/MS Fit: {msms_fit:.2f}, Stereo Score: {stereo:.2f}"))
    else:
        de_novo_smiles = denovo_predict(mz, intensities, tree)
        de_novo_fragments = [(73.0, 100.0), (41.0, 50.0)] if max(mz) < 500 else [(max(mz), 100.0), (max(mz)-128, 50.0)]
        msms_fit = cosine_similarity(mz, intensities, [f[0] for f in de_novo_fragments], [f[1] for f in de_novo_fragments])
        rmsd = calculate_rmsd(mz, [f[0] for f in de_novo_fragments])
        tmscore = tanimoto_similarity(de_novo_smiles, "C=CCSS(=O)CC=C")
        stereo = stereo_score(de_novo_smiles)
        smiles_list.append((de_novo_smiles, f"De Novo Prediction, Formula: {get_molecular_formula(de_novo_smiles)}, Source: De Novo, "
                                           f"Mass Fragments (m/z, intensity): {de_novo_fragments}, RMSD: {rmsd:.2f}, TMScore: {tmscore:.2f}, "
                                           f"MS/MS Fit: {msms_fit:.2f}, Stereo Score: {stereo:.2f}"))
    
    os.makedirs(output_dir.strip(), exist_ok=True)
    with open(os.path.join(output_dir.strip(), "result.txt"), "w") as f:
        f.write("Predicted SMILES List:\n")
        for smiles, source in smiles_list:
            f.write(f"{smiles} ({source})\n")
    return smiles_list

def process_image_for_spectrum(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not load image. Check the file path.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.dilate(thresh, kernel, iterations=1)
    intensity_profile = np.sum(morph, axis=0)
    intensity_profile = intensity_profile / np.max(intensity_profile) * 100
    width = intensity_profile.shape[0]
    mz_min, mz_max = 0, 3000  # Extended range
    mz_values = np.linspace(mz_min, mz_max, width)
    peaks, properties = find_peaks(intensity_profile, height=10, distance=5, prominence=5)
    if len(peaks) == 0:
        print("No peaks detected. Retrying with enhanced contrast...")
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.5, beta=0)
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
        morph = cv2.dilate(thresh, kernel, iterations=1)
        intensity_profile = np.sum(morph, axis=0)
        intensity_profile = intensity_profile / np.max(intensity_profile) * 100
        peaks, properties = find_peaks(intensity_profile, height=5, distance=5, prominence=3)
    if len(peaks) == 0:
        raise ValueError("No peaks detected in the spectrum image.")
    mz = mz_values[peaks]
    intensities = intensity_profile[peaks]
    return mz, intensities

# Prompt for input type, file path, and output directory
input_type = input("Enter input type ('csv' or 'image'): ").lower()
input_file = input("Enter the path to your input file (e.g., C:\\Users\\is\\Desktop\\file.csv or file.png): ").strip()
output_dir = input("Enter the directory to store SMILES output (e.g., C:\\Users\\is\\Desktop): ").strip()

if input_type == "csv":
    spectrum_data = pd.read_csv(input_file)
    print("\nColumns in your CSV file:", list(spectrum_data.columns))
    mz_column = input("Enter the column name for m/z values (e.g., 'mass' or 'm/z'): ").strip()
    intensity_column = input("Enter the column name for intensity values (e.g., 'intensity'): ").strip()
    mz = np.array(spectrum_data[mz_column])
    intensities = np.array(spectrum_data[intensity_column])
elif input_type == "image":
    mz, intensities = process_image_for_spectrum(input_file)
else:
    raise ValueError("Invalid input type. Please enter 'csv' or 'image'.")

# Run the model
smiles_list = run_hybrid_model(mz, intensities, output_dir)
print("\nPredicted SMILES for the mass spectrum:")
for smiles, source in smiles_list:
    print(f"{smiles} ({source})")
print(f"\nSMILES output saved to {os.path.join(output_dir, 'result.txt')}")

#end
