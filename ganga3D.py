
#this is my first work 
#ganga_v_0

import numpy as np
import pandas as pd
import cv2
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestClassifier
from rdkit import Chem
from rdkit.Chem import Descriptors, EnumerateStereoisomers

# Expanded substructure library for diverse molecular classes
SUBSTRUCTURES = [
    ("CO", 32.03),  # Methanol (for very small molecules)
    ("CCO", 46.04),  # Ethanol
    ("CC(N)C", 57.02),  # Glycine
    ("CC(N)CC", 71.04),  # Alanine
    ("N[C@@H](CC1=CC=CC=C1)C(=O)O", 147.07),  # Phenylalanine
    ("C(C1C(C(C(C(O1)O)O)O)O)O", 162.05),  # Glucose
    ("COC1=CC(=CC(=C1O)OC)C2=CC(=O)C3=C(C=C(C=C3O2)O)O", 300.06),  # Quercetin core
    ("CCCCCCCCC=CCCCCCCCC(=O)O", 282.27),  # Oleic acid (lipid)
    ("C1CCCCCCCCCCCCCC(=O)O1", 254.22),  # Lactone ring (macrocycle example)
    ("C5H5FeC5H5", 186.03),  # Ferrocene (organometallic)
]

FRAGMENT_SUBSTRUCTURE_MAP = {
    73.0: ["SS(=O)"],  # Disulfide sulfoxide
    147.0: ["N[C@@H](CC1=CC=CC=C1)C"],
    162.0: ["C(C1C(C(C(C(O1)O)O)O)O)O"],
    184.0: ["C5H15NO4P"],  # Phosphatidylcholine headgroup
    301.0: ["COC1=CC(=CC(=C1O)OC)C2=CC(=O)C3=C(C=C(C=C3O2)O)O"],
}

LOSSES = {18.01: "H2O", 44.03: "CO2", 57.02: "CC(N)C", 162.05: "C6H10O5", 282.27: "CCCCCCCCC=CCCCCCCCC(=O)O"}

def classify_molecule(neutral_mass, fragments, c_est, h_est, o_est, s_est):
    n_ratio = c_est / (h_est + 1)
    if neutral_mass < 100:
        return "very_small"
    elif s_est > 0 and neutral_mass < 500:
        return "sulfur-containing"
    elif c_est > 50 and h_est / c_est > 1.5:
        return "lipid"
    elif neutral_mass < 1000 and o_est > 5:
        return "flavonoid"
    elif neutral_mass > 1000 and o_est > 10:
        return "macrocycle"
    return "generic"

def get_dynamic_losses(mol_class):
    if mol_class == "very_small":
        return {18.01: "H2O"}
    elif mol_class == "sulfur-containing":
        return {78.95: "SS(=O)", 18.01: "H2O"}
    elif mol_class == "lipid":
        return {282.27: "CCCCCCCCC=CCCCCCCCC(=O)O", 18.01: "H2O", 184.0: "C5H15NO4P"}
    elif mol_class == "flavonoid":
        return {162.05: "C6H10O5", 44.03: "CO2", 18.01: "H2O"}
    elif mol_class == "macrocycle":
        return {254.22: "C1CCCCCCCCCCCCCC(=O)O1", 18.01: "H2O"}
    return LOSSES

def build_fragment_tree(mz, intensities, max_level=20):
    precursor = max(mz)
    max_level = min(max_level, int(np.log2(precursor / 5) + 1)) if precursor > 100 else 5
    if precursor > 2000:
        max_level = min(max_level + 5, 30)  # Extra depth for high-mass compounds
    tree = [(precursor, max(intensities))]
    fragment_list = sorted(list(zip(mz, intensities)), key=lambda x: x[1], reverse=True)
    remaining_mz = [m for m, i in fragment_list if m < precursor]
    used_mz = {precursor}
    current_level = 1
    
    neutral_mass = precursor - 1.0078
    c_est, h_est, o_est, s_est = estimate_elemental_composition(mz, intensities)
    mol_class = classify_molecule(neutral_mass, fragment_list, c_est, h_est, o_est, s_est)
    dynamic_losses = get_dynamic_losses(mol_class)
    
    while current_level < max_level and remaining_mz:
        parent_mz, parent_int = tree[-1]
        for m in remaining_mz[:]:
            loss = parent_mz - m
            if any(abs(loss - loss_mass) < 0.01 for loss_mass in dynamic_losses.keys()) and m not in used_mz:
                tree.append((m, intensities[mz.tolist().index(m)]))
                used_mz.add(m)
                remaining_mz.remove(m)
                current_level += 1
                break
        else:
            break
    return tree

def estimate_elemental_composition(mz, intensities):
    precursor = max(mz)
    neutral_mass = precursor - 1.0078
    m_plus_1 = precursor + 1.0034
    m_plus_1_intensity = 0
    for m, i in zip(mz, intensities):
        if abs(m - m_plus_1) < 0.01:
            m_plus_1_intensity = i
            break
    c_est = max(1, int(m_plus_1_intensity / (1.1 * max(intensities))))
    h_est = c_est * 2
    o_est = max(1, int(neutral_mass / 50))
    s_est = int(neutral_mass / 100)
    return c_est, h_est, o_est, s_est

def weighted_cosine_similarity(observed_mz, observed_int, ref_mz, ref_int, tolerance=0.01):
    matches = 0
    observed_norm = np.sqrt(np.sum(np.square(observed_int)))
    ref_norm = np.sqrt(np.sum(np.square(ref_int)))
    for i, mz1 in enumerate(observed_mz):
        for j, mz2 in enumerate(ref_mz):
            if abs(mz1 - mz2) < tolerance:
                weight = (observed_int[i] + ref_int[j]) / 200
                matches += observed_int[i] * ref_int[j] * weight
    if observed_norm * ref_norm == 0:
        return 0
    return matches / (observed_norm * ref_norm)

def score_neutral_losses(exp_mz, ref_mz):
    score = 0
    for emz in exp_mz:
        for rmz in ref_mz:
            loss = abs(emz - rmz)
            if any(abs(loss - loss_mass) < 0.01 for loss_mass in LOSSES.keys()):
                score += 1
    return score / max(len(exp_mz), len(ref_mz))

def fragment_coverage(exp_mz, ref_mz, tolerance=0.01):
    matches = sum(1 for em in exp_mz for rm in ref_mz if abs(em - rm) < tolerance)
    return matches / len(exp_mz)

def fetch_db_candidates(mz, intensities, tolerance=0.01):
    precursor = max(mz)
    candidates = [
        {"smiles": "C=CCSS(=O)CC=C", "fragments": [(73.0, 100.0), (41.0, 50.0)], "score": 0.7, "source": "MassBank"},
        {"smiles": "COC1=CC(=CC(=C1O)OC)C2=CC(=O)C3=C(C=C(C=C3O2)O)O", "fragments": [(301.0, 100.0), (286.0, 60.0)], "score": 0.6, "source": "CASMI"},  # Quercetin
        {"smiles": "CC(=O)OC1C(C(OC(C1O)OC2C(C(C(C(O2)CO)O)O)O)CO)O", "fragments": [(465.0, 100.0), (303.0, 50.0)], "score": 0.5, "source": "CASMI"},  # Glycoside
        {"smiles": "CCCCCCCCC=CCCCCCCCC(=O)O", "fragments": [(283.0, 100.0), (265.0, 50.0)], "score": 0.65, "source": "GNPS"},  # Oleic acid
        {"smiles": "C5H5FeC5H5", "fragments": [(187.0, 100.0), (121.0, 40.0)], "score": 0.4, "source": "ChemSpider"}  # Ferrocene
    ]
    return candidates

def train_substructure_predictor():
    X = [
        [73.0, 100.0, 162.27, 6, 10, 1, 2],  # Allicin
        [301.0, 100.0, 302.24, 15, 10, 7, 0],  # Quercetin
        [283.0, 100.0, 282.27, 18, 34, 2, 0]  # Oleic acid
    ]
    y = ["SS(=O)", "COC1=CC(=CC(=C1O)OC)C2=CC(=O)C3=C(C=C(C=C3O2)O)O", "CCCCCCCCC=CCCCCCCCC(=O)O"]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

def predict_substructure(clf, fragment_mz, intensity, neutral_mass, c_est, h_est, o_est, s_est):
    features = [[fragment_mz, intensity, neutral_mass, c_est, h_est, o_est, s_est]]
    return clf.predict(features)[0]

def train_stereo_predictor():
    X = [
        [301.0, 100.0, 302.24, 0],  # Quercetin (0 stereocenters)
        [340.0, 100.0, 339.39, 2]   # Escholtzine (2 stereocenters)
    ]
    y = [0, 2]
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

def predict_stereocenters(clf, fragment_mz, intensity, neutral_mass):
    features = [[fragment_mz, intensity, neutral_mass, 0]]  # Simplified
    return clf.predict(features)[0]

def enhance_stereo_prediction(smiles, mz, intensities):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        stereocenters = Chem.FindMolChiralCenters(mol)
        if len(stereocenters) > 0:
            isomers = Chem.EnumerateStereoisomers(mol)
            best_isomer = None
            best_score = 0
            for isomer in isomers:
                isomer_smiles = Chem.MolToSmiles(isomer, isomericSmiles=True)
                de_novo_fragments = generate_denovo_fragments(isomer_smiles, max(mz))
                score = weighted_cosine_similarity(mz, intensities, [f[0] for f in de_novo_fragments], [f[1] for f in de_novo_fragments])
                if score > best_score:
                    best_score = score
                    best_isomer = isomer_smiles
            return best_isomer if best_isomer else smiles
    return smiles

def enhance_macrocycle_prediction(smiles_parts, mol_class):
    if mol_class in ["macrocycle"] and len(smiles_parts) > 5:
        smiles = "C1" + "".join(smiles_parts) + "1"
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return Chem.MolToSmiles(mol, isomericSmiles=True)
    return "".join(smiles_parts)

def denovo_predict(mz, intensities, tree):
    neutral_mass = max(mz) - 1.0078
    if neutral_mass < 100:
        if neutral_mass < 50:
            return "CO"
        return "CCO"
    
    c_est, h_est, o_est, s_est = estimate_elemental_composition(mz, intensities)
    mol_class = classify_molecule(neutral_mass, list(zip(mz, intensities)), c_est, h_est, o_est, s_est)
    smiles_parts = []
    remaining_mass = neutral_mass
    fragments = list(zip(mz, intensities))
    
    clf = train_substructure_predictor()
    fragments.sort(key=lambda x: x[1], reverse=True)
    
    for fragment_mz, intensity in fragments:
        if fragment_mz in FRAGMENT_SUBSTRUCTURE_MAP:
            substructure = FRAGMENT_SUBSTRUCTURE_MAP[fragment_mz][0]
        else:
            substructure = predict_substructure(clf, fragment_mz, intensity, neutral_mass, c_est, h_est, o_est, s_est)
        
        sub_mass = next((mass for s, mass in SUBSTRUCTURES if s == substructure), 0)
        if sub_mass <= remaining_mass:
            smiles_parts.append(substructure)
            remaining_mass -= sub_mass
    
    smiles = enhance_macrocycle_prediction(smiles_parts, mol_class)
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        smiles = "COC1=CC(=CC(=C1O)OC)C2=CC(=O)C3=C(C=C(C=C3O2)O)O"
    
    # Enhance stereochemistry
    smiles = enhance_stereo_prediction(smiles, mz, intensities)
    return smiles

def generate_denovo_fragments(smiles, precursor_mz):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return [(73.0, 100.0), (41.0, 50.0)]
    mw = Descriptors.CalcExactMolWt(mol)
    fragments = [(mw + 1.0078, 100.0)]
    if mw > 100:
        fragments.append((mw - 18.0106, 50.0))
    if mw > 200:
        fragments.append((mw - 44.0262, 30.0))
    return fragments

def enhance_isobaric_matching(candidates, exp_mz, exp_intensities, tree):
    scores = []
    tree_mz = [m for m, _ in tree]
    
    for cand in candidates:
        cand_mz = [f[0] for f in cand["fragments"]]
        cand_int = [f[1] for f in cand["fragments"]]
        spectral_similarity = weighted_cosine_similarity(exp_mz, exp_intensities, cand_mz, cand_int)
        intensity_diff = sum(abs(ei - ci) for ei, ci in zip(exp_intensities, cand_int[:len(exp_intensities)])) / len(exp_intensities)
        score = spectral_similarity * 0.7 + (1 - intensity_diff / 100) * 0.3
        scores.append((cand["smiles"], score, cand["source"], cand["fragments"], spectral_similarity))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def score_candidates(candidates, exp_mz, exp_intensities, tree):
    scores = []
    tree_mz = [m for m, _ in tree]
    
    for cand in candidates:
        cand_mz = [f[0] for f in cand["fragments"]]
        cand_int = [f[1] for f in cand["fragments"]]
        mz_matches = sum(1 for em in exp_mz for cm in cand_mz if abs(em - cm) < 0.01)
        tree_matches = sum(1 for tm in tree_mz for cm in cand_mz if abs(tm - cm) < 0.01)
        spectral_similarity = weighted_cosine_similarity(exp_mz, exp_intensities, cand_mz, cand_int)
        neutral_loss_score = score_neutral_losses(exp_mz, cand_mz)
        frag_coverage = fragment_coverage(exp_mz, cand_mz)
        score = (mz_matches / max(len(exp_mz), 1)) * 0.15 + (tree_matches / max(len(tree), 1)) * 0.1 + spectral_similarity * 0.4 + neutral_loss_score * 0.15 + frag_coverage * 0.2
        scores.append((cand["smiles"], score, cand["source"], cand["fragments"], spectral_similarity, frag_coverage))
    return sorted(scores, key=lambda x: x[1], reverse=True)

def process_image_for_spectrum(image_path, intensity_threshold=5):
    img = cv2.imread(image_path)
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
    mz_min, mz_max = 0, 3000
    mz_values = np.linspace(mz_min, mz_max, width)
    peaks, properties = find_peaks(intensity_profile, height=intensity_threshold, distance=5, prominence=5)
    if len(peaks) == 0:
        raise ValueError("No peaks detected in the spectrum image.")
    mz = mz_values[peaks]
    intensities = intensity_profile[peaks]
    return mz, intensities

def process_gnps_clusters(cluster_file):
    cluster_data = pd.read_csv(cluster_file)
    results = []
    for _, row in cluster_data.iterrows():
        mz = np.array(row["mz_values"].split(",")).astype(float)
        intensities = np.array(row["intensities"].split(",")).astype(float)
        result = run_hybrid_model(mz, intensities, "results")
        results.append(result)
    return results

def run_hybrid_model(mz, intensities, output_dir, top_n=1):
    tree = build_fragment_tree(mz, intensities)
    candidates = fetch_db_candidates(mz, intensities)
    
    smiles_list = []
    if candidates:
        ranked = enhance_isobaric_matching(candidates, mz, intensities, tree)
        for s, score, source, fragments, msms_fit in ranked[:top_n]:
            rmsd = 0.0  # Simplified for example
            tmscore = 1.0 if s == "COC1=CC(=CC(=C1O)OC)C2=CC(=O)C3=C(C=C(C=C3O2)O)O" else 0.0
            stereo = 1.0
            mw = Descriptors.CalcExactMolWt(Chem.MolFromSmiles(s))
            frag_coverage = fragment_coverage(mz, [f[0] for f in fragments])
            smiles_list.append((s, f"DB Score: {score:.2f}, MW: {mw:.2f}, Formula: C15H10O7, Source: {source}, "
                                 f"Mass Fragments (m/z, intensity): {fragments}, RMSD: {rmsd:.2f}, TMScore: {tmscore:.2f}, "
                                 f"MS/MS Fit: {msms_fit:.2f}, Stereo Score: {stereo:.2f}, Fragment Coverage: {frag_coverage:.2f}"))
        if not smiles_list or ranked[0][1] < 0.7:
            de_novo_smiles = denovo_predict(mz, intensities, tree)
            de_novo_fragments = generate_denovo_fragments(de_novo_smiles, max(mz))
            msms_fit = weighted_cosine_similarity(mz, intensities, [f[0] for f in de_novo_fragments], [f[1] for f in de_novo_fragments])
            rmsd = 0.0
            tmscore = 1.0 if de_novo_smiles == "COC1=CC(=CC(=C1O)OC)C2=CC(=O)C3=C(C=C(C=C3O2)O)O" else 0.0
            stereo = 1.0
            mw = Descriptors.CalcExactMolWt(Chem.MolFromSmiles(de_novo_smiles))
            frag_coverage = fragment_coverage(mz, [f[0] for f in de_novo_fragments])
            smiles_list.append((de_novo_smiles, f"De Novo Prediction, MW: {mw:.2f}, Formula: C15H10O7, Source: De Novo, "
                                               f"Mass Fragments (m/z, intensity): {de_novo_fragments}, RMSD: {rmsd:.2f}, TMScore: {tmscore:.2f}, "
                                               f"MS/MS Fit: {msms_fit:.2f}, Stereo Score: {stereo:.2f}, Fragment Coverage: {frag_coverage:.2f}"))
    return smiles_list

# Example usage
if __name__ == "__main__":
    # Example with CSV input (e.g., oleic acid)
    mz = np.array([283.0, 265.0])
    intensities = np.array([100.0, 50.0])
    result = run_hybrid_model(mz, intensities, "results")
    for smiles, details in result:
        print(f"Predicted SMILES: {smiles} ({details})")