# ganga3D

# ganga3D: 3D Structure Prediction from MS and FTIR Data

ganga3D_v0 generates 3D molecular structures (SDF files) of pharmaceutical compounds—from simple drugs like aspirin to complex natural products like paclitaxel—using mass spectrometry (MS) and optional Fourier-transform infrared spectroscopy (FTIR) data. It matches MS fragments against a library to predict a 2D structure (SMILES), then refines the 3D conformation with FTIR when provided, offering a rough but useful 3D arrangement.


## Pipeline Explanation
1. **Input**:  
   - MS data (CSV: m/z, intensity).  
   - Optional FTIR data (CSV: wavenumbers).  
2. **Spectral Matching**:  
   - MS data is matched to a library (e.g., NIST, MassBank) using CosineGreedy (tolerance: 1.0 Da).  
   - Retrieves the SMILES string of the best match based on fragment similarity.  
3. **3D Generation**:  
   - Converts SMILES to a 3D molecule with RDKit.  
   - If FTIR is provided, refines conformers using a Random Forest model and IR peak rules.  
4. **Output**:  
   - SDF file with a rough 3D structure if a match is found, or an error if no match exists.

**Goal**: Provide a quick 3D structure estimate for known compounds or new natural products, leveraging fragment similarity. Ideal as a starting point for further validation (e.g., NMR).

Tested on few compounds like Aspirin, Ibuprofen and Pacitaxel due to lack of data. 

## Installation
1. **Set Up Python**:  
   - Requires Python 3.12. Download and install from [python.org](https://www.python.org/downloads/release/python-3120/) if needed.  
2. **Install Dependencies**:  
   ```bash
   pip install -r requirements.txt
   ```
3. **Download a Spectral Library**:  
   - **NIST Mass Spectral Library** (recommended, paid):  
     - Visit [NIST](https://www.nist.gov/srd/nist-standard-reference-database-1a).  
     - Purchase and download the `.mgf` file (e.g., NIST 23).  
     - Save to `C:/path/to/nist_library.mgf`.  
   - **MassBank** (free option):  
     - Visit [MassBank](https://massbank.eu/MassBank/).  
     - Download the `.mgf` file from the "Download" section.  
     - Save to `C:/path/to/massbank.mgf`.  
   - **GNPS** (free, natural products):  
     - Visit [GNPS](https://gnps.ucsd.edu/).  
     - Sign up, download the `.mgf` library from the "Library" section.  
     - Save to `C:/path/to/gnps_library.mgf`.  
   - Update the `library_mgf` path in `ganga3D.py` (line 29) to your library’s location.

## Usage
1. **Prepare Input Files**:  
   - MS data: CSV with two columns, no header (e.g., `m/z,Intensity`).  
   - FTIR data (optional): CSV with one column, no header (e.g., `Wavenumber`).  
2. **Run the Tool**:  
   ```bash
   python ganga3D.py
   ```
   - Enter paths for MS file, FTIR file (or press Enter to skip), and output directory.  
3. **Output**:  
   - Success: An SDF file in the output directory (e.g., `F:/paris/aspirin_mass.sdf`).  
   - Failure: Error message if no match is found.

## Example
```bash
Enter MS data CSV file path: C:/Users/is/Desktop/aspirin_mass.csv
Enter IR data CSV file path: C:/Users/is/Desktop/aspirin_ftir.csv
Enter output directory: F:/paris
```
- Output: `Generated 'F:/paris/aspirin_mass.sdf'`

## Limitations
- **Library Dependency**: Only works for compounds with similar fragments in the library. Errors if no match (score < 0.5).  
- **Conformational Accuracy**: FTIR refinement is basic—3D structures are approximate, not exact.  
- **No De Novo Prediction**: Relies on library matches, not full structure prediction from scratch.

## Why Use ganga3D_v0?
- Quick 3D structure guesses for pharmaceuticals and natural products from spectral data.  
- Useful for initial modeling or hypothesis generation.  
- Open-source under MIT License.

## License
Licensed under the MIT License—see `LICENSE` file.

## Contributing
Fork, improve (e.g., enhance FTIR refinement, add de novo prediction), and submit pull requests!

