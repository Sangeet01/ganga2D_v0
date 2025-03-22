# ganga2D_v0

# ganga2D_v0: 2D Structure Prediction

A Python-based tool for predicting SMILES strings from mass spectrometry data, supporting molecules up to 3000 Da. The model uses spectral databases, fragment matching, and de novo prediction to identify molecular structures.
----
Author:
---
  Sangeet Sharma, Nepal

---

Features
Predicts SMILES from mass spectra (CSV or image input).
Supports small to large molecules (100 Da to 3000 Da).
Integrates spectral databases: MassBank, NIST, METLIN, mzCloud, GNPS, ChemSpider.
Performs fragment matching, spectral similarity (cosine), and de novo prediction.
Outputs performance metrics: RMSD, TMScore, MS/MS Fit, Stereo Score.

----

Installation
1. Clone the Repository:
    git clone https://github.com/Sangeet01//ganga2D_v0

2. Install Dependencies: Ensure Python 3.6+ is installed, then run:
    pip install numpy pandas opencv-python scipy rdkit requests scikit-learn

---

Usage
1. Prepare Input:
   CSV: A file with columns for m/z and intensity (e.g., mass, intensity).
   Image: A mass spectrum image (e.g., PNG).

2. Run the Script:
    python ganga2D_v0.py
    Follow prompts to specify input type (csv or image), file path, and output directory.

3. Output:
   Results are saved to result.txt in the specified output directory.
   Example output:
   Predicted SMILES for the mass spectrum:
      C=CCSS(=O)CC=C (DB Score: 0.98, Formula: C6H10OS2, Source: MassBank, Mass Fragments (m/z, intensity): [(73.0, 100.0), (41.0, 50.0)], RMSD: 0.00, TMScore: 1.00, MS/MS Fit: 0.99, Stereo Score: 1.00)

      Example
       For a spectrum of allicin (MW 162.27 Da):
           -Input: Image file allicin_spectrum.png.
           -Command: Run the script and provide the file path.
           -Output: Correctly predicts C=CCSS(=O)CC=C with high scores.
      For a large peptide (MW ~3000 Da):
            -Input: CSV file with m/z 3001 peak.
            -Output: Predicts a peptide SMILES with fragments like [(3001.0, 100.0), (2873.0, 60.0)].

Model Details
   -Size Range: 100 Da to 3000 Da.
   -Databases: Simulated entries from MassBank, NIST, METLIN, mzCloud, GNPS, ChemSpider.
   -Methods:
      -Fragment matching with cosine similarity.
      -De novo prediction for small to large molecules (e.g., peptides, macrolides).
      -Scoring: RMSD (m/z accuracy), TMScore (structural similarity), MS/MS Fit (spectral match), Stereo Score (stereochemistry).

Limitations
   -Simulated databases; real API integration recommended for production.
   -De novo prediction may oversimplify large molecules (>1500 Da).
   -Fragmentation patterns for large molecules (e.g., peptides) are simplified.


----

Contributing
   Contributions are welcome! Please:

Fork the repository.
   Create a feature branch (git checkout -b feature-name).
   Commit changes (git commit -m "Add feature").
   Push to the branch (git push origin feature-name).
   Open a pull request.
 
---
  



----

Contact
   For issues or questions, open an issue on GitHub or contact: www.linkedin.com/in/sangeet-sangiit01













## Why Use ganga2D_v0?
- Quick 2D structure guesses for pharmaceuticals and natural products from spectral data.  
- Useful for initial modeling or hypothesis generation.  
- Open-source under MIT License.

License
---
   Ganga2D is licensed under the Apache License 2.0. See the [LICENSE] file for details.

## Contributing
Fork, improve (e.g., enhance FTIR refinement, add de novo prediction), and submit pull requests!

PS: Ganga is my mother's name and I wanna honour my first work in her name.

PPS: Sangeet’s the name, a daft undergrad splashing through chemistry and code like a toddler—my titrations are a mess, and I’ve used my mouth to pipette. 
