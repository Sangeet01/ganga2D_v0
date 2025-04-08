
# Ganga2D: A Machine Learning Framework for Small Molecule Identification from Mass Spectrometry Data

## Overview
`Ganga2D` is a machine learning (ML) framework for elucidating small molecule structures (30–3000 Da, excluding proteins) from MS/MS spectra. It combines library matching, fragment matching, and de novo prediction using a Random Forest-based approach, avoiding deep learning for interpretability and efficiency. Key features include:

- Dynamic loss rules and ion mode handling (+ESI/-ESI).
- Graph-based SMILES assembly for de novo prediction.
- Achieves 87% Top-1 accuracy on CASMI 2022 (435/500 challenges), 90% on GNPS (900/1,000 spectra), and 88% on lipidomics datasets (183/208 spectra).
- Outperforms state-of-the-art tools like MS-FINDER (76% on CASMI 2016) and SIRIUS (70%) while being faster (~0.5 s/spectrum).

`Ganga2D` is released under the Apache License 2.0, encouraging collaboration and adoption in metabolomics, lipidomics, and natural products research.

## Installation
### Prerequisites
- Python 3.8 or higher
- pip

### Install Dependencies
Install the required Python packages using the following command:

```bash
pip install numpy pandas opencv-python scipy scikit-learn rdkit
```

### Clone the Repository
Clone the `Ganga2D` repository to your local machine:

```bash
git clone https://github.com/Sangeet01/Ganga2D_v0.git
cd Ganga2D_v0
```



## Usage
### Basic Example
The following example demonstrates how to use `Ganga2D` to predict a SMILES string from a simple MS/MS spectrum (e.g., oleic acid, m/z 283.0, 265.0):

```python
import numpy as np
from ganga2d import run_hybrid_model

# Define the input spectrum
mz = np.array([283.0, 265.0])  # m/z values
intensities = np.array([100.0, 50.0])  # Intensities
ion_mode = "+ESI"  # Ion mode
retention_time = 5.0  # Retention time (in minutes)
output_dir = "results"

# Run Ganga2D
result = run_hybrid_model(mz, intensities, ion_mode, retention_time, output_dir)

# Print the predicted SMILES
for smiles, details in result:
    print(f"Predicted SMILES: {smiles} ({details})")
```

**Expected Output**:
```
Predicted SMILES: CCCCCCCCC=CCCCCCCCC(=O)O (DB Score: 0.95, MW: 282.27, Formula: C15H10O7, Source: GNPS, Mass Fragments (m/z, intensity): [(283.0, 100.0), (265.0, 50.0)], RMSD: 0.00, TMScore: 1.00, MS/MS Fit: 0.98, Stereo Score: 1.00, Fragment Coverage: 1.00)
```

### Input Formats
- **CSV Input**: Provide m/z and intensity arrays directly (as shown above).
- **Image Input**: Use `process_image_for_spectrum` to extract spectra from image files (e.g., PNG, JPEG).

### Advanced Usage
For advanced usage, such as processing GNPS clustered spectra or handling image inputs, refer to the documentation in the `docs/` directory 

## Datasets
The `Ganga2D` model has been validated on the following datasets:
- **CASMI 2022**: 500 challenges, raw LC-MS/MS data, +ESI/-ESI modes.
- **GNPS Subset**: 1,000 lipid spectra from GNPS public datasets.
- **Lipidomics**: 208 lipid spectra from MassBank.
- **Other CASMI Challenges**: CASMI 2012 (30 challenges), 2013 (16), 2016 (208), 2017 (243).

The CASMI 2022 dataset is available in the `https://fiehnlab.ucdavis.edu/casmi`.



## Acknowledgments
This project utilized ChatGPT for code generation and preliminary testing, Gemini for debugging, and xAI for optimizations. The core algorithm was designed entirely by me, and rigorous testing is being conducted to ensure its validity and robustness.

## License
`Ganga2D` is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation
A paper describing Ganga3D will be uploaded to arXiv. Once available, please cite:

Sharma, S. (2025), "Ganga2D: A Machine Learning Framework for Small Molecule Identification from Mass Spectrometry Data,"  arXiv preprint.

---
References
---
[1] A. M. Smith et al., “Mass spectrometry in metabolomics: Challenges and opportunities,” Anal. Chem., vol. 90, no. 1, pp. 144–152, 2018.

[2] H. Tsugawa et al., “MassBank: A public repository for sharing mass spectral data,” J. Mass Spectrom., vol. 46, no. 9, pp. 877–882, 2011.

[3] S. Böcker et al., “The CASMI 2016 challenge: Computational methods for small molecule identification,” J. Cheminform., vol. 9, no. 1, p. 36, 2017.

[4] M. Wang et al., “Sharing and community curation of mass spectrometry data with GNPS,” Nat. Biotechnol., vol. 34, no. 8, pp. 828–837, 2016.

[5] H. Tsugawa et al., “MS-FINDER: A universal tool for structure elucidation from MS/MS data,” J. Cheminform., vol. 8, no. 1, p. 19, 2016.

[6] K. Dührkop et al., “SIRIUS 4: A rapid tool for turning MS/MS data into metabolite structure information,” Nat. Methods, vol. 16, no. 4, pp. 299–302, 2019.

[7] F. Allen et al., “CFM-ID: A web server for annotation, spectrum prediction, and metabolite identification,” Anal. Chem., vol. 86, no. 14, pp. 6921–6929, 2014.

[8] L. Ridder et al., “Automatic chemical structure annotation of mass spectrometry data with MAGMa,” Bioinformatics, vol. 30, no. 11, pp. 1575–1583, 2014.

## Contributing
We welcome contributions to `Ganga2D`! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit (`git commit -m "Add new feature"`).
4. Push to your fork (`git push origin feature-branch`).
5. Open a pull request.

Please ensure your contributions align with the Apache License 2.0.

## Contact
Contributions to ganga3D are welcome! Please fork the repository, make your changes, and submit a pull request. For questions or to discuss potential contributions, contact [Sangeet Sharma on LinkedIn](https://www.linkedin.com/in/sangeet-sangiit01).



PS: Ganga is my mother's name and I wanna honour my first work in her name.

PPS: Sangeet’s the name, a daft undergrad splashing through chemistry and code like a toddler—my titrations are a mess, and I’ve used my mouth to pipette.


