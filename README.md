# VIs_to_LAI: Simulate Leaf Area Index from Vegetation Indices

**Authors**: Jonghan Ko at Chonnam National University and Chi Tim Ng at Hang Seng University of Hong Kong

**Collaborator**: Jong-oh Ban at Hallym Polytechnic University

**GitHub Repository**: [https://github.com/RS-iCM/VIs_to_LAI](https://github.com/RS-iCM/VIs_to_LAI)

**HuggingFace Dataset**: [https://huggingface.co/datasets/jonghanko/VIs_to_LAI/tree/main](https://huggingface.co/datasets/jonghanko/VIs_to_LAI/tree/main)

---

## Overview

VIsToLAI is a Python-based, open-source software framework designed to estimate leaf area index (LAI) from time series of satellite-derived vegetation indices (NDVI, RDVI, OSAVI, and MTVI₁). By integrating empirical regression, Log–log, and machine learning modules, VIsToLAI offers a flexible, scalable workflow that bypasses destructive sampling and intensive calibration. Pretrained models, an extensible API, and interactive Jupyter notebooks streamline data ingestion, model execution, and visualization. Demonstrated on staple crops under varied conditions, VIsToLAI accurately reconstructs LAI dynamics and integrates seamlessly into remote sensing workflows for precision agriculture, crop monitoring, and ecological modeling.

---

## Features

- **Multiple Vegetation Indices**: Support for NDVI, RDVI, OSAVI, and MTVI₁
- **Three Modeling Approaches**: 
  - Empirical exponential regression
  - Log–log regression
  - Machine learning (Extra Trees, Gradient Boosting, DNN)
- **Flexible Workflows**: 
  - 1D time-series simulation for point/field data
  - 2D geospatial simulation for regional mapping
- **Pretrained Models**: Ready-to-use models for rice, barley, wheat, and maize
- **Interactive Notebooks**: Jupyter notebooks for reproducible workflows
- **Extensible API**: Modular design for custom indices and algorithms
- **Built-in Visualization**: Time-series plots, scatter diagnostics, and geospatial maps
- **Ensemble Methods**: Combine multiple models for improved accuracy

---

## Installation

### Prerequisites

- Python ≥ 3.8 (recommended: Python 3.10+)
- pip package manager

### Quick Install

Install the package with all dependencies:

```bash
pip install -r requirements.txt
```

Or install as an editable package:

```bash
pip install -e .
```

### Full Installation (with 2D/spatial features and Jupyter)

For 2D mapping and geospatial analysis:

```bash
pip install -e ".[all]"
```

This includes:
- Core dependencies
- Cartopy (for 2D mapping and shapefile support)
- tqdm (progress bars)
- Jupyter notebooks

### Optional Extras

```bash
# For 2D/spatial analysis only
pip install -e ".[2d]"

# For Jupyter development only
pip install -e ".[dev]"
```

### Docker Installation

Build and run with Docker:

```bash
# Build the image
docker build -t vis-to-lai-crops .

# Run with Docker Compose (recommended)
docker-compose up --build
```

Access Jupyter Lab at `http://localhost:8888` (check container logs for token).

---

## Quick Start

### 1D Time-Series Simulation

Run a notebook for 1D LAI simulation:

```bash
jupyter notebook RUN_Python_Rice.ipynb
```

Or use Python directly:

```python
from codes.sim_VIs_to_LAI_crops import main
import os

# Set paths
path = os.path.abspath(os.getcwd())
para_FN = path + '/data/empirical_reg_parameters_rice.txt'
wobs_FN2 = path + '/data/Rice_LAI_n_VIs.csv'
data_FN = path + '/data/Rice_FN_NICS_2021.csv'
output_FN = path + '/outputs/SLAI_rice.out'

# Model files
DNN_FN = path + '/models/rice_NN.h5'
pkl_FN = path + '/models/pickle_extra_trees_Rice.pkl'
pkl_seq_FN = path + '/models/pickle_extra_trees_Rice_seq.pkl'

# Run simulation
# reg_opt: 0=DNN, 1=ML, 3=NDVI-based, 4=four VIs-based, 5=log-log, 7=Ensemble
main(DNN_FN, pkl_FN, pkl_seq_FN, 
     reg_opt=7,    # Ensemble method
     plot_opt=1,   # Show plot
     file_opt=1,   # Save output
     flag=5.5,     # Max LAI value
     para_FN=para_FN,
     wobs_FN2=wobs_FN2,
     data_FN=data_FN,
     output_FN=output_FN)
```

### 2D Geospatial Simulation

For 2D regional mapping:

```bash
jupyter notebook RUN_Python_LAI_2D_Rice.ipynb
```

---

## Available Notebooks

### 1D Time-Series Notebooks
- `RUN_Python_Rice.ipynb` - Rice LAI simulation
- `RUN_Python_Barley.ipynb` - Barley LAI simulation
- `RUN_Python_Wheat.ipynb` - Wheat LAI simulation
- `RUN_Python_Maize.ipynb` - Maize LAI simulation

### 2D Geospatial Notebooks
- `RUN_Python_LAI_2D_Rice.ipynb` - Regional rice LAI mapping
- `RUN_Python_LAI_2D_Maize.ipynb` - Regional maize LAI mapping

---

## Model Options

The framework supports multiple regression options (`reg_opt` parameter):

- **0**: Deep Neural Network (DNN)
- **1**: Machine Learning - Extra Trees Regressor
- **2**: Machine Learning - Sequential (with temporal features)
- **3**: NDVI-based empirical regression
- **4**: Four VIs-based empirical regression (ensemble of all VIs)
- **5**: Log-log regression
- **6**: Ensemble 1 (DNN + ML + VIs + Log-log)
- **7**: Ensemble 2 (ML + VIs + Log-log) - **Recommended**

---

## Project Structure

```
VIs_to_LAI_crops/
├── codes/                      # Core Python modules
│   ├── sim_VIs_to_LAI_crops.py      # Main 1D simulation module
│   ├── empirical_VIs_to_LAI_2D_*.py # 2D empirical modules
│   └── each_crop_model/        # Crop-specific models
├── data/                       # Input data (CSV, OBS, TXT)
│   ├── *_LAI_n_VIs.csv        # Training data
│   └── empirical_reg_parameters_*.txt  # Regression parameters
├── models/                     # Pretrained models
│   ├── *_NN.h5                # DNN models
│   └── pickle_*.pkl           # ML models
├── outputs/                    # Simulation outputs
│   └── SLAI_*.out             # Simulated LAI files
├── class_map_*/                # 2D class maps
├── vis_*/                      # 2D vegetation indices
├── Shape_*/                    # Shapefile boundaries (2D)
├── RUN_Python_*.ipynb          # Jupyter notebooks
├── setup.py                    # Package setup
├── requirements.txt            # Python dependencies
└── Dockerfile                  # Docker configuration
```

---

## Requirements

### Core Dependencies
- numpy ≥ 1.20.0
- pandas ≥ 1.3.0
- scipy ≥ 1.7.0
- scikit-learn ≥ 1.0.0
- matplotlib ≥ 3.4.0
- tensorflow ≥ 2.8.0
- keras ≥ 2.8.0
- h5py ≥ 3.0.0
- pyyaml ≥ 5.4.0

### Optional Dependencies (for 2D features)
- cartopy ≥ 0.20.0 (geospatial mapping)
- tqdm ≥ 4.64.0 (progress bars)

### Development Dependencies
- jupyter ≥ 1.0.0
- ipykernel ≥ 6.0.0
- notebook ≥ 6.4.0

See `requirements.txt` for a complete list.

---

## Usage Examples

### Example 1: Rice LAI Simulation with Ensemble Method

```python
from codes.sim_VIs_to_LAI_crops import main
import os

path = os.path.abspath(os.getcwd())
main(
    DNN_FN=path + '/models/rice_NN.h5',
    pkl_FN=path + '/models/pickle_extra_trees_Rice.pkl',
    pkl_seq_FN=path + '/models/pickle_extra_trees_Rice_seq.pkl',
    reg_opt=7,  # Ensemble method
    plot_opt=1,
    file_opt=1,
    flag=5.5,
    para_FN=path + '/data/empirical_reg_parameters_rice.txt',
    wobs_FN2=path + '/data/Rice_LAI_n_VIs.csv',
    data_FN=path + '/data/Rice_FN_NICS_2021.csv',
    output_FN=path + '/outputs/SLAI_rice.out'
)
```

---

## Citation

If you use this software in your research, please cite:

```bibtex
@software{vistolai2024,
  author = {Ko, Jonghan and Ng, Chi Tim},
  title = {VIs_to_LAI: Simulate Leaf Area Index from Vegetation Indices},
  year = {2024},
  url = {https://github.com/RS-iCM/VIs_to_LAI}
}
```

---

## License

[Specify your license here - e.g., MIT, Apache 2.0, etc.]

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## Support

For questions, issues, or contributions, please visit:
- **GitHub Issues**: [https://github.com/RS-iCM/VIs_to_LAI/issues](https://github.com/RS-iCM/VIs_to_LAI/issues)
- **HuggingFace**: [https://huggingface.co/datasets/jonghanko/VIs_to_LAI](https://huggingface.co/datasets/jonghanko/VIs_to_LAI)

---

## Acknowledgments

- Chonnam National University
- Hang Seng University of Hong Kong
- Hallym Polytechnic University

---

**Last Updated**: August 2025
