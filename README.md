# VIs_to_LAI: Simulate leaf area index from vegetation indices

**Authors**: Jonghan Ko at Chonnam National University and Chi Tim Ng at Hang Seng University of Hong Kong

**Collaborator**: Jong-oh Ban at Hallym Polytechnic University

**Repository for the model**: https://github.com/RS-iCM/VIs_to_LAI

**Repository for bigger data**: https://huggingface.co/datasets/jonghanko/VIs_to_LAI/tree/main

---

## Overview

VIsToLAI is a Python-based, open-source software framework designed to estimate leaf area index (LAI) from time series of satellite-derived vegetation indices (NDVI, RDVI, OSAVI, and MTVI₁). By integrating empirical regression, Log–log, and machine learning modules, VIsToLAI offers a flexible, scalable workflow that bypasses destructive sampling and intensive calibration. Pretrained models, an extensible API, and interactive Jupyter notebooks streamline data ingestion, model execution, and visualization. Demonstrated on staple crops under varied conditions, VIsToLAI accurately reconstructs LAI dynamics and integrates seamlessly into remote sensing workflows for precision agriculture, crop monitoring, and ecological modeling.

---

## Features

- Support for multiple vegetation indices: NDVI, RDVI, OSAVI, and MTVI₁
- Three modeling approaches: empirical exponential, Log–log, and machine learning regression
- Extendable API for custom indices and algorithms
- Pretrained models and reproducible Jupyter notebooks for rice, barley, wheat, and maize
- 1D and 2D simulation workflows for time-series and geospatial projections
- Built-in visualization for time-series plots, scatter diagnostics, and geospatial maps
- Modular design for adaptation to various crops and regions

---

## Requirements

- Python ≥ 3.10  
- numpy  
- pandas  
- matplotlib  
- scikit-learn  
- scipy 

Install dependencies using:

```bash
pip install -r requirements.txt

