# EMBER2024_Lifghtgbm

A comprehensive repository for malware detection using LightGBM, featuring utilities for dataset download, conversion, training, and inference. This project is based on the EMBER 2024 dataset and includes enhanced extraction, efficient training pipelines, and malware detector scripts.

---

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Dataset Download](#dataset-download)
- [Dataset Conversion](#dataset-conversion)
- [Model Training](#model-training)
- [Malware Detection](#malware-detection)
- [References](#references)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository provides scripts and guides for working with the EMBER 2024 malware dataset. It supports:

- Downloading and preparing the EMBER dataset (with support for splits and file types)
- Converting raw samples into feature-rich formats
- Training LightGBM models for malware detection
- Running trained models for inference and evaluation

---

## System Requirements

To ensure smooth operation, please confirm your system meets the following minimum requirements:

- **Python version:** 3.9.0
- **Memory (RAM):** 16 GB DDR5 (for processing ~150,000 samples)
- **Processor:** Intel i5 or AMD Ryzen 5 (minimum recommended for model training)
- **Disk Space:** ~50 GB free (for the full EMBER 2024 dataset and processed files)

---

## Directory Structure

```
EMBER2024_Lifghtgbm/
├── thrember/                     # Dataset download utilities and scripts
│   └── download.py               # Dataset downloader script
├── test.py                       # Example usage for dataset download
├── enhanced_ember_extractor.py   # Dataset conversion to features
├── ember_focused_extractor.py    # Ultra-focused features extractor
├── train_malware_models.py       # Model training script (LightGBM/XGBoost)
├── TRAINING_COMMANDS_GUIDE.txt   # Detailed training command scenarios and parameters
├── training_commands_guide.txt   # Model training command references (alias)
├── malware_detactor.py           # Malware detection script
├── README.md
└── ... (other files)
```

---

## Installation

### Prerequisites

- Python 3.9.0
- pip

### Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

Additional dependencies may include:

```bash
pip install lightgbm pandas numpy tqdm pyarrow huggingface_hub xgboost scikit-learn seaborn matplotlib
```

---

## Dataset Download

Download the EMBER dataset using the provided Python utilities in the `thrember` directory.

### Using the Python API

You can use `download.py` directly:

```python
from download import download_dataset

download_dataset(download_dir="path/to/your/dataset", split="all", file_type="all")
```

Or run the example script:

```bash
python test.py
```

### Command-line Usage

You can adapt the code in `download.py` to create an argument parser for command-line options. 
If refactored, you could use:

```bash
python thrember/download.py --download_dir "path/to/your/dataset" --split "all" --file_type "all"
```
**Parameters:**
- `download_dir`: Target folder for the downloaded dataset.
- `split`: Choose from `[all, train, test, challenge]`
- `file_type`: Choose from `[all, PE, Win32, Win64, Dot_Net, APK, ELF, PDF]`

---

## Dataset Conversion

### Convert to Enhanced Feature Format

To convert the downloaded EMBER dataset into feature-rich format using `enhanced_ember_extractor.py`:

```bash
python enhanced_ember_extractor.py --input data/ember2024/ --output data/processed/
```
- `--input`: Path to the downloaded EMBER dataset.
- `--output`: Path where processed feature files will be written.

### Convert to Ultra-Focused Feature Format

To extract only ultra-focused features (histogram, byteentropy, entropy fields, label, family) using `ember_focused_extractor.py`:

```bash
python ember_focused_extractor.py
```
By default, this script expects:
- Input directory: `a:/Collage/PROJECT/antivirus/ml2/data`
- Output directory: `a:/Collage/PROJECT/antivirus/ml2/ember2024_ultra_focused_features`

You can modify the `main()` function or add argument parsing for custom paths.

#### What it Does

- Processes all `.jsonl` files in the input directory.
- Extracts only the following features:  
  - `histogram_0` to `histogram_255` (256 features)
  - `byteentropy_0` to `byteentropy_255` (256 features)
  - `label`, `family`, `family_confidence`
  - All entropy fields (`general_entropy`, `strings_entropy`, `section_overlay_entropy`, and nested entropy fields)
- Saves output as `.parquet` files in the output directory.
- Creates sample CSVs and schema documentation.

#### Example Output Files
- `2023-09-24_2023-09-30_Win32_train_ultra_focused.parquet`
- `ultra_focused_extraction_schema.json`
- `column_schema.json`

#### WARNING

**High RAM Usage:**  
Processing large `.jsonl` files, especially with 150,000+ samples, requires significant memory (16GB+ recommended).  
If you encounter MemoryError or system slowdown:
- Increase RAM or swap space.
- Process files in smaller batches or subsets.
- Monitor system resources and close unused applications.

**Other Warnings:**  
- Make sure your input directory contains only EMBER `.jsonl` dataset files.
- Output files may be very large (hundreds of MBs to several GBs).
- Ensure sufficient free disk space before starting conversion.
- Some corrupted or incomplete `.jsonl` lines may trigger warnings but will be skipped automatically.
- Always verify output files and schema for completeness.

---

## Model Training

> **Note:**  
> Models are trained on converted EMBER2024 `.jsonl` files to tabular `.parquet` files, which are split by submission dates. This allows for efficient training and validation using temporal splits.

### Main Training Script

All training commands should be executed via:

```bash
python train_malware_models.py [OPTIONS]
```
To see all available options, arguments, and parameter descriptions, run:
```bash
python train_malware_models.py --help
```

### Example Training Commands

#### 1. Production-Ready Training (Win32, LightGBM & XGBoost)
```bash
python train_malware_models.py --category Win32 --data_dir ember2024_ultra_focused_features --output_dir optimal_models --sample_size 1000000 --models lightgbm xgboost --cv_folds 5 --val_size 0.2 --early_stopping_rounds 50 --random_state 42
```

#### 2. Multi-Category Comprehensive Training (Win32, Win64, APK)
```bash
python train_malware_models.py --category Win32 Win64 APK --data_dir ember2024_ultra_focused_features --output_dir multi_category_models --sample_size 1500000 --models lightgbm xgboost --cv_folds 5 --val_size 0.2
```

#### 3. All Categories Training (Universal Detector)
```bash
python train_malware_models.py --category all --data_dir ember2024_ultra_focused_features --output_dir universal_models --sample_size 2000000 --models lightgbm xgboost --cv_folds 5 --val_size 0.2
```

#### 4. Optimized LightGBM Training
```bash
python train_malware_models.py --category Win32 --data_dir ember2024_ultra_focused_features --output_dir best_lightgbm_models --sample_size 1500000 --models lightgbm --cv_folds 7 --val_size 0.15 --early_stopping_rounds 100 --random_state 42 --lgb_n_estimators 5000 --lgb_num_leaves 63 --lgb_learning_rate 0.08 --lgb_feature_fraction 0.95 --lgb_bagging_fraction 0.95 --lgb_bagging_freq 3 --lgb_min_child_samples 25
```

#### 5. Fast Development Training
```bash
python train_malware_models.py --category Win32 --data_dir ember2024_ultra_focused_features --output_dir fast_models --sample_size 100000 --models lightgbm xgboost --cv_folds 3 --val_size 0.2 --early_stopping_rounds 30 --random_state 42 --lgb_learning_rate 0.1 --lgb_n_estimators 500 --xgb_learning_rate 0.1 --xgb_n_estimators 500
```

#### 6. Memory-Efficient Training
```bash
python train_malware_models.py --category Win32 --data_dir ember2024_ultra_focused_features --output_dir memory_efficient_models --sample_size 500000 --models lightgbm xgboost --cv_folds 5 --val_size 0.2 --early_stopping_rounds 50 --random_state 42
```

> For more scenarios, parameters, and troubleshooting, see the full [TRAINING_COMMANDS_GUIDE.txt](TRAINING_COMMANDS_GUIDE.txt) in this repository.

---

## Malware Detection

Use `malware_detactor.py` to run inference on new samples.

### Example Usage

```bash
python malware_detactor.py --model model.txt --input data/test_samples/ --output detection_results.csv
```

**Parameters:**
- `--model`: Path to the trained LightGBM model.
- `--input`: Directory of files or features for inference.
- `--output`: Output CSV with detection results.

---

## References

- [EMBER 2024 Dataset](https://github.com/elastic/ember)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Python Documentation](https://docs.python.org/3/)
- [Hugging Face Hub](https://huggingface.co/docs/hub/index)
- [PyArrow Documentation](https://arrow.apache.org/docs/python/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [TRAINING_COMMANDS_GUIDE.txt](TRAINING_COMMANDS_GUIDE.txt)

---

## License

See [LICENSE](LICENSE) for details.

---

## Contact

For questions or contributions, create an issue or pull request in this repository.
