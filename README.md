# FAIR-Eval: Feature-Augmented IRT for Reliable and Efficient LLM Benchmarking

Core code for **FAIR-Eval**, a feature-augmented semantic IRT pipeline for LLM benchmarking.

![FAIR-Eval architecture](assets/figures/fdc_model.png)

[PDF version](assets/figures/fdc_model.pdf)

## Overview

Given prepared item tables and model response tables, FAIR-Eval:

1. builds a unified HDF5 benchmark file,
2. trains a semantic calibration model with item-level cross-validation,
3. exports item discrimination and difficulty parameters, and
4. exports ranked model ability (`theta`) estimates.

## Repository Contents

- `build_hdf5_from_prepared.py`: converts prepared response/item parquet files into a single HDF5 file.
- `calibration_model_train.py`: trains the semantic 2PL calibration model with cross-validation.
- `item_params_inference.py`: exports item-level IRT parameters (`a`, `b`) from a trained checkpoint.
- `theta_inference.py`: exports model-level ability estimates (`theta`) from a trained checkpoint.

## Data Format

`build_hdf5_from_prepared.py` expects:

- a response parquet file with at least `scenario`, `question`, `model`, `score`
- an item parquet file with at least `scenario`, `question`

Optional item columns include:

- `question_embedding`
- `rationale` or `solution`
- `rationale_embedding`
- item feature columns such as question-length, answer-length, interaction, and task-type features

## Quick Start

### 1. Build the HDF5 benchmark file

```bash
python build_hdf5_from_prepared.py \
  --responses path/to/irt_data_responses.parquet \
  --items path/to/irt_data_items.parquet \
  --output path/to/irt_data.hdf5
```

### 2. Train the calibration model

```bash
python calibration_model_train.py \
  --h5 path/to/irt_data.hdf5 \
  --output-dir path/to/model_train_outputs \
  --epochs 200 \
  --batch-size 256 \
  --lr 1e-3 \
  --n-splits 10
```

### 3. Export item parameters

```bash
python item_params_inference.py \
  --h5 path/to/irt_data.hdf5 \
  --cv-results path/to/model_train_outputs/cv_results.json \
  --output-csv path/to/items_params.csv \
  --fold-select best_auc
```

### 4. Export model ability estimates

```bash
python theta_inference.py \
  --h5 path/to/irt_data.hdf5 \
  --cv-results path/to/model_train_outputs/cv_results.json \
  --output-csv path/to/theta_estimates.csv \
  --fold-select best_auc
```

## Main Outputs

- `irt_data.hdf5`: unified benchmark file used by the full pipeline
- `cv_results.json`: best validation metrics and checkpoint path for each fold
- `feat_norm.pt`: feature normalization statistics reused at inference time
- `fold_*.pt`: trained calibration checkpoints
- `items_params.csv`: inferred item discrimination and difficulty parameters
- `theta_estimates.csv`: ranked model ability estimates

## Environment

Use Python 3.10 or newer. Install the base dependencies with:

```bash
pip install -r requirements.txt
```

If you need a CUDA-enabled PyTorch build, install the matching wheel from the official PyTorch index for your platform.

## Citation

If you use this code, please cite:

```bibtex
@misc{fair_eval,
  title={FAIR-Eval: Feature-Augmented IRT for Reliable and Efficient LLM Benchmarking},
  author={To be updated},
  year={2026}
}
```
