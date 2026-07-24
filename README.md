# SegTSF

PyTorch implementation of the paper:

> **SegTSF: Hierarchical Segment Learning For Lightweight Multivariate Time-Series ForeCasting**
> Hyunjun Park, Hee-Gook Jun, Seongyong Kim, and Dong-Hyuk Im
> *Computer Modeling in Engineering & Sciences*, 147(3), 2026.

[Paper](https://www.techscience.com/CMES/v147n3/67919) · [DOI](https://doi.org/10.32604/cmes.2026.082506)

## Overview

SegTSF is a lightweight model for multivariate time-series forecasting.

The model reconstructs an input time series into periodic subsequences and applies hierarchical segment-wise learning to capture:

* Intra-period temporal relationships
* Local patterns within individual segments
* Global relationships across segments

SegTSF improves the representational capacity of linear layers while maintaining a small parameter count and low computational cost.

## Architecture

<p align="center">
  <img src="Figures/Figure1.png" alt="SegTSF architecture" width="850">
</p>

The main implementation is provided in [`models/SegTSF.py`](models/SegTSF.py).

The model consists of:

1. Mean normalization of the input sequence
2. One-dimensional convolution-based local aggregation
3. Period-wise reconstruction of the input
4. Intra-period linear learning
5. Hierarchical learning within and across segments
6. Reconstruction and denormalization of the forecast

## Repository Structure

```text
SegTSF/
├── data/
│   └── data_loader.py       # Dataset loading and preprocessing
├── exp/
│   ├── exp_basic.py
│   └── exp_main.py          # Training, validation, and evaluation
├── layers/                  # Supporting neural-network layers
├── models/
│   └── SegTSF.py            # SegTSF model
├── utils/                   # Metrics, time features, and training utilities
├── Figures/
│   ├── Figure1.png
│   ├── Figure2.png
│   ├── Figure3.png
│   └── Experiments.xlsx     # Detailed experimental results
├── checkpoints/             # Saved model checkpoints
├── main.ipynb               # Example training and evaluation notebook
├── requirements.txt
└── README.md
```

## Environment

The example notebook was created with Python 3.8.20. The main dependencies include:

* Python 3.8
* PyTorch 2.4.1
* NumPy 1.24.4
* pandas 2.0.3
* scikit-learn 1.3.2
* matplotlib 3.7.5
* ptflops 0.7.3

A CUDA-compatible GPU is optional. When CUDA is unavailable, the code automatically uses the CPU.

## Installation

Clone this repository:

```bash
git clone https://github.com/Pigeon1999/SegTSF.git
cd SegTSF
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

To run the notebook locally, install Jupyter if necessary:

```bash
pip install notebook
```

## Datasets

The datasets are not included in this repository.

Create a directory named `Dataset` in the repository root:

```bash
mkdir Dataset
```

Place the downloaded CSV files inside this directory:

```text
Dataset/
├── ETTh1.csv
├── ETTh2.csv
├── ETTm1.csv
├── ETTm2.csv
├── electricity.csv
├── traffic.csv
├── weather.csv
├── Solar.csv
└── exchange_rate.csv
```

The expected format is:

```text
date, feature_1, feature_2, ..., target
```

For the ETT datasets, the target column is expected to be named `OT`.

## Running an Experiment

Open the example notebook:

```bash
jupyter notebook main.ipynb
```

The experiment configuration can be modified in `main.ipynb`.

Example configuration for ETTh1:

```python
from types import SimpleNamespace

config = SimpleNamespace(
    model="SegTSF",
    dataset="ETTh1.csv",

    seq_len=720,
    pred_len=96,
    channels=7,

    batch_size=64,
    learning_rate=0.02,

    period_len=24,
    seg_len_x=1,
    seg_len_y=1,

    lradj="type3",
    epochs=30,
    patience=5,
    criterion="MSE",
)
```

Load the data and run training and evaluation:

```python
import data.data_loader as data_loader
import exp.exp_main as exp_main

train_dataset, train_loader, test_dataset, test_loader, \
vali_dataset, vali_loader = data_loader.data_provider(config)

model = exp_main.train(
    config,
    train_loader,
    vali_loader,
    test_loader,
)

exp_main.test(
    config,
    test_dataset,
    test_loader,
    model,
)
```

When using another dataset, update the following parameters:

* `dataset`: CSV filename
* `channels`: number of variables in the dataset
* `period_len`: period used to divide the time series
* `seq_len`: input sequence length
* `pred_len`: forecasting horizon
* `seg_len_x`: input segment length
* `seg_len_y`: output segment length

The input and prediction lengths should be divisible by `period_len`.

## Outputs

During training, the code reports:

* Training loss
* Validation loss
* Test loss
* Number of model parameters
* Computational complexity
* Maximum GPU memory usage

Early stopping is applied according to the validation loss. Model checkpoints are written to:

```text
checkpoints/ETT_hour_linear.pth/
```

During evaluation, the code reports:

* Mean Squared Error (MSE)
* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Inference time
* Average inference time per sample

A sample prediction plot is saved under:

```text
img/SegTSF.png
```

Detailed experimental results used in the paper are available in:

```text
Figures/Experiments.xlsx
```

## Citation

Please cite the following paper when using this repository:

```bibtex
@article{park2026segtsf,
  title   = {SegTSF: Hierarchical Segment Learning For Lightweight Multivariate Time-Series ForeCasting},
  author  = {Park, Hyunjun and Jun, Hee-Gook and Kim, Seongyong and Im, Dong-Hyuk},
  journal = {Computer Modeling in Engineering \& Sciences},
  volume  = {147},
  number  = {3},
  pages   = {36},
  year    = {2026},
  doi     = {10.32604/cmes.2026.082506}
}
```

## Paper

Hyunjun Park, Hee-Gook Jun, Seongyong Kim, and Dong-Hyuk Im,
“SegTSF: Hierarchical Segment Learning For Lightweight Multivariate Time-Series ForeCasting,”
*Computer Modeling in Engineering & Sciences*, vol. 147, no. 3, 2026.

* Paper: https://www.techscience.com/CMES/v147n3/67919
* DOI: https://doi.org/10.32604/cmes.2026.082506
