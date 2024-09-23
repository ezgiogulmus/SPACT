# SPACT for Survival Prediction

This repository contains the SPACT model for discrete-time survival prediction tasks using whole-slide images and genetic data.

## Installation

Tested on:
- Ubuntu 22.04
- Nvidia GeForce RTX 4090
- Python 3.10.14
- PyTorch 2.3.0

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/ezgiogulmus/SPACT.git
cd SPACT
```

Create a Conda environment and install the required packages:

```bash
conda env create -n spact python=3.10 -y
conda activate spact
pip install --upgrade pip 
pip install -e .
```

## Usage

First, extract patch coordinates and patch-level features using the CLAM library available at [CLAM GitHub](https://github.com/Mahmoodlab/CLAM). Then, run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --coords_dir path/to/coords/dir --feats_dir path/to/feats/dir --data_name tcga_ov_os --model_type spact --omics rna,dna,cnv,pro --selected_features --slide_aggregation early --pooling avgpoolto1 --target_nb_patches 5 15 25
```

- `coords_dir`: Directory where the patch coordinates are stored (h5 files).
- `feats_dir`: Directory where the patch features are stored (pt files).

See [scripts](./scripts/) for more commands.

## Acknowledgements

This code is adapted from the [PORPOISE](https://github.com/mahmoodlab/PORPOISE) model.

## License

This repository is licensed under the [GPLv3 License](./LICENSE).
