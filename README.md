# PALM-H3 (CPU/MPS Compatible Version)

A fork of the original PALM-H3 model modified to run on CPU and Apple Silicon (MPS).

## Setup Instructions

1. Clone this repository
2. Download pre-trained models from Zenodo: https://doi.org/10.5281/zenodo.7794583
3. Create the following directory structure in your PALM folder:

Result_covid_heavy/checkpoints/BERT-Pretrain-common-MAA-NGPUs/pretrained/
Result_covid_light/checkpoints/BERT-Pretrain-common-MAA-NGPUs/pretrained/
Result_seq2seq/checkpoints/ABAG-Finetuning-Seq2seq-Common/pretrained/
Result_cov_adbab/checkpoints/BERT-Finetunning-Antibody-Binding-common-abdab/pretrained/

4. Move the Zenodo model files to these directories:
- Move `Heavy_roformer/*` to `Result_covid_heavy/checkpoints/BERT-Pretrain-common-MAA-NGPUs/pretrained/`
- Move `Light_roformer/*` to `Result_covid_light/checkpoints/BERT-Pretrain-common-MAA-NGPUs/pretrained/`
- Move `PALM_seq2seq/*` to `Result_seq2seq/checkpoints/ABAG-Finetuning-Seq2seq-Common/pretrained/`
- Move `A2binder_affinity/*` to `Result_cov_adbab/checkpoints/BERT-Finetunning-Antibody-Binding-common-abdab/pretrained/`

## Changes from Original

- Modified `generate_antibody.py` to support CPU and MPS (Apple Silicon) devices
- Reduced beam search parameters for compatibility with devices without CUDA support
- Added device detection and appropriate fallback options

## Usage

conda create -n palm python=3.9
conda activate palm
pip install -r requirements.txt

python cur.py
