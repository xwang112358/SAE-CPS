# SAE-CPS: Interpretable Anomaly Detection in Cyber-Physical Systems via Sparse Autoencoder

Final Project for the Course CS8395 AI for Cyber-Physical Systems 

## Environment Setup

```bash
conda create --name sae-cps python=3.12
conda activate sae-cps
```

```bash
pip3 install torch==2.5.1 torchvision==0.16.1 torchaudio==2.5.1
pip install -r requirements.txt
```

## Dataset Preparation
If you want to rerun the data preprocessing, run `preprocess_dataset.ipynb`.


## Finetuning Models

```bash
python optimize_sae.py
python optimize_ae.py
python opyimize_topksae.py
```

## Visualize the results

Please run cells in `evaluate_methods.ipynb` to reproduce the results in the final report.




