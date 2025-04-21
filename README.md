# SAE-CPS
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

## Finetuning Models

```bash
python optimize_sae.py
```

## Visualize the results

Please run cells in `evaluate_methods.ipynb` to reproduce the results in the final report.




