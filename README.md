# gmm-guided synthetic transfer data

for ICPE'26 or eScience'26

## GPU-accelerated GMM training

You can train the Gaussian mixture on a CUDA-capable GPU by installing a PyTorch build with CUDA support and enabling the GPU trainer:

```bash
python gmm.py og-transfer.csv --output output/output.csv --use-gpu --gpu-device cuda:0
```

The GPU path uses custom PyTorch implementations (`fit_best_gmm_gpu`, `TorchPowerTransformer`) so preprocessing (Box-Cox) and GMM fitting both run on the selected CUDA device while keeping the rest of the synthetic generation pipeline unchanged.

Optional: install `tqdm` to see progress bars for the component search and GPU EM iterations (`pip install tqdm`).
