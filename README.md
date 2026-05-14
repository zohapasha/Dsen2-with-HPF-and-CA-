# Enhanced DSen2: Sentinel-2 Super-Resolution via Channel Attention & HPF

PyTorch implementation of _Deep learning-based Sentinel-2 super-resolution via channel attention and high-frequency feature enhancement_ (Nguyen-Vi et al., 2025).

## Project Structure

```
config.py              # Hyperparameters, STAC settings, band definitions
data_loader.py         # STAC dataset, Wald's protocol, patch caching
model.py              # EnhancedDSen2 architecture (SE + HPF + residual blocks)
train.py              # Training loop, checkpoint management, CLI
eval.py               # Inference and visualization
diagnostics.py        # Per-sample and per-asset timing diagnostics
requirements.txt      # Dependencies
```

## Installation

```bash
pip install -r requirements.txt
```

**Optional** (for better AWS COG access):
```bash
pip install planetary-computer
```

## Quick Start: Synthetic Data

Test the pipeline without network access:

```bash
python train.py --synthetic --epochs 1 --batch-size 8 --num-workers 0
```

Expected output:
- ~0.55 training RMSE (synthetic data baseline)
- Completes in <1 minute on GPU

## Real Data: STAC + Sentinel-2

### Step 1: Prefetch Patches into Local Cache

First run prefetches patches from Earth Search STAC and stores them as compressed .npz files:

```bash
python train.py \
  --max-items 4 \
  --max-patches-per-item 16 \
  --max-total-patches 64 \
  --num-workers 0 \
  --prefetch-cache .\cache_prefetch \
  --prefetch-max 256 \
  --dry-run
```

This will:
- Query STAC for Sentinel-2 scenes (4 scenes)
- Build a patch index (64 samples total)
- Download and prefetch patches into `.\cache_prefetch` as .npz files
- Estimate epoch time and exit

Per-sample read time before prefetch: ~2s (HTTPS COG access + window reads)  
Per-sample read time after prefetch: ~0.008s (local .npz load)

### Step 2: Train with Cached Data

```bash
python train.py \
  --max-items 4 \
  --max-patches-per-item 16 \
  --max-total-patches 64 \
  --prefetch-cache .\cache_prefetch \
  --epochs 10 \
  --batch-size 8 \
  --num-workers 0
```

This will:
- Load dataset from index
- Read patches from cache (0.008s/sample)
- Train for 10 epochs
- Save checkpoints to `./checkpoints`

Expected training time: ~1.5 hours per epoch on NVIDIA GPU.

## Advanced Usage

### Diagnostics: Per-Asset Timing

Identify which STAC asset (band/COG) is slow:

```bash
python diagnostics.py \
  --max-items 4 \
  --max-patches-per-item 16 \
  --max-total-patches 64 \
  --num-samples 8
```

## Quick Visuals

If you only need a small number of visual samples to inspect model outputs, use `visualize.py`. It runs the model on a few indexed patches (defaults to a tiny index) and writes comparison images and per-band grayscale images.

Example (small, fast):
```bash
python visualize.py \
  --checkpoint ./checkpoints/enhanced_dsen2_best.pt \
  --samples 4 \
  --max-items 2 \
  --max-patches-per-item 8 \
  --max-total-patches 16 \
  --output-dir ./outputs/visuals
```

The script saves `sample_i_comparison.png` and a `sample_i_bands/` folder with per-band grayscale comparisons for each sample.

If you'd like CSV output of the per-band numeric metrics or internal SE attention weights, tell me and I'll add it.

Output includes:
- Scene ID and asset hrefs (http/s3 URLs)
- Per-sample read time
- Per-band read time (if verbose enabled)

### Dry-Run: Estimate Epoch Time

```bash
python train.py \
  --max-items 10 \
  --max-patches-per-item 32 \
  --max-total-patches 320 \
  --prefetch-cache .\cache_prefetch \
  --dry-run
```

Estimates full-epoch time based on measured per-sample latency.

### Command-Line Reference

#### Dataset Control
- `--max-items N`: Limit number of STAC scenes (default: unlimited)
- `--max-patches-per-item N`: Max patches per scene (default: 512)
- `--max-total-patches N`: Stop indexing after N total patches (default: unlimited)
- `--prefetch-cache DIR`: Directory for cached .npz patch files
- `--prefetch-max N`: Max patches to prefetch (default: 256)

#### Training
- `--epochs N`: Number of training epochs (default: 50)
- `--batch-size N`: Batch size (default: 8)
- `--num-workers N`: DataLoader workers (default: 4, use 0 on Windows)
- `--learning-rate F`: Adam learning rate (default: 1e-5)
- `--validation-split F`: Train/val split (default: 0.8)

#### Other
- `--synthetic`: Use synthetic data (no STAC required)
- `--synthetic-samples N`: Number of synthetic samples (default: 32)
- `--dry-run`: Index and estimate epoch time, then exit
- `--seed N`: Random seed (default: 42)

## Architecture

- **Input**: 10-band Sentinel-2 patch (4×10m guide + 6×20m upsampled)
- **Blocks**: Residual blocks with:
  - Conv path
  - Squeeze-Excitation channel attention
  - High-pass filter (Laplacian, normalized by 8) applied to block input
  - Concatenation + 1×1 fusion + residual add
- **Output**: 6-band enhanced 20m target bands
- **Loss**: MSE
- **Optimizer**: SGD (momentum=0.9), initial lr=1e-4, step-based scheduling (÷2 if val loss plateaus for 5 epochs), gradient clip norm=1.0

## Platform-Specific Notes

### Windows (Local Machine)

Always use `--num-workers 0` to avoid multiprocessing issues:

```bash
python train.py --num-workers 0 --batch-size 4 --epochs 1
```

### Kaggle Notebook

1. Mount workspace or upload files
2. Install dependencies:
   ```bash
   !pip install -r requirements.txt
   ```
3. For faster data access, reduce caps or use local cache:
   ```bash
   !python train.py --synthetic --epochs 1  # test pipeline
   !python train.py --max-items 2 --max-patches-per-item 8 --prefetch-cache ./cache --num-workers 2 --epochs 5
   ```

## Performance Tips

1. **First run is slow**: Initial STAC query and COG window reads take ~2s/sample. Use `--dry-run` to measure before committing to a long epoch.
2. **Use prefetch cache**: After first run, subsequent runs load from .npz (~0.008s/sample).
3. **Reduce caps for quick iteration**: `--max-items 2 --max-patches-per-item 4 --max-total-patches 8` (8 samples, ~15s to train).
4. **Check GPU memory**: Batch size 8 uses ~8GB. Reduce `--batch-size` if needed.

## Evaluation

Load a checkpoint and compute RMSE on validation set:

```bash
python eval.py \
  --checkpoint ./checkpoints/enhanced_dsen2_best.pt \
  --max-items 4 \
  --batch-size 8 \
  --num-workers 0
```

## Troubleshooting

### Slow Remote I/O (20-30s/sample)

**Cause**: boto3/GDAL EC2 metadata timeout on non-AWS machines.

**Solution**: The code sets `AWS_EC2_METADATA_DISABLED=true` automatically. If still slow, ensure `planetary-computer` is installed:
```bash
pip install planetary-computer
```

### Network Errors on Kaggle

**Cause**: Intermittent STAC/COG access failures.

**Solution**: Use `--synthetic` to validate pipeline, or prefetch locally then upload cache.

### Out of Memory (OOM)

**Cause**: Batch size too large for GPU.

**Solution**: Reduce `--batch-size` (e.g., 4 or 2) or use `--max-patches-per-item` to reduce dataset size.

### STAC Search Fails

**Cause**: Network connectivity or Earth Search server issues.

**Solution**: Test with `--synthetic`, or try later. Retry logic built-in (3 attempts with exponential backoff).

## References

- Nguyen-Vi et al. (2025): _Deep learning-based Sentinel-2 super-resolution via channel attention and high-frequency feature enhancement_
- Wald's Protocol: [Wald et al. (1997)](https://doi.org/10.1016/S0034-4257(97)00049-X)
- Earth Search STAC: https://earth-search.aws.element84.com/
- Planetary Computer: https://planetarycomputer.microsoft.com/

## License

See LICENSE file (or specify as appropriate).
