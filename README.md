# Continuous Text Autoencoder

We propose the **Continuous Text Autoencoder** (CTA), a framework that encodes tokenized text into compact **continuous latent sequences**.

CTA significantly reduces sequence length while preserving reconstruction quality. Empirically, CTA achieves near-lossless text reconstruction even at high compression ratios and substantially lowers computational costs for downstream modeling. These results suggest that continuous text representations offer a promising direction for scalable and efficient long-context language modeling.

## Overview

- **1D Convolutional Encoder-Decoder Architecture** with residual blocks
- **Variational Autoencoder Framework** with reparameterization trick
- **Distributed Data Parallel Training** with multi-GPU support
- **Muon Optimizer** (orthogonalized momentum) for improved convergence

### Model Architecture

- **Encoder**: Token embeddings → Conv1D residual blocks → Latent space (mu, logvar)
- **Decoder**: Latent vectors → Transposed Conv1D blocks → Token logits
- **Downsample Ratio**: 4x compression (2 downsampling layers)
- **Model Size**: ~25M parameters (configurable)

## Environment

**Requirements:**
- Python 3.13
- PyTorch 2.9

## Installation

```bash
pip install -r requirements.txt
```

## Dataset Setup

### 1. Download Dataset

Download Falcon RefinedWeb parquet files from HuggingFace:

https://huggingface.co/datasets/tiiuae/falcon-refinedweb/tree/main/data

Place the parquet files in `./datas`

### 2. Generate Test Sets

The pipeline creates two test sets: **Falcon-test** (real data) and **Random-test** (random tokens baseline).

```bash
# Step 1: Generate token statistics from training set
# This analyzes token frequencies across the entire training corpus
python tools/token_static.py

# Step 2: Filter and sample Falcon test set
# Creates test set with 512-token sequences, excluding rare tokens
python tools/filter_and_sample_texts.py

# Step 3: Generate random test set for baseline comparison
python tools/generate_random_texts.py
```

**Test Set Filtering Rules:**
- Sequences must be exactly 512 tokens (shorter discarded, longer randomly truncated)
- Only tokens that appear in training data are used
- Excludes tokens in the lowest 10% frequency (rare tokens)
- Default: 5,000 test samples

## Training

### Run Training

```bash
python train.py
```

### Configuration

Edit `config.py` to customize training parameters:

```python
dataset_path = 'datas/falcon_train'      # Training data path
tokenizer_path = 'datas/tokenizer'       # Tokenizer path
batch_size = 48                          # Batch size per GPU
learning_rate = 1e-4                     # Learning rate
max_steps = 80000                        # Total training steps
chunk_size = 128                         # Sequence length for training
model_save_path = './checkpoints/0'      # Checkpoint directory
log_dir = './runs/0'                     # TensorBoard logs
```

### Monitor Training

```bash
tensorboard --logdir runs/0
```

View training metrics:
- Reconstruction loss
- KL divergence loss
- Total loss
- Gradient norm
- Learning rate schedule

## Evaluation

### Run Accuracy Test

```bash
python test_accuracy.py
```

**Configure evaluation in test_accuracy.py:**
```python
MODEL_PATH = 'checkpoints/0/checkpoint_10000.pt'  # Trained checkpoint
DATA_PATH = 'datas/filtered_falcon_5000.jsonl'    # Test dataset
ERROR_OUTPUT_PATH = './reports/error_cases.jsonl' # Error report output
```

**Output:**
- Token-level accuracy: Percentage of correctly reconstructed tokens
- Sequence-level accuracy: Percentage of perfectly reconstructed sequences
- Error report saved to `reports/error_cases.jsonl`

### Visualize Results

Launch the interactive error visualization GUI:

```bash
python visualize_errors.py
```

Access the web interface at: `http://localhost:7860`

## Project Structure

```
.
├── config.py                    # Training configuration
├── train.py                     # Distributed training script
├── test_accuracy.py            # Model evaluation script
├── visualize_errors.py         # Gradio UI for error visualization
├── models/
│   ├── model.py                # TextVAE architecture (Encoder/Decoder)
│   ├── dataset.py              # Data loading and preprocessing
│   └── tokenizer.py            # Tokenizer wrapper
├── utils/
│   ├── muon.py                 # Muon optimizer implementation
│   ├── scheduler.py            # Learning rate schedulers
│   └── load.py                 # Checkpoint loading utilities
├── tools/
│   ├── token_static.py         # Token frequency analysis
│   ├── filter_and_sample_texts.py  # Test set generation
│   └── generate_random_texts.py    # Random baseline generation
├── datas/                      # Dataset directory
│   ├── tokenizer/              # Pretrained tokenizer
│   └── falcon_train/           # Training parquet files
├── checkpoints/                # Saved model checkpoints
├── runs/                       # TensorBoard logs
└── reports/                    # Evaluation reports
```
