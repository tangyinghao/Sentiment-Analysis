# BERT Sentiment Analysis - Training Guide

## Overview

This project implements a BERT-based sentiment analysis classifier for product reviews. The training script (`train_bert.py`) is designed to run on a remote GPU, while the analysis notebook (`analysis.ipynb`) helps visualize and compare results locally.

## Project Status

### ✅ Completed
- **Step 1: Data Loading** - `train.json` and `test.json` loaded successfully
- **Step 2: Data Processing** - Text cleaning (lowercase, stopword removal, lemmatization) and TF-IDF vectorization completed in `main.ipynb`

### 🔄 Remaining Tasks
- **Step 3: Model Selection** - BERT model implementation (ready to train)
- **Step 4: Training** - Hyperparameter tuning and model training
- **Step 5: Prediction** - Generate `submission.csv` file

## Files Created

1. **`train_bert.py`** - Main training script for remote GPU
2. **`analysis.ipynb`** - Results analysis and visualization notebook
3. **`requirements.txt`** - Python dependencies

## Setup

### 1. Install Dependencies

On your remote GPU machine:
```bash
pip install -r requirements.txt
```

### 2. Transfer Files to Remote GPU

Make sure these files are available on your remote GPU:
- `train_bert.py`
- `train.json`
- `test.json`
- `requirements.txt`

## Usage

### Basic Training

Train with default hyperparameters:
```bash
python train_bert.py --generate_submission
```

### Hyperparameter Search (Recommended)

Test multiple configurations and automatically select the best model:
```bash
python train_bert.py --hyperparameter_search --generate_submission --save_model
```

This will:
- Test multiple hyperparameter configurations (default: 4 different configs)
- Automatically select the best model based on validation F1 score
- Generate submission.csv using the best model
- Save comparison results to `hyperparameter_comparison_TIMESTAMP.csv`

### Custom Hyperparameter Search

Use a custom configuration file:
```bash
python train_bert.py --hyperparameter_search --config_file experiment_configs.json --generate_submission
```

### Custom Hyperparameters

Train with specific hyperparameters:
```bash
python train_bert.py \
    --model_name bert-base-uncased \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --num_epochs 3 \
    --max_length 512 \
    --val_split 0.2 \
    --generate_submission \
    --save_model
```

### Hyperparameter Tuning

Run multiple experiments with different configurations:

```bash
# Experiment 1: Smaller learning rate
python train_bert.py --learning_rate 1e-5 --num_epochs 3 --generate_submission

# Experiment 2: Larger batch size
python train_bert.py --batch_size 32 --learning_rate 2e-5 --num_epochs 3 --generate_submission

# Experiment 3: More epochs
python train_bert.py --num_epochs 5 --learning_rate 2e-5 --generate_submission

# Experiment 4: Different BERT model
python train_bert.py --model_name distilbert-base-uncased --batch_size 16 --generate_submission
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `bert-base-uncased` | BERT model to use (e.g., `bert-base-uncased`, `distilbert-base-uncased`) |
| `--batch_size` | `16` | Training batch size |
| `--learning_rate` | `2e-5` | Learning rate for optimizer |
| `--num_epochs` | `3` | Number of training epochs |
| `--max_length` | `512` | Maximum sequence length |
| `--val_split` | `0.2` | Validation split ratio |
| `--warmup_steps` | `0` | Warmup steps for learning rate scheduler |
| `--weight_decay` | `0.01` | Weight decay for regularization |
| `--train_path` | `train.json` | Path to training data |
| `--test_path` | `test.json` | Path to test data |
| `--output_dir` | `results` | Directory to save results |
| `--save_model` | `False` | Save the trained model |
| `--generate_submission` | `False` | Generate submission.csv file |
| `--hyperparameter_search` | `False` | Run hyperparameter search across multiple configurations |
| `--config_file` | `None` | JSON file with experiment configurations for hyperparameter search |

## Output Files

The training script generates the following files in the `results/` directory:

1. **`history_TIMESTAMP.json`** - Detailed training history (loss, accuracy, F1, etc. per epoch)
2. **`summary_TIMESTAMP.json`** - Summary metrics and configuration
3. **`submission.csv`** - Predictions for test set (if `--generate_submission` is used)
4. **`model_TIMESTAMP/`** - Saved model files (if `--save_model` is used)
5. **`hyperparameter_comparison_TIMESTAMP.csv`** - Comparison of all experiments (if `--hyperparameter_search` is used)

## Analysis

After training, transfer the `results/` directory back to your local machine and open `analysis.ipynb` to:

1. Compare different model configurations
2. Visualize training curves (loss, accuracy, F1 score)
3. Analyze hyperparameter effects
4. Review submission file predictions
5. Generate summary statistics

## Example Workflow

### Option 1: Hyperparameter Search (Recommended)

1. **On Remote GPU:**
   ```bash
   # Run hyperparameter search - tests multiple configs and picks best
   python train_bert.py --hyperparameter_search --generate_submission --save_model
   ```

2. **Transfer results back to local machine:**
   ```bash
   scp -r user@remote-gpu:~/Sentiment-Analysis/results ./
   ```

3. **On Local Machine:**
   - Open `analysis.ipynb`
   - Run all cells to visualize and compare results
   - Review `submission.csv` (generated with best model)

### Option 2: Manual Multiple Runs

1. **On Remote GPU:**
   ```bash
   # Run multiple experiments manually
   python train_bert.py --batch_size 16 --learning_rate 2e-5 --num_epochs 3 --generate_submission
   python train_bert.py --batch_size 32 --learning_rate 2e-5 --num_epochs 3 --generate_submission
   python train_bert.py --batch_size 16 --learning_rate 1e-5 --num_epochs 5 --generate_submission
   ```

2. **Transfer results back to local machine:**
   ```bash
   scp -r user@remote-gpu:~/Sentiment-Analysis/results ./
   ```

3. **On Local Machine:**
   - Open `analysis.ipynb`
   - Run all cells to visualize and compare results
   - Identify best configuration
   - Review `submission.csv` if needed

## Tips

- Start with default parameters to establish a baseline
- Use `--save_model` if you want to reuse the trained model later
- Monitor GPU memory usage - reduce `batch_size` if you run out of memory
- For faster training, try `distilbert-base-uncased` (smaller, faster model)
- Increase `num_epochs` if validation metrics are still improving
- Use `--warmup_steps` (e.g., 100-500) for better learning rate scheduling

## Model Selection Options

You can experiment with different BERT models:
- `bert-base-uncased` - Standard BERT (default)
- `distilbert-base-uncased` - Smaller, faster version
- `bert-large-uncased` - Larger, more powerful (requires more GPU memory)
- `roberta-base` - RoBERTa model (may require slight code modifications)

## Notes

- The script automatically uses cleaned reviews from `main.ipynb` if available
- Validation split is stratified to maintain class distribution
- Best model (by validation F1 score) is saved automatically
- All metrics are logged per epoch for detailed analysis

