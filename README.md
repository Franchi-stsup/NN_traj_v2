# Circle Trajectory Neural Network

A modular neural network implementation for reconstructing circle trajectories from time vectors.

## Project Structure

```
NN_traj_v0/
├── network.py          # Model architectures and dataset classes
├── train.py            # Training logic, loss functions, model save/load
├── plots.py            # Visualization and plotting functions
├── run.py              # Main pipeline runner with argument parser
├── models/             # Directory for CircleCNN models
├── models_cnn02/       # Directory for CircleCNN_02 models
├── plots/              # Directory for CircleCNN plots
└── plots_cnn02/        # Directory for CircleCNN_02 plots
```

## Requirements

- PyTorch
- NumPy
- Matplotlib
- tqdm

## Usage

### Quick Start - Run Complete Pipeline
```bash
# Activate your conda environment
conda activate bart-env

# Run the complete pipeline (train + evaluate + plot + demo)
python run.py --mode all
```

### Modular Execution

#### Training Only
```bash
python run.py --mode train --num_epochs 100 --batch_size 32 --learning_rate 1e-3
```

#### Evaluation Only (requires pre-trained model)
```bash
python run.py --mode evaluate --model_name circle_cnn_model.pth
```

#### Plotting/Analysis Only
```bash
python run.py --mode plot --model_name circle_cnn_model.pth
```

#### Demo with Pre-trained Model
```bash
python run.py --mode demo --model_name circle_cnn_model.pth
```

## Neural Network Architectures

The pipeline supports two different CNN architectures:

### CircleCNN (Standard Architecture)
- **Layers**: 4 convolutional layers
- **Features**: Basic 1D CNN with batch normalization and ReLU activations
- **Use case**: Faster training, good for basic trajectory reconstruction
- **Usage**: Default architecture (no additional flags needed)

### CircleCNN_02 (Deeper Architecture) 
- **Layers**: 8 convolutional layers with residual connections
- **Features**: Enhanced architecture with skip connections and multi-scale processing
- **Use case**: Better accuracy for complex trajectories, longer training time
- **Usage**: Add `--use_cnn_02` flag
- **Model Storage**: Automatically saves to `models_cnn02/` directory
- **Plot Storage**: Automatically saves to `plots_cnn02/` directory

### Architecture Selection Examples
```bash
# Use standard CircleCNN
python run.py --mode train --num_epochs 100

# Use deeper CircleCNN_02
python run.py --mode train --use_cnn_02 --num_epochs 200

# Load and continue training a CNN_02 model
python run.py --mode train --pretrained_model your_model.pth --use_cnn_02
```

## Pretrained Model Support

The pipeline supports loading and continuing training from pretrained models:

### Loading Pretrained Models
- Use `--pretrained_model` to specify the model file
- Automatic architecture detection based on `--use_cnn_02` flag
- Automatic directory resolution (models/ or models_cnn02/)
- PyTorch version compatibility handling

### Pretrained Model Examples
```bash
# Continue training from a standard CNN model
python run.py --mode train --pretrained_model 20250812_115047_ep500_lr1e-03_bs64_r1.0.pth --num_epochs 100

# Continue training from a CNN_02 model  
python run.py --mode train --pretrained_model 20250812_115047_ep500_lr1e-03_bs64_r1.0_cnn02.pth --use_cnn_02 --num_epochs 100

# Evaluate a pretrained model
python run.py --mode evaluate --pretrained_model your_model.pth --use_cnn_02
```

### Keyboard Interruption During Training
- Type 'q' and press Enter to gracefully stop training and save the model
- Works on both Windows and Linux/Unix systems
- Interrupted models are saved with timestamp and interrupted status

### Configuration Options

#### Model Parameters
- `--input_length`: Input sequence length (default: 128)
- `--output_length`: Output sequence length (default: 128)
- `--hidden_channels`: CNN hidden channels (default: 64)
- `--use_cnn_02`: Use deeper CircleCNN_02 architecture instead of CircleCNN (default: False)

#### Dataset Parameters
- `--train_samples`: Number of training samples (default: 1000)
- `--val_samples`: Number of validation samples (default: 200)
- `--radius`: Circle radius (default: 1.0)
- `--noise_level`: Input signal noise level (default: 0.01)

#### Training Parameters
- `--batch_size`: Training batch size (default: 32)
- `--num_epochs`: Number of training epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-3)
- `--use_smooth_loss`: Use custom smooth loss function (default: True)
- `--no_smooth_loss`: Disable smooth loss function
- `--mse_weight`: MSE weight in smooth loss (default: 1.0)
- `--first_deriv_weight`: First derivative weight in smooth loss (default: 0.001)
- `--second_deriv_weight`: Second derivative weight in smooth loss (default: 0.000)
- `--use_scheduler`: Use learning rate scheduler (default: False)

#### File Paths
- `--model_name`: Model filename (default: circle_cnn_model.pth)
- `--pretrained_model`: Path to pretrained model to continue training from (optional)
- `--model_dir`: Model directory (default: models)
- `--plot_dir`: Plot directory (default: plots)

### Example Commands

```bash
# Train with custom parameters
python run.py --mode train --num_epochs 150 --batch_size 64 --radius 2.0

# Train using the deeper CNN_02 architecture
python run.py --mode train --use_cnn_02 --num_epochs 200 --batch_size 32

# Continue training from a pretrained model
python run.py --mode train --pretrained_model 20250812_115047_ep500_lr1e-03_bs64_r1.0_cnn02.pth --use_cnn_02 --num_epochs 100

# Train with custom smooth loss weights
python run.py --mode train --mse_weight 1.0 --first_deriv_weight 0.0005 --second_deriv_weight 1e-6

# Evaluate with different validation set
python run.py --mode evaluate --val_samples 500 --radius 2.0

# Evaluate CNN_02 model
python run.py --mode evaluate --use_cnn_02 --model_name your_cnn02_model.pth

# Run complete pipeline with custom settings
python run.py --mode all --num_epochs 80 --batch_size 16 --radius 1.5
```

## Module Descriptions

### network.py
- `CircleDataset`: Dataset class for generating circle trajectory data
- `CircleCNN`: Standard 1D CNN model (4 layers) for trajectory reconstruction
- `CircleCNN_02`: Deeper 1D CNN model (8 layers) with residual connections for enhanced performance
- Device management utilities

### train.py
- `SmoothLoss`: Custom loss function with derivative smoothness penalties
- `train_model()`: Main training loop with validation and keyboard interruption support
- Model save/load functionality with structured naming and PyTorch version compatibility
- `load_pretrained_model()`: Load pretrained models with automatic architecture detection
- Inference functions with batch and single sample support

### plots.py
- `plot_training_curves()`: Training/validation loss visualization
- `visualize_results()`: Time series plotting of kx(t) and ky(t)
- `plot_circle()`: 2D circle visualization
- `plot_derivatives()`: Derivative analysis plots
- `analyze_model_predictions()`: Complete prediction analysis

### run.py
- Command-line argument parsing with dual architecture support
- Modular execution modes with automatic directory management
- Complete pipeline orchestration with interruption handling
- Configuration management and pretrained model integration

## Environment Setup

Make sure your conda environment has the required packages:

```bash
conda activate bart-env
conda install pytorch matplotlib numpy tqdm
```

## Output

The pipeline generates:
- **Models**: 
  - CircleCNN models saved in `models/` directory as `.pth` files
  - CircleCNN_02 models saved in `models_cnn02/` directory as `.pth` files
  - Structured naming with timestamps and training parameters
- **Plots**: Various visualizations with automatic directory selection:
  - CircleCNN plots in `plots/` directory
  - CircleCNN_02 plots in `plots_cnn02/` directory
  - Training loss curves
  - Time series plots (kx, ky vs time)
  - 2D circle trajectories
  - Derivative analysis plots
  - Velocity and acceleration magnitude plots
