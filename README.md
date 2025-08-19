# Circle Trajectory Neural Network

A modular neural network implementation for reconstructing circle trajectories from time vectors with support for multiple neural network architectures.

## Project Structure

```
NN_traj_v2/
├── src/
│   ├── network.py          # Model architectures and dataset classes
│   ├── train.py            # Training logic, loss functions, model save/load
│   ├── plots.py            # Visualization and plotting functions
│   └── network_utils.py    # Trajectory utilities and transformations
├── run.py                  # Main pipeline runner with argument parser
├── models/                 # Directory for CircleCNN models
├── models_cnn02/          # Directory for CircleCNN_02 models
├── models_unet/           # Directory for TrajectoryUNet models
├── models_freq_net/       # Directory for FrequencyAwareNet models
├── plots/                 # Directory for CircleCNN plots
├── plots_cnn02/           # Directory for CircleCNN_02 plots
├── plots_unet/            # Directory for TrajectoryUNet plots
├── plots_freq_net/        # Directory for FrequencyAwareNet plots
└── test/                  # Test scripts and utilities
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

The pipeline supports four different neural network architectures for trajectory reconstruction:

### CircleCNN (Standard Architecture)
- **Layers**: 4 convolutional layers
- **Features**: Basic 1D CNN with batch normalization and ReLU activations
- **Use case**: Faster training, good for basic trajectory reconstruction
- **Usage**: Default architecture (no additional flags needed)
- **Storage**: Models in `models/`, plots in `plots/`

### CircleCNN_02 (Deeper Architecture) 
- **Layers**: 8 convolutional layers with residual connections
- **Features**: Enhanced architecture with skip connections and multi-scale processing
- **Use case**: Better accuracy for complex trajectories, longer training time
- **Usage**: Add `--use_cnn_02` flag
- **Storage**: Models in `models_cnn02/`, plots in `plots_cnn02/`

### TrajectoryUNet (U-Net Architecture)
- **Features**: U-Net inspired encoder-decoder architecture with skip connections
- **Use case**: Advanced trajectory reconstruction with spatial context preservation
- **Usage**: Add `--test_net unet` flag
- **Storage**: Models in `models_unet/`, plots in `plots_unet/`

### FrequencyAwareNet (Frequency Domain Processing)
- **Features**: Specialized architecture for frequency-domain trajectory processing
- **Use case**: Optimized for frequency-based trajectory features
- **Usage**: Add `--test_net freq_net` flag  
- **Storage**: Models in `models_freq_net/`, plots in `plots_freq_net/`

### Architecture Selection Examples
```bash
# Use standard CircleCNN (default)
python run.py --mode train --num_epochs 100

# Use deeper CircleCNN_02
python run.py --mode train --use_cnn_02 --num_epochs 200

# Use TrajectoryUNet
python run.py --mode train --test_net unet --num_epochs 150

# Use FrequencyAwareNet
python run.py --mode train --test_net freq_net --num_epochs 150

# Load and continue training with specific architecture
python run.py --mode train --pretrained_model your_model.pth --test_net unet
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

# Continue training from a TrajectoryUNet model
python run.py --mode train --pretrained_model 20250819_120000_ep300_lr1e-04_bs32_unet.pth --test_net unet --num_epochs 100

# Continue training from a FrequencyAwareNet model
python run.py --mode train --pretrained_model 20250819_120000_ep300_lr1e-04_bs32_freq_net.pth --test_net freq_net --num_epochs 100

# Evaluate a pretrained model with correct architecture
python run.py --mode evaluate --pretrained_model your_model.pth --use_cnn_02
python run.py --mode evaluate --pretrained_model your_model.pth --test_net unet
python run.py --mode evaluate --pretrained_model your_model.pth --test_net freq_net
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
- `--test_net`: Use alternative network architecture: 'unet' (TrajectoryUNet) or 'freq_net' (FrequencyAwareNet) (default: None)

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

# Train using different architectures
python run.py --mode train --use_cnn_02 --num_epochs 200 --batch_size 32
python run.py --mode train --test_net unet --num_epochs 150 --batch_size 16
python run.py --mode train --test_net freq_net --num_epochs 200 --batch_size 32

# Continue training from pretrained models with correct architecture
python run.py --mode train --pretrained_model 20250812_115047_ep500_lr1e-03_bs64_r1.0_cnn02.pth --use_cnn_02 --num_epochs 100
python run.py --mode train --pretrained_model your_unet_model.pth --test_net unet --num_epochs 50
python run.py --mode train --pretrained_model your_freq_model.pth --test_net freq_net --num_epochs 50

# Train with custom smooth loss weights
python run.py --mode train --mse_weight 1.0 --first_deriv_weight 0.0005 --second_deriv_weight 1e-6

# Evaluate with different architectures
python run.py --mode evaluate --val_samples 500 --radius 2.0
python run.py --mode evaluate --use_cnn_02 --model_name your_cnn02_model.pth
python run.py --mode evaluate --test_net unet --model_name your_unet_model.pth
python run.py --mode evaluate --test_net freq_net --model_name your_freq_model.pth

# Run complete pipeline with different architectures
python run.py --mode all --num_epochs 80 --batch_size 16 --radius 1.5
python run.py --mode all --test_net unet --num_epochs 100 --batch_size 8
python run.py --mode all --test_net freq_net --num_epochs 120 --batch_size 16
```

## Module Descriptions

### src/network.py
- `CircleDataset`: Dataset class for generating circle trajectory data
- `CircleCNN`: Standard 1D CNN model (4 layers) for trajectory reconstruction
- `CircleCNN_02`: Deeper 1D CNN model (8 layers) with residual connections for enhanced performance
- `TrajectoryUNet`: U-Net inspired encoder-decoder architecture with skip connections
- `FrequencyAwareNet`: Specialized architecture for frequency-domain trajectory processing
- Device management utilities

### src/train.py
- `SmoothLoss`: Custom loss function with derivative smoothness penalties
- `train_model()`: Main training loop with validation and keyboard interruption support
- Model save/load functionality with structured naming and PyTorch version compatibility
- `load_pretrained_model()`: Load pretrained models with automatic architecture detection
- Inference functions with batch and single sample support

### src/plots.py
- `plot_training_curves()`: Training/validation loss visualization
- `visualize_results()`: Time series plotting of kx(t) and ky(t)
- `plot_circle()`: 2D circle visualization
- `plot_derivatives()`: Derivative analysis plots
- `analyze_model_predictions()`: Complete prediction analysis

### src/network_utils.py
- Trajectory transformation utilities (shift, rotate, complex conversion)
- Demonstration functions for trajectory processing
- Integration with BART reconstruction pipeline

### run.py
- Command-line argument parsing with multi-architecture support
- Modular execution modes with automatic directory management
- Complete pipeline orchestration with interruption handling  
- Configuration management and pretrained model integration with architecture detection

## Environment Setup

Make sure your conda environment has the required packages:

```bash
conda activate bart-env
conda install pytorch matplotlib numpy tqdm
```

## Output

The pipeline generates organized outputs with automatic directory management:

### Models
- **CircleCNN models**: Saved in `models/` directory as `.pth` files
- **CircleCNN_02 models**: Saved in `models_cnn02/` directory as `.pth` files  
- **TrajectoryUNet models**: Saved in `models_unet/` directory as `.pth` files
- **FrequencyAwareNet models**: Saved in `models_freq_net/` directory as `.pth` files
- **Naming**: Structured filenames with timestamps and training parameters

### Plots
Various visualizations with automatic directory selection based on architecture:
- **CircleCNN plots**: `plots/` directory
- **CircleCNN_02 plots**: `plots_cnn02/` directory
- **TrajectoryUNet plots**: `plots_unet/` directory  
- **FrequencyAwareNet plots**: `plots_freq_net/` directory
- **Content**: Training loss curves, time series plots (kx, ky vs time), 2D circle trajectories, derivative analysis plots, velocity and acceleration magnitude plots


## Architecture Selection Guide

### When to Use Each Architecture

| Architecture | Best For | Training Time | Memory Usage | Complexity |
|-------------|----------|---------------|--------------|------------|
| **CircleCNN** | Simple trajectories, fast prototyping | Fast | Low | Basic |
| **CircleCNN_02** | Complex trajectories, higher accuracy | Medium | Medium | Enhanced |
| **TrajectoryUNet** | Spatial context preservation | Slow | High | Advanced |
| **FrequencyAwareNet** | Frequency domain features | Medium | Medium | Specialized |

### Performance Characteristics

- **CircleCNN**: 4 layers, ~50K parameters, best for quick experiments
- **CircleCNN_02**: 8 layers with residual connections, ~200K parameters, better accuracy
- **TrajectoryUNet**: U-Net architecture with skip connections, ~500K parameters, advanced reconstruction
- **FrequencyAwareNet**: Frequency-domain processing, ~300K parameters, specialized for frequency features

### Migration Between Architectures

```bash
# Start with simple architecture for prototyping
python run.py --mode train --num_epochs 50 --batch_size 32

# Scale up to more complex architecture for production
python run.py --mode train --test_net unet --num_epochs 200 --batch_size 16

# Compare architectures
python run.py --mode evaluate --model_name simple_model.pth
python run.py --mode evaluate --test_net unet --model_name unet_model.pth
```

## Advanced Features

### BART Integration
The pipeline includes integration with BART (Berkeley Advanced Reconstruction Toolbox) for MRI reconstruction:

```
CNN Weights → Predicted Trajectories → K-space Sampling → Image Reconstruction → Image Quality Metrics
     ↑                                                                                    ↓
     ←←←←←←←←←←←← Gradients Flow Backward Through Entire Pipeline ←←←←←←←←←←←←←←←←←

     Gradient Flow Breakdown:
SSIM/MSE gradients computed from image comparison
Flow through BART reconstruction (with pseudo-gradient connection)
Flow through k-space interpolation (differentiable operations)
Flow through trajectory transformations (shift, rotate, complex conversion)
Flow through derivative computations (finite differences)
Finally reach CNN parameters for weight updates
```

### Why This Works
The CNN learns to generate trajectories that:
- Reconstruct high-quality images (SSIM/MSE gradients)
- Are physically smooth (derivative penalties)
- Respect hardware limits (gradient/slew constraints)
- Sample k-space effectively (through the BART pipeline)

This creates a physics-informed neural network that understands both the CNN's trajectory generation capabilities and the downstream MRI reconstruction requirements.

### Gradient Flow Logic
```
Loss → ∂Loss/∂recon_image → ∂recon_image/∂kspace_data → ∂kspace_data/∂trajectory → ∂trajectory/∂CNN_params

∂Loss/∂kspace_data = (∂Loss/∂recon_image) × (∂recon_image/∂kspace_data)
                   =  grad_output_image  ×     [BART adjoint]
```