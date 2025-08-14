"""
Main Pipeline Runner
Modular execution script for training, evaluation, and visualization of circle trajectory models.
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
import time

# Local imports
from src.network import CircleCNN, CircleCNN_02, CircleDataset, print_device_info
from src.train import train_model, save_model, load_pretrained_model, demo_pretrained_usage
from src.plots import (plot_training_curves, visualize_results, plot_circle, 
                   analyze_model_predictions, plot_pretrained_demo)
from src.network_utils import shift_trajectory, rotate_traj, complex_traj, demo_trajectory_utilities
from src.bart_config import PipelineConfig, create_default_config
from src.bart_interpolation_fct import fast_kspace_interpolation_v3
from src.bart_utils import run_bart_nufft, build_para, rescale_recon_img, plot_comparison, downsample_image


RES = 50   # Resolution in pixels
FOV = 224  # Field of View in mm
KMAX_RES = RES / (FOV * 1e-3) / 2.0  # kmax in 1/mm, assuming FOV is in mm

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Circle Trajectory Neural Network Pipeline')
    
    # Main execution modes
    parser.add_argument('--mode', type=str, default='all', 
                       choices=['train', 'evaluate', 'plot', 'demo', 'all'],
                       help='Execution mode: train, evaluate, plot, demo, or all')
    
    # Model parameters
    parser.add_argument('--input_length', type=int, default=128,
                       help='Input sequence length (default: 128)')
    parser.add_argument('--output_length', type=int, default=128,
                       help='Output sequence length (default: 128)')
    parser.add_argument('--hidden_channels', type=int, default=64,
                       help='Number of hidden channels in CNN (default: 64)')
    parser.add_argument('--use_cnn_02', action='store_true',
                       help='Use deeper CircleCNN_02 architecture instead of CircleCNN (default: False)')
    
    # Dataset parameters
    parser.add_argument('--train_samples', type=int, default=1000,
                       help='Number of training samples (default: 1000)')
    parser.add_argument('--val_samples', type=int, default=200,
                       help='Number of validation samples (default: 200)')
    parser.add_argument('--radius', type=float, default=KMAX_RES / 2,
                       help='Circle radius (default: 1.0)')
    parser.add_argument('--noise_level', type=float, default=0,
                       help='Noise level for input signals (default: 0.0)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    parser.add_argument('--use_scheduler', action='store_true',
                       help='Use learning rate scheduler (default: False)')
    parser.add_argument('--use_smooth_loss', action='store_true', default=True,
                       help='Use smooth loss function (default: True)')
    parser.add_argument('--no_smooth_loss', action='store_true',
                       help='Disable smooth loss function')
    
    # Smooth loss parameters (only used if use_smooth_loss is True)
    parser.add_argument('--mse_weight', type=float, default=1.0,
                       help='MSE weight in smooth loss (default: 1.0)')
    parser.add_argument('--first_deriv_weight', type=float, default=0.001,
                       help='First derivative weight in smooth loss (default: 0.001)')
    parser.add_argument('--second_deriv_weight', type=float, default=0.0005,
                       help='Second derivative weight in smooth loss (default: 0.0005)')
    
    # File paths
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    parser.add_argument('--model_name', type=str, default=f'circle_cnn_model_{timestamp}.pth',
                       help='Model filename (default: circle_cnn_model_<timestamp>.pth)')
    parser.add_argument('--pretrained_model', type=str, default=None,
                       help='Path to pretrained model to continue training from (optional)')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save/load models (default: models)')
    parser.add_argument('--plot_dir', type=str, default='plots',
                       help='Directory to save plots (default: plots)')
    
    # Plotting parameters
    parser.add_argument('--num_plot_samples', type=int, default=3,
                       help='Number of samples to plot (default: 3)')
    
    return parser.parse_args()

def get_model_subdirs(use_cnn_02):
    """Get appropriate subdirectories based on model architecture"""
    if use_cnn_02:
        return 'models_cnn02', 'plots_cnn02'
    else:
        return 'models', 'plots'

def run_training(args):
    """Run the training pipeline"""
    print("=== TRAINING MODE ===")
    print_device_info()
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = CircleDataset(
        num_samples=args.train_samples,
        input_length=args.input_length,
        output_length=args.output_length,
        radius=args.radius,
        noise_level=args.noise_level
    )
    
    val_dataset = CircleDataset(
        num_samples=args.val_samples,
        input_length=args.input_length,
        output_length=args.output_length,
        radius=args.radius,
        noise_level=args.noise_level
    )
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create or load model
    if args.pretrained_model:
        print(f"Loading pretrained model: {args.pretrained_model}")
        model = load_pretrained_model(args.pretrained_model, 
                                    args.input_length, 
                                    args.output_length,
                                    load_dir=args.model_dir,
                                    use_cnn_02=args.use_cnn_02)
        if model is None:
            print("Failed to load pretrained model, creating new model...")
            if args.use_cnn_02:
                print("Creating new CircleCNN_02 (deeper) model...")
                model = CircleCNN_02(
                    input_length=args.input_length,
                    output_length=args.output_length,
                    base_channels=args.hidden_channels
                )
            else:
                model = CircleCNN(
                    input_length=args.input_length,
                    output_length=args.output_length,
                    hidden_channels=args.hidden_channels
                )
        else:
            print("Successfully loaded pretrained model for retraining")
            # Set model back to training mode (load_pretrained_model sets it to eval)
            model.train()
    else:
        if args.use_cnn_02:
            print("Creating new CircleCNN_02 (deeper) model...")
            model = CircleCNN_02(
                input_length=args.input_length,
                output_length=args.output_length,
                base_channels=args.hidden_channels
            )
        else:
            print("Creating new CircleCNN model...")
            model = CircleCNN(
                input_length=args.input_length,
                output_length=args.output_length,
                hidden_channels=args.hidden_channels
            )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Train model
    print("Starting training...")
    use_smooth = args.use_smooth_loss and not args.no_smooth_loss
    
    # Prepare save parameters for emergency save
    save_params = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        'radius': args.radius,
        'use_scheduler': args.use_scheduler,
        'use_smooth_loss': use_smooth,
        'use_cnn_02': args.use_cnn_02
    }
    
    # Add smooth loss weights if used
    if use_smooth:
        save_params.update({
            'mse_weight': args.mse_weight,
            'first_deriv_weight': args.first_deriv_weight,
            'second_deriv_weight': args.second_deriv_weight
        })
    
    # Call train_model with the new signature
    result = train_model(
        model, train_loader, val_loader,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        use_smooth_loss=use_smooth,
        model_save_dir=args.model_dir,
        use_scheduler=args.use_scheduler,
        mse_weight=args.mse_weight if use_smooth else 1.0,
        first_deriv_weight=args.first_deriv_weight if use_smooth else 0.001,
        second_deriv_weight=args.second_deriv_weight if use_smooth else 0.0005,
        save_params=save_params
    )
    
    # Handle the result (now includes interruption flag)
    if len(result) == 3:
        train_losses, val_losses, was_interrupted = result
    else:
        # Backward compatibility
        train_losses, val_losses = result
        was_interrupted = False
    
    # If training was interrupted, stop the pipeline
    if was_interrupted:
        print("\n=== PIPELINE STOPPED DUE TO USER INTERRUPTION ===")
        return None, None, None, None, None
    
    # Plot training curves with parameters for folder naming
    training_params = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        # 'radius': args.radius,  # Removed radius from folder names
        'use_scheduler': args.use_scheduler,
        'use_smooth_loss': args.use_smooth_loss and not args.no_smooth_loss
    }
    
    # Add smooth loss weights to folder name if smooth loss is used
    if args.use_smooth_loss and not args.no_smooth_loss:
        training_params.update({
            'mse_weight': args.mse_weight,
            'first_deriv_weight': args.first_deriv_weight,
            'second_deriv_weight': args.second_deriv_weight
        })
    
    run_folder = plot_training_curves(train_losses, val_losses, save_path=args.plot_dir, **training_params)
    
    # Save model with structured filename including parameters
    model_params = {
        'num_epochs': args.num_epochs,
        'learning_rate': args.learning_rate,
        'batch_size': args.batch_size,
        # 'radius': args.radius,  # Removed radius from model names
        'use_scheduler': args.use_scheduler,
        'use_smooth_loss': args.use_smooth_loss and not args.no_smooth_loss,
        'use_cnn_02': args.use_cnn_02
    }
    
    # Add smooth loss weights if used
    if args.use_smooth_loss and not args.no_smooth_loss:
        model_params.update({
            'mse_weight': args.mse_weight,
            'first_deriv_weight': args.first_deriv_weight,
            'second_deriv_weight': args.second_deriv_weight
        })
    
    saved_model_path = save_model(model, args.model_name, save_dir=args.model_dir, **model_params)
    
    return model, train_dataset, val_dataset, run_folder, saved_model_path

def run_evaluation(args, model=None, val_dataset=None, run_folder=None):
    """Run the evaluation pipeline"""
    print("=== EVALUATION MODE ===")
    
    # Load model if not provided
    if model is None:
        model = load_pretrained_model(args.model_name, 
                                    args.input_length, 
                                    args.output_length,
                                    load_dir=args.model_dir,
                                    use_cnn_02=args.use_cnn_02)
        if model is None:
            print("No model found for evaluation!")
            return None, None
    
    # Create validation dataset if not provided
    if val_dataset is None:
        val_dataset = CircleDataset(
            num_samples=args.val_samples,
            input_length=args.input_length,
            output_length=args.output_length,
            radius=args.radius,
            noise_level=args.noise_level
        )
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(model, val_dataset, 
                     num_samples=args.num_plot_samples, 
                     save_path=args.plot_dir,
                     run_folder=run_folder)
    
    plot_circle(model, val_dataset, save_path=args.plot_dir, run_folder=run_folder)
    
    return model, val_dataset

def run_plotting(args, model=None, val_dataset=None, run_folder=None):
    """Run derivative analysis and plotting"""
    print("=== PLOTTING MODE ===")
    
    # Load model if not provided
    if model is None:
        model = load_pretrained_model(args.model_name,
                                    args.input_length,
                                    args.output_length,
                                    load_dir=args.model_dir,
                                    use_cnn_02=args.use_cnn_02)
        if model is None:
            print("No model found for plotting!")
            return
    
    # Create validation dataset if not provided
    if val_dataset is None:
        val_dataset = CircleDataset(
            num_samples=args.val_samples,
            input_length=args.input_length,
            output_length=args.output_length,
            radius=args.radius,
            noise_level=args.noise_level
        )
    
    # Analyze predictions and derivatives
    analyze_model_predictions(model, val_dataset, save_path=args.plot_dir, run_folder=run_folder)

def run_demo(args, run_folder=None, model=None, model_path=None):
    """Run pretrained model demonstration with trajectory utilities"""
    print("=== DEMO MODE ===")
    
    # If model is provided directly, use it (from training pipeline)
    if model is not None:
        print("Using provided trained model for demo")
        # Generate a prediction to get kx, ky trajectories
        model.eval()
        with torch.no_grad():
            # Create test input
            test_input = torch.linspace(0, 0.5, 128 + 1) #+ 0.01 * torch.randn(128)
            test_input = test_input[:-1]  # Remove last point to match output length
            test_input = test_input.unsqueeze(0).to(next(model.parameters()).device)
            
            # Get prediction
            prediction = model(test_input)
            kx = prediction[0, 0, :].cpu().numpy()  # First channel (kx)
            ky = prediction[0, 1, :].cpu().numpy()  # Second channel (ky)
            
            # Apply trajectory utilities
            result = demo_trajectory_utilities(kx, ky, args.plot_dir, run_folder)
            
        # Also run the original plotting
        plot_pretrained_demo(model, save_path=args.plot_dir, run_folder=run_folder)
        return model    # Otherwise, try to load from file
    model_name = model_path if model_path is not None else args.model_name
    
    result = demo_pretrained_usage(model_name, load_dir=args.model_dir, use_cnn_02=args.use_cnn_02)
    
    # Handle the case where demo_pretrained_usage returns None
    if result is None:
        print("Demo failed - no model available")
        return None
    
    # Unpack the result safely
    if isinstance(result, tuple) and len(result) == 2:
        model, prediction = result
        
        # Extract kx, ky from prediction if available
        if prediction is not None:
            kx = prediction[0].numpy()  # First channel (kx)
            ky = prediction[1].numpy()  # Second channel (ky)
            
            # Apply trajectory utilities
            result = demo_trajectory_utilities(kx, ky, args.plot_dir, run_folder)
            
    else:
        model = result
        prediction = None
        
        # Generate a new prediction to get kx, ky trajectories
        if model is not None:
            model.eval()
            with torch.no_grad():
                # Create test input
                test_input = torch.linspace(0, 0.5, 128) #+ 0.01 * torch.randn(128)
                test_input = test_input.unsqueeze(0).to(next(model.parameters()).device)
                
                # Get prediction
                prediction = model(test_input)
                kx = prediction[0, 0, :].cpu().numpy()  # First channel (kx)
                ky = prediction[0, 1, :].cpu().numpy()  # Second channel (ky)
                
                # Apply trajectory utilities
                result = demo_trajectory_utilities(kx, ky, args.plot_dir, run_folder)
    
    if model is not None:
        plot_pretrained_demo(model, save_path=args.plot_dir, run_folder=run_folder)
    
    return model


def main():
    """Main execution function"""
    args = parse_arguments()
    
    # Get appropriate directories based on model architecture
    model_dir, plot_dir = get_model_subdirs(args.use_cnn_02)
    
    # Override the default directories
    if args.model_dir == 'models':  # Only override if using default
        args.model_dir = model_dir
    if args.plot_dir == 'plots':    # Only override if using default
        args.plot_dir = plot_dir
    
    # Create directories
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    
    print(f"Running in {args.mode} mode...")
    print(f"Configuration:")
    print(f"  - Model architecture: {'CircleCNN_02 (deeper)' if args.use_cnn_02 else 'CircleCNN (standard)'}")
    print(f"  - Input/Output length: {args.input_length}/{args.output_length}")
    print(f"  - Hidden channels: {args.hidden_channels}")
    print(f"  - Training samples: {args.train_samples}")
    print(f"  - Validation samples: {args.val_samples}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Epochs: {args.num_epochs}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Circle radius: {args.radius}")
    print(f"  - Noise level: {args.noise_level}")
    print(f"  - Use smooth loss: {args.use_smooth_loss and not args.no_smooth_loss}")
    if args.use_smooth_loss and not args.no_smooth_loss:
        print(f"    - MSE weight: {args.mse_weight}")
        print(f"    - First deriv weight: {args.first_deriv_weight}")
        print(f"    - Second deriv weight: {args.second_deriv_weight}")
    print(f"  - Use learning rate scheduler: {args.use_scheduler}")
    if args.pretrained_model:
        print(f"  - Pretrained model: {args.pretrained_model}")
    print()
    
    # Track the actual saved model name for final instructions
    saved_model_name = args.model_name
    
    # Execute based on mode
    if args.mode == 'train':
        result = run_training(args)
        if result[0] is None:  # Training was interrupted
            return
        model, train_dataset, val_dataset, run_folder, saved_model_path = result
        # Extract just the filename from the full path
        saved_model_name = os.path.basename(saved_model_path) if saved_model_path else args.model_name
        
    elif args.mode == 'evaluate':
        model, val_dataset = run_evaluation(args)
        
    elif args.mode == 'plot':
        run_plotting(args)
        
    elif args.mode == 'demo':
        model = run_demo(args)
        
    elif args.mode == 'all':
        # Run complete pipeline
        print("=== RUNNING COMPLETE PIPELINE ===")
        result = run_training(args)
        if result[0] is None:  # Training was interrupted
            return
        model, train_dataset, val_dataset, run_folder, saved_model_path = result
        # Extract just the filename from the full path
        saved_model_name = os.path.basename(saved_model_path) if saved_model_path else args.model_name
        
        if model is not None:
            run_evaluation(args, model, val_dataset, run_folder)
            run_plotting(args, model, val_dataset, run_folder)
            run_demo(args, run_folder, model=model)  # Pass the trained model directly
    
    print("\n=== PIPELINE COMPLETED ===")
    print(f"Models saved in: {args.model_dir}")
    print(f"Plots saved in: {args.plot_dir}")
    if 'run_folder' in locals():
        print(f"Current run plots saved in: {run_folder}")
    print("\n=== HOW TO REUSE THIS MODEL ===")
    if args.use_cnn_02:
        print(f"1. Load model: python run.py --mode demo --use_cnn_02 --model_name {saved_model_name}")
        print(f"2. Evaluate: python run.py --mode evaluate --use_cnn_02 --model_name {saved_model_name}")
        print(f"3. Retrain: python run.py --mode train --use_cnn_02 --pretrained_model {saved_model_name}")
    else :
        print(f"1. Load model: python run.py --mode demo --model_name {saved_model_name}")
        print(f"2. Evaluate: python run.py --mode evaluate --model_name {saved_model_name}")
        print(f"3. Retrain: python run.py --mode train --pretrained_model {saved_model_name}")


if __name__ == "__main__":
    main()
