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
from src.network_utils import shift_trajectory, rotate_traj,complex_traj, plot_trajs
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
    parser.add_argument('--noise_level', type=float, default=0.01,
                       help='Noise level for input signals (default: 0.01)')
    
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
    parser.add_argument('--second_deriv_weight', type=float, default=0.000,
                       help='Second derivative weight in smooth loss (default: 0.000)')
    
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
        second_deriv_weight=args.second_deriv_weight if use_smooth else 0.000,
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
        'radius': args.radius,
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
        'radius': args.radius,
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


def demo_trajectory_utilities(kx, ky, save_path, run_folder=None):
    """
    Apply trajectory utilities (shift, rotate, plot) to kx, ky trajectories
    
    Args:
        kx: numpy array of kx coordinates
        ky: numpy array of ky coordinates  
        save_path: directory to save plots
        run_folder: optional subfolder for organized saving
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    print("\n=== TRAJECTORY UTILITIES DEMO ===")
    
    # 1. Original trajectory
    print("1. Original trajectory:")
    print(f"   kx range: [{kx.min():.3f}, {kx.max():.3f}]")
    print(f"   ky range: [{ky.min():.3f}, {ky.max():.3f}]")
    
    # 2. Shift trajectory
    print("2. Applying trajectory shift...")
    kx_shifted, ky_shifted = shift_trajectory(kx, ky)
    print(f"   Shifted kx range: [{kx_shifted.min():.3f}, {kx_shifted.max():.3f}]")
    print(f"   Shifted ky range: [{ky_shifted.min():.3f}, {ky_shifted.max():.3f}]")
    
    # 3. Rotate trajectory (using shifted version)
    print("3. Generating rotated trajectories...")
    n_rotations = 79  # Reduce number for demo
    kSpaceTrj = rotate_traj(kx_shifted, ky_shifted, n_rotation=n_rotations)
    print(f"   Generated {n_rotations} rotated trajectories")
    
    # 4. Create plots
    print("4. Creating trajectory plots...")
    
    # Create save directory
    if run_folder:
        plot_save_dir = os.path.join(save_path, run_folder)
    else:
        plot_save_dir = save_path
    os.makedirs(plot_save_dir, exist_ok=True)
    
    # Plot 1: Original vs Shifted
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(kx, ky, 'b-', linewidth=2, label='Original')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title('Original Trajectory')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.subplot(1, 2, 2)
    plt.plot(kx_shifted, ky_shifted, 'r-', linewidth=2, label='Shifted')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title('Shifted Trajectory')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plt.tight_layout()
    shift_plot_path = os.path.join(plot_save_dir, 'trajectory_shift_demo.png')
    plt.savefig(shift_plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved shift comparison to: {shift_plot_path}")
    plt.close()
    
    # Plot 2: Rotated trajectories using the plot_trajs function
    print("5. Plotting rotated trajectories...")
    
    # Temporarily redirect the plot_trajs function to save instead of show
    plt.figure(figsize=(8, 8))
    kxx = kSpaceTrj['kxx']
    kyy = kSpaceTrj['kyy']
    
    for i in range(kxx.shape[1]):
        plt.plot(kxx[:, i], kyy[:, i], lw=0.8, alpha=0.7)
    
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.axis("equal")
    plt.title(f"Rotated k-space Trajectories (n={n_rotations})")
    plt.grid(True, alpha=0.3)
    
    rotated_plot_path = os.path.join(plot_save_dir, 'trajectory_rotations_demo.png')
    plt.savefig(rotated_plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved rotated trajectories to: {rotated_plot_path}")
    plt.close()
    
    # Plot 3: Combined view
    plt.figure(figsize=(10, 8))
    
    # Plot original
    plt.plot(kx, ky, 'b-', linewidth=3, label='Original', alpha=0.8)
    
    # Plot shifted
    plt.plot(kx_shifted, ky_shifted, 'r-', linewidth=2, label='Shifted', alpha=0.8)
    
    # Plot a few rotated versions
    for i in range(0, min(5, kxx.shape[1])):
        plt.plot(kxx[:, i], kyy[:, i], '--', linewidth=1, alpha=0.5, 
                label=f'Rotated {i+1}' if i < 3 else None)
    
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.axis("equal")
    plt.title("Trajectory Utilities Demo: Original, Shifted, and Rotated")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    combined_plot_path = os.path.join(plot_save_dir, 'trajectory_utilities_combined.png')
    plt.savefig(combined_plot_path, dpi=150, bbox_inches='tight')
    print(f"   Saved combined view to: {combined_plot_path}")
    plt.close()
    
    config = create_default_config()
    os.makedirs(config.tmp_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)
    kspace_file_path = os.path.join(config.tmp_dir, 'kspace_cartesian.npy')
    if os.path.exists(kspace_file_path):
        kspace_data = np.load(kspace_file_path)
        print(f"  Loaded k-space data from {kspace_file_path}")
        print(f"  K-space data shape: {kspace_data.shape}")

    rosette_traj = complex_traj(kSpaceTrj)

    print(f" \n\n\nGenerated rosette trajectory with shape: {rosette_traj.shape}")
    print(f" kspace data shape: {kspace_data.shape}\n\n\n")

    kspace_sampled = fast_kspace_interpolation_v3(kspace_data, rosette_traj, FOV)

    # kx_traj = np.real(rosette_traj)
    # mr_data = kspace_sampled.reshape(kx_traj.shape)
    para = build_para(config)
    mr_data = kspace_sampled.reshape(kxx.shape)
    run_gpu = True if torch.cuda.is_available() else False
    recon_img = run_bart_nufft(mr_data, kSpaceTrj, para, run_gpu)

    print(f"Reconstructed image shape: {recon_img.shape}")
    print(f"Reconstructed image dtype: {recon_img.dtype}\n \n")
    [Nx, Ny] = kspace_data.shape[0], kspace_data.shape[1]

    recon_img = rescale_recon_img(recon_img, Nx, Ny, RES)
    np.save(os.path.join(config.tmp_dir, 'recon_img.npy'), recon_img)

    ground_truth_img_down_path = os.path.join(config.tmp_dir, 'ground_truth_img_down.npy')
    ground_truth_img_down = np.load(ground_truth_img_down_path) 

    plot_comparison(ground_truth_img_down, recon_img,
        save_path='plots_bart/comparison.png',  # Optional: path to save the figure
        show=False                        # Optional: display the plot interactively
    )


    print(f"=== TRAJECTORY UTILITIES DEMO COMPLETED ===\n")

    


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
            test_input = torch.linspace(0, 0.64, 128) + 0.01 * torch.randn(128)
            test_input = test_input.unsqueeze(0).to(next(model.parameters()).device)
            
            # Get prediction
            prediction = model(test_input)
            kx = prediction[0, 0, :].cpu().numpy()  # First channel (kx)
            ky = prediction[0, 1, :].cpu().numpy()  # Second channel (ky)
            
        # Apply trajectory utilities
        demo_trajectory_utilities(kx, ky, args.plot_dir, run_folder)
        
        # Also run the original plotting
        plot_pretrained_demo(model, save_path=args.plot_dir, run_folder=run_folder)
        return model
    
    # Otherwise, try to load from file
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
            demo_trajectory_utilities(kx, ky, args.plot_dir, run_folder)
            
    else:
        model = result
        prediction = None
        
        # Generate a new prediction to get kx, ky trajectories
        if model is not None:
            model.eval()
            with torch.no_grad():
                # Create test input
                test_input = torch.linspace(0, 0.64, 128) + 0.01 * torch.randn(128)
                test_input = test_input.unsqueeze(0).to(next(model.parameters()).device)
                
                # Get prediction
                prediction = model(test_input)
                kx = prediction[0, 0, :].cpu().numpy()  # First channel (kx)
                ky = prediction[0, 1, :].cpu().numpy()  # Second channel (ky)
                
                # Apply trajectory utilities
                demo_trajectory_utilities(kx, ky, args.plot_dir, run_folder)
    
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
    print(f"1. Load model: python run.py --mode demo --model_name {saved_model_name}")
    print(f"2. Evaluate: python run.py --mode evaluate --model_name {saved_model_name}")


if __name__ == "__main__":
    main()
