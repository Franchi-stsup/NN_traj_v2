"""
Plotting and Visualization Functions
Contains all functions for plotting training results, model predictions, and derivative analysis.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from datetime import datetime
from src.network import get_device
from src.train import inference_single_sample

GAMMA = 42.575575  # Gyromagnetic ratio in MHz/T
DT_NN = 0.005/1.28
DURATION = 0.5  # Duration of the trajectory in seconds
device = get_device()


def create_run_folder(save_path='plots', **kwargs):
    """
    Create a subfolder with training parameters and timestamp
    
    Args:
        save_path: Base path for plots
        **kwargs: Training parameters (num_epochs, learning_rate, batch_size, etc.)
    
    Returns:
        str: Path to the created subfolder
    """
    return create_structured_subfolder(save_path, None, **kwargs)

def create_structured_subfolder(save_path='plots', mode_suffix=None, **kwargs):
    """
    Create a structured subfolder with parameters and timestamp
    
    Args:
        save_path: Base path for plots
        mode_suffix: Optional suffix for the mode (e.g., 'eval', 'demo')
        **kwargs: Training parameters (num_epochs, learning_rate, batch_size, etc.)
    
    Returns:
        str: Path to the created subfolder
    """
    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Extract common parameters
    params = []
    if 'num_epochs' in kwargs:
        params.append(f"ep{kwargs['num_epochs']}")
    if 'learning_rate' in kwargs:
        params.append(f"lr{kwargs['learning_rate']:.0e}")
    if 'batch_size' in kwargs:
        params.append(f"bs{kwargs['batch_size']}")
    # if 'radius' in kwargs:
    #     params.append(f"r{kwargs['radius']}")
    if 'use_scheduler' in kwargs and kwargs['use_scheduler']:
        params.append("sched")
    if 'use_smooth_loss' in kwargs and kwargs['use_smooth_loss']:
        params.append("smooth")
        # Add smooth loss weights if present
        if 'mse_weight' in kwargs:
            params.append(f"mse{kwargs['mse_weight']}")
        if 'first_deriv_weight' in kwargs:
            params.append(f"fd{kwargs['first_deriv_weight']}")
        if 'second_deriv_weight' in kwargs:
            params.append(f"sd{kwargs['second_deriv_weight']}")
    
    # Add mode suffix if provided
    if mode_suffix:
        params.append(mode_suffix)
    
    # Create folder name
    if params:
        folder_name = f"{timestamp}_{'_'.join(params)}"
    else:
        folder_name = timestamp
    
    # Create full path
    full_path = os.path.join(save_path, folder_name)
    os.makedirs(full_path, exist_ok=True)
    
    return full_path


def plot_training_curves(train_losses, val_losses, save_path='plots', **kwargs):
    """Plot training and validation loss curves"""
    # Create run-specific subfolder
    run_folder = create_run_folder(save_path, **kwargs)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_folder, 'training_loss.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    # plt.show()  # Disabled - only save figures
    
    return run_folder  # Return the folder path for other functions to use


def visualize_results(model, dataset, num_samples=3, save_path='plots', run_folder=None):
    """Visualize the model predictions"""
    # Use existing run folder or create plots directory
    if run_folder is None:
        run_folder = save_path
    os.makedirs(run_folder, exist_ok=True)
    
    model.eval()
    
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 8))
    
    with torch.no_grad():
        for i in range(num_samples):
            input_signal, target = dataset[i]
            input_signal = input_signal.unsqueeze(0).to(device)
            
            prediction = model(input_signal).cpu().squeeze(0)
            
            # Time vector for output (128 points at 5ms intervals)
            time_output = np.linspace(0, DURATION, len(target[0].numpy()))
            
            # Plot kx
            axes[0, i].plot(time_output, target[0].numpy(), 'b-', label='Target kx', linewidth=2)
            axes[0, i].plot(time_output, prediction[0].numpy(), 'r--', label='Predicted kx', linewidth=2)
            axes[0, i].set_title(f'kx(t) - Sample {i+1}')
            axes[0, i].set_xlabel('Time (s)')
            axes[0, i].set_ylabel('kx')
            axes[0, i].legend()
            axes[0, i].grid(True)
            
            # Plot ky
            axes[1, i].plot(time_output, target[1].numpy(), 'b-', label='Target ky', linewidth=2)
            axes[1, i].plot(time_output, prediction[1].numpy(), 'r--', label='Predicted ky', linewidth=2)
            axes[1, i].set_title(f'ky(t) - Sample {i+1}')
            axes[1, i].set_xlabel('Time (s)')
            axes[1, i].set_ylabel('ky')
            axes[1, i].legend()
            axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, 'kx_ky_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    # plt.show()  # Disabled - only save figures


def plot_circle(model, dataset, sample_idx=0, save_path='plots', run_folder=None):
    """Plot the predicted circle in 2D space"""
    # Use existing run folder or create plots directory
    if run_folder is None:
        run_folder = save_path
    os.makedirs(run_folder, exist_ok=True)
    
    model.eval()
    
    with torch.no_grad():
        input_signal, target = dataset[sample_idx]
        input_signal = input_signal.unsqueeze(0).to(device)
        
        prediction = model(input_signal).cpu().squeeze(0)
        
        plt.figure(figsize=(8, 8))
        plt.plot(target[0].numpy(), target[1].numpy(), 'b-', label='Target Circle', linewidth=3)
        plt.plot(prediction[0].numpy(), prediction[1].numpy(), 'r--', label='Predicted Circle', linewidth=2)
        plt.xlabel('kx')
        plt.ylabel('ky')
        plt.title('Circle in k-space')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(os.path.join(run_folder, 'circle_2d_plot.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to free memory
        # plt.show()  # Disabled - only save figures


def calculate_derivatives(kx, ky, dt=DT_NN):
    """
    Calculate first and second derivatives of kx and ky
    
    Args:
        kx: numpy array of kx coordinates (128 points)
        ky: numpy array of ky coordinates (128 points)
        dt: time step in seconds (default 5ms)
    
    Returns:
        dict containing first and second derivatives
    """
    # First derivatives (velocity)
    dkx_dt = np.gradient(kx, dt)  # dx/dt
    dky_dt = np.gradient(ky, dt)  # dy/dt
    
    # Second derivatives (acceleration)
    d2kx_dt2 = np.gradient(dkx_dt, dt)  # d²x/dt²
    d2ky_dt2 = np.gradient(dky_dt, dt)  # d²y/dt²

    # Calculate magnitude of velocity and acceleration
    velocity_magnitude = np.sqrt(dkx_dt**2 + dky_dt**2)
    acceleration_magnitude = np.sqrt(d2kx_dt2**2 + d2ky_dt2**2)
    

    # Divide by gamma to convert to physical units (Hz)
    dkx_dt /= GAMMA
    dky_dt /= GAMMA
    d2kx_dt2 /= GAMMA
    d2ky_dt2 /= GAMMA
    velocity_magnitude /= GAMMA
    acceleration_magnitude /= GAMMA
    return {
        'dkx_dt': dkx_dt,
        'dky_dt': dky_dt,
        'd2kx_dt2': d2kx_dt2,
        'd2ky_dt2': d2ky_dt2,
        'velocity_magnitude': velocity_magnitude,
        'acceleration_magnitude': acceleration_magnitude
    }


def plot_derivatives(kx_target, ky_target, kx_pred, ky_pred, dt=DT_NN, save_path='plots', run_folder=None):
    """
    Plot derivatives of target vs predicted trajectories
    
    Args:
        kx_target, ky_target: target coordinates
        kx_pred, ky_pred: predicted coordinates  
        dt: time step in seconds
        save_path: base directory to save plots
        run_folder: specific run folder (if None, uses save_path)
    """
    # Use existing run folder or create plots directory
    if run_folder is None:
        run_folder = save_path
    os.makedirs(run_folder, exist_ok=True)
    
    # Calculate derivatives for both target and predicted
    target_derivs = calculate_derivatives(kx_target, ky_target, dt)
    pred_derivs = calculate_derivatives(kx_pred, ky_pred, dt)
    
    # Time vector
    time = np.linspace(0, DURATION, len(kx_target))  # 128 points at 5ms intervals
    
    # Create subplots
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    
    # Plot kx and ky positions
    axes[0, 0].plot(time, kx_target, 'b-', label='Target kx', linewidth=2)
    axes[0, 0].plot(time, kx_pred, 'r--', label='Predicted kx', linewidth=2)
    axes[0, 0].set_title('Position: kx(t)')
    axes[0, 0].set_ylabel('kx')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    axes[0, 1].plot(time, ky_target, 'b-', label='Target ky', linewidth=2)
    axes[0, 1].plot(time, ky_pred, 'r--', label='Predicted ky', linewidth=2)
    axes[0, 1].set_title('Position: ky(t)')
    axes[0, 1].set_ylabel('ky')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot first derivatives (velocity)
    axes[1, 0].plot(time, target_derivs['dkx_dt'], 'b-', label='Target dkx/dt', linewidth=2)
    axes[1, 0].plot(time, pred_derivs['dkx_dt'], 'r--', label='Predicted dkx/dt', linewidth=2)
    axes[1, 0].set_title('First Derivative: dkx/dt (velocity)')
    axes[1, 0].set_ylabel('dkx/dt')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    axes[1, 1].plot(time, target_derivs['dky_dt'], 'b-', label='Target dky/dt', linewidth=2)
    axes[1, 1].plot(time, pred_derivs['dky_dt'], 'r--', label='Predicted dky/dt', linewidth=2)
    axes[1, 1].set_title('First Derivative: dky/dt (velocity)')
    axes[1, 1].set_ylabel('dky/dt')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    # Plot second derivatives (acceleration)
    axes[2, 0].plot(time, target_derivs['d2kx_dt2'], 'b-', label='Target d²kx/dt²', linewidth=2)
    axes[2, 0].plot(time, pred_derivs['d2kx_dt2'], 'r--', label='Predicted d²kx/dt²', linewidth=2)
    axes[2, 0].set_title('Second Derivative: d²kx/dt² (acceleration)')
    axes[2, 0].set_ylabel('d²kx/dt²')
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].legend()
    axes[2, 0].grid(True)
    
    axes[2, 1].plot(time, target_derivs['d2ky_dt2'], 'b-', label='Target d²ky/dt²', linewidth=2)
    axes[2, 1].plot(time, pred_derivs['d2ky_dt2'], 'r--', label='Predicted d²ky/dt²', linewidth=2)
    axes[2, 1].set_title('Second Derivative: d²ky/dt² (acceleration)')
    axes[2, 1].set_ylabel('d²ky/dt²')
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].legend()
    axes[2, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, 'derivatives_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    # plt.show()  # Disabled - only save figures
    
    # Plot magnitude comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Velocity magnitude
    axes[0].plot(time, target_derivs['velocity_magnitude'], 'b-', label='Target |velocity|', linewidth=2)
    axes[0].plot(time, pred_derivs['velocity_magnitude'], 'r--', label='Predicted |velocity|', linewidth=2)
    axes[0].set_title('Velocity Magnitude')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('|dK/dt|')
    axes[0].legend()
    axes[0].grid(True)
    
    # Acceleration magnitude
    axes[1].plot(time, target_derivs['acceleration_magnitude'], 'b-', label='Target |acceleration|', linewidth=2)
    axes[1].plot(time, pred_derivs['acceleration_magnitude'], 'r--', label='Predicted |acceleration|', linewidth=2)
    axes[1].set_title('Acceleration Magnitude')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('|d²K/dt²|')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_folder, 'magnitude_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    # plt.show()  # Disabled - only save figures
    
    # Print statistics
    print("\n=== DERIVATIVE ANALYSIS ===")
    print(f"Target velocity range: [{target_derivs['velocity_magnitude'].min():.3f}, {target_derivs['velocity_magnitude'].max():.3f}]")
    print(f"Predicted velocity range: [{pred_derivs['velocity_magnitude'].min():.3f}, {pred_derivs['velocity_magnitude'].max():.3f}]")
    print(f"Target acceleration range: [{target_derivs['acceleration_magnitude'].min():.3f}, {target_derivs['acceleration_magnitude'].max():.3f}]")
    print(f"Predicted acceleration range: [{pred_derivs['acceleration_magnitude'].min():.3f}, {pred_derivs['acceleration_magnitude'].max():.3f}]")
    
    # Calculate errors
    velocity_error = np.mean(np.abs(target_derivs['velocity_magnitude'] - pred_derivs['velocity_magnitude']))
    accel_error = np.mean(np.abs(target_derivs['acceleration_magnitude'] - pred_derivs['acceleration_magnitude']))
    print(f"Mean velocity magnitude error: {velocity_error:.6f}")
    print(f"Mean acceleration magnitude error: {accel_error:.6f}")
    
    return target_derivs, pred_derivs


def plot_trajectory_shift_comparison(kx, ky, kx_shifted, ky_shifted, save_path, filename='trajectory_shift_demo.png'):
    """
    Plot comparison between original and shifted trajectories
    
    Args:
        kx, ky: original trajectory coordinates
        kx_shifted, ky_shifted: shifted trajectory coordinates
        save_path: directory to save the plot
        filename: name of the output file
    
    Returns:
        str: path to saved plot
    """
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
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_rotated_trajectories(kSpaceTrj, save_path, filename='trajectory_rotations_demo.png'):
    """
    Plot rotated trajectories
    
    Args:
        kSpaceTrj: dictionary with 'kxx' and 'kyy' arrays
        save_path: directory to save the plot
        filename: name of the output file
    
    Returns:
        str: path to saved plot
    """
    plt.figure(figsize=(8, 8))
    kxx = kSpaceTrj['kxx']
    kyy = kSpaceTrj['kyy']
    n_rotations = kxx.shape[1]
    
    for i in range(kxx.shape[0]):
        plt.plot(kxx[i, :], kyy[i, :], lw=0.8, alpha=0.7)

    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.axis("equal")
    plt.title(f"Rotated k-space Trajectories (n={n_rotations})")
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_trajectory_utilities_combined(kx, ky, kx_shifted, ky_shifted, kSpaceTrj, save_path, filename='trajectory_utilities_combined.png'):
    """
    Plot combined view of original, shifted, and rotated trajectories
    
    Args:
        kx, ky: original trajectory coordinates
        kx_shifted, ky_shifted: shifted trajectory coordinates
        kSpaceTrj: dictionary with 'kxx' and 'kyy' arrays for rotated trajectories
        save_path: directory to save the plot
        filename: name of the output file
    
    Returns:
        str: path to saved plot
    """
    plt.figure(figsize=(10, 8))
    
    kxx = kSpaceTrj['kxx']
    kyy = kSpaceTrj['kyy']
    
    # Plot original
    plt.plot(kx, ky, 'b-', linewidth=3, label='Original', alpha=0.8)
    
    # Plot shifted
    plt.plot(kx_shifted, ky_shifted, 'r-', linewidth=2, label='Shifted', alpha=0.8)
    
    # Plot a few rotated versions
    for i in range(0, min(5, kxx.shape[0])):
        plt.plot(kxx[i, :], kyy[i, :], '--', linewidth=1, alpha=0.5, 
                label=f'Rotated {i+1}' if i < 3 else None)
    
    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.axis("equal")
    plt.title("Trajectory Utilities Demo: Original, Shifted, and Rotated")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(save_path, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


def plot_pretrained_demo(model, save_path='plots', run_folder=None):
    """Plot the predicted circle from pretrained model demo"""
    # Use existing run folder or create plots directory
    if run_folder is None:
        run_folder = save_path
    os.makedirs(run_folder, exist_ok=True)
    
    # Create some test input (simulated time signal)
    test_input = torch.linspace(0, DURATION, 128 + 1) # + 0.01 * torch.randn(128)
    test_input = test_input[:-1]  # Remove last point to match output length
    
    # Single sample inference
    prediction = inference_single_sample(model, test_input)
    
    # Extract kx and ky
    kx_pred = prediction[0].numpy()
    ky_pred = prediction[1].numpy()
    
    # Plot the predicted circle
    plt.figure(figsize=(8, 8))
    plt.plot(kx_pred, ky_pred, 'r-', linewidth=2, label='Predicted Circle')
    plt.xlabel('kx')
    plt.ylabel('ky')
    plt.title('Predicted Circle from Pretrained Model')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(os.path.join(run_folder, 'pretrained_demo_circle.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    # plt.show()  # Disabled - only save figures


def analyze_model_predictions(model, dataset, save_path='plots', run_folder=None):
    """Complete analysis of model predictions including derivatives"""
    print("Analyzing model predictions and derivatives...")
    
    # Use existing run folder or create plots directory
    if run_folder is None:
        run_folder = save_path
    
    # Get a sample prediction for derivative analysis
    model.eval()
    with torch.no_grad():
        sample_input, sample_target = dataset[0]
        sample_input = sample_input.unsqueeze(0).to(device)
        sample_prediction = model(sample_input).cpu().squeeze(0)
        
        # Extract coordinates
        kx_target = sample_target[0].numpy()
        ky_target = sample_target[1].numpy()
        kx_pred = sample_prediction[0].numpy()
        ky_pred = sample_prediction[1].numpy()
        
        # Plot derivatives
        target_derivs, pred_derivs = plot_derivatives(kx_target, ky_target, 
                                                    kx_pred, ky_pred, 
                                                    dt=DT_NN, save_path=save_path, 
                                                    run_folder=run_folder)
    
    return target_derivs, pred_derivs
