"""
Utility functions for BART NUFFT Rosette Trajectory Reconstruction
================================================================

This module contains utility functions for:
- Image loading and preprocessing
- Data visualization
- BART interface functions
- File I/O operations
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from typing import Tuple, Optional, Dict, Any
import logging
from src.bart_interface import bart
from skimage.transform import resize

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def load_image(image_path: str) -> np.ndarray:
    """Load and preprocess image for reconstruction."""
    img = Image.open(image_path).convert('L')
    img = np.array(img, dtype=np.float32)
    return img

def bart_reformat_nufft(image: np.ndarray, kSpaceTrj: dict) -> tuple:
    """Reformat image and trajectory for BART NUFFT (direct MATLAB translation)."""
    N = image.shape[0]
    image_in_6D = image.reshape(1, image.shape[0], image.shape[1], 1, 1, 1)
    kxx = kSpaceTrj['kxx'] / np.max(np.abs(kSpaceTrj['kxx'])) * N/2
    kyy = kSpaceTrj['kyy'] / np.max(np.abs(kSpaceTrj['kyy'])) * N/2
    kxx = kxx.reshape(1, kxx.shape[0], kxx.shape[1])
    kyy = kyy.reshape(1, kyy.shape[0], kyy.shape[1])
    kSpace_Trj_BART = np.concatenate([kxx, kyy], axis=0)
    kSpace_Trj_BART = np.pad(kSpace_Trj_BART, ((0,1),(0,0),(0,0)), 'constant') # 3rd row zeros
    return kSpace_Trj_BART, image_in_6D

def create_cartesian_trajectory(nx: int, ny: int) -> dict:
    kx, ky = np.meshgrid(np.arange(-ny//2, ny//2), np.arange(-nx//2, nx//2))
    return {'kxx': kx, 'kyy': ky}

def generate_rosette_trajectory(FoV, res, noPaddles, points_per_paddle, o1=1.0, o2=1.0):
    """
    Generates a 2D rosette k-space trajectory (direct translation from MATLAB).
    """
    Kmax_res = res / (FoV * 1e-3) / 2
    tVec = np.linspace(0, 1, points_per_paddle, endpoint=False)
    beta_angles = np.linspace(0, 2 * np.pi, noPaddles, endpoint=False)
    Kxy_smp_paddles = []
    for beta_angle in beta_angles:
        theta = (o2 / o1) * np.pi * tVec + beta_angle
        radius = np.sin((o1 / o1) * np.pi * tVec)
        paddle = Kmax_res * radius * np.exp(1j * theta)
        Kxy_smp_paddles.append(paddle)
    Kxy_smp = np.concatenate(Kxy_smp_paddles)
    print(f"Trajectory generated: {len(Kxy_smp)} total points ({points_per_paddle} × {noPaddles})")
    return Kxy_smp


def run_bart_nufft(mrData, kSpaceTrj, para, run_gpu=False):
    """
    Python translation of MATLAB runBartRecon4 for BART NUFFT reconstruction.
    Args:
        mrData: numpy array, k-space data
        kSpaceTrj: dict with keys 'kxx', 'kyy'
        para: dict of parameters
    Returns:
        reconMrsi: reconstructed image (numpy array)
    """

    # --- Parameter defaults ---
    para = para.copy()
    para.setdefault('oprPath', None)
    para.setdefault('isWaterRef', False)
    para.setdefault('dbgOn', False)
    para.setdefault('csReg', 0.05)
    para.setdefault('csMap', None)
    para.setdefault('itr', 50)
    para.setdefault('CGitr', 5)
    para.setdefault('useNorm', False)
    para.setdefault('applyFilter', '')
    para.setdefault('reconType', None)
    para.setdefault('kFac', 1)
    para.setdefault('doK0Cor', False)
    para.setdefault('k0ZeroFill', 4)
    para.setdefault('doB0Cor', False)
    para.setdefault('B0refCh', 27)
    para.setdefault('B0corMode', 'lin')
    para.setdefault('acqConf', {'H1offset': 4.7})
    para.setdefault('mSize', mrData.shape[1] if mrData.ndim > 1 else mrData.shape[0])

    # --- Normalize k-space trajectory ---
    # print kspace_traj
    # print(f"Normalizing k-space trajectory with max value: {np.max(np.abs(kSpaceTrj['kxx']))}")
    kxx = kSpaceTrj['kxx'] / np.max(np.abs(kSpaceTrj['kxx'])) * para['mSize']/2 * para['kFac']
    kyy = kSpaceTrj['kyy'] / np.max(np.abs(kSpaceTrj['kyy'])) * para['mSize']/2 * para['kFac']

    # print(f"Normalized k-space trajectory shapes: kxx={kxx.shape}, kyy={kyy.shape}")
    kxx = kxx.reshape(1, kxx.shape[0], kxx.shape[1])
    kyy = kyy.reshape(1, kyy.shape[0], kyy.shape[1])

    # print(f"Added a 3rd dimension to kxx and kyy: kxx={kxx.shape}, kyy={kyy.shape}")
    kspaceRSI = np.concatenate([kxx, kyy], axis=0)
    # print(f"kspace RSI shape: {kspaceRSI.shape}")
    kspaceRSI = np.concatenate([kspaceRSI, np.zeros((1, kxx.shape[1], kxx.shape[2]))], axis=0)
    # print(f"kspace RSI shape with 3rd axis: {kspaceRSI.shape}")
    # print(f"Last column of kspaceRSI (should be zeros): {kspaceRSI[-1, :, :]}")

    # --- Reshape mrData ---
    # MATLAB: mrData = reshape( mrData, [1 size(mrData,1) size(mrData,2) size(mrData,3) 1 1 1 1 1 1 size(mrData,4)] );
    # For typical 2D data, this becomes (1, Nx, Ny, 1, 1, 1, 1, 1, 1, 1, 1)
    shape = [1] + list(mrData.shape) + [1]*(11-len(mrData.shape)-1)
    # print(f"Shape {shape}")
    mrData_reshaped = mrData.reshape(shape)
    mrData2 = np.squeeze(mrData_reshaped)
    mrData2 = mrData2[np.newaxis, :, :]

    # --- BART NUFFT reconstruction ---
    if run_gpu:
        print(f"Run_gpu = {run_gpu}")
        reconMrsi = bart(1, 'nufft -i -t -g', kspaceRSI, mrData2)
    else: 
        print(f"Run_gpu = {run_gpu}")
        reconMrsi = bart(1, 'nufft -i -t', kspaceRSI, mrData2)
    return reconMrsi


def run_bart_nufft_gpu(mrData, kSpaceTrj, para):
    """
    Python translation of MATLAB runBartRecon4 for BART NUFFT reconstruction.
    Args:
        mrData: numpy array, k-space data
        kSpaceTrj: dict with keys 'kxx', 'kyy'
        para: dict of parameters
    Returns:
        reconMrsi: reconstructed image (numpy array)
    """

    # --- Parameter defaults ---
    para = para.copy()
    para.setdefault('oprPath', None)
    para.setdefault('isWaterRef', False)
    para.setdefault('dbgOn', False)
    para.setdefault('csReg', 0.05)
    para.setdefault('csMap', None)
    para.setdefault('itr', 50)
    para.setdefault('CGitr', 5)
    para.setdefault('useNorm', False)
    para.setdefault('applyFilter', '')
    para.setdefault('reconType', None)
    para.setdefault('kFac', 1)
    para.setdefault('doK0Cor', False)
    para.setdefault('k0ZeroFill', 4)
    para.setdefault('doB0Cor', False)
    para.setdefault('B0refCh', 27)
    para.setdefault('B0corMode', 'lin')
    para.setdefault('acqConf', {'H1offset': 4.7})
    para.setdefault('mSize', mrData.shape[1] if mrData.ndim > 1 else mrData.shape[0])

    # --- Normalize k-space trajectory ---
    # print kspace_traj
    # print(f"Normalizing k-space trajectory with max value: {np.max(np.abs(kSpaceTrj['kxx']))}")
    kxx = kSpaceTrj['kxx'] / np.max(np.abs(kSpaceTrj['kxx'])) * para['mSize']/2 * para['kFac']
    kyy = kSpaceTrj['kyy'] / np.max(np.abs(kSpaceTrj['kyy'])) * para['mSize']/2 * para['kFac']

    # print(f"Normalized k-space trajectory shapes: kxx={kxx.shape}, kyy={kyy.shape}")
    kxx = kxx.reshape(1, kxx.shape[0], kxx.shape[1])
    kyy = kyy.reshape(1, kyy.shape[0], kyy.shape[1])

    # print(f"Added a 3rd dimension to kxx and kyy: kxx={kxx.shape}, kyy={kyy.shape}")
    kspaceRSI = np.concatenate([kxx, kyy], axis=0)
    # print(f"kspace RSI shape: {kspaceRSI.shape}")
    kspaceRSI = np.concatenate([kspaceRSI, np.zeros((1, kxx.shape[1], kxx.shape[2]))], axis=0)
    # print(f"kspace RSI shape with 3rd axis: {kspaceRSI.shape}")
    # print(f"Last column of kspaceRSI (should be zeros): {kspaceRSI[-1, :, :]}")

    # --- Reshape mrData ---
    # MATLAB: mrData = reshape( mrData, [1 size(mrData,1) size(mrData,2) size(mrData,3) 1 1 1 1 1 1 size(mrData,4)] );
    # For typical 2D data, this becomes (1, Nx, Ny, 1, 1, 1, 1, 1, 1, 1, 1)
    shape = [1] + list(mrData.shape) + [1]*(11-len(mrData.shape)-1)
    # print(f"Shape {shape}")
    mrData_reshaped = mrData.reshape(shape)
    mrData2 = np.squeeze(mrData_reshaped)
    mrData2 = mrData2[np.newaxis, :, :]

    # --- BART NUFFT reconstruction ---
    reconMrsi = bart(1, 'nufft -i -g', kspaceRSI, mrData2)
    return reconMrsi

def build_para(config):
    """
    Build the para dictionary for run_bart_nufft from the config object.
    """
    para = {
        'oprPath': getattr(config.reconstruction, 'oprPath', None),
        'isWaterRef': getattr(config.reconstruction, 'is_water_ref', False),
        'dbgOn': getattr(config.reconstruction, 'debug_on', False),
        'csReg': getattr(config.reconstruction, 'cs_reg', 0.05),
        'csMap': None,
        'itr': getattr(config.reconstruction, 'iterations', 50),
        'CGitr': getattr(config.reconstruction, 'cg_iterations', 5),
        'useNorm': getattr(config.reconstruction, 'use_norm', False),
        'applyFilter': getattr(config.reconstruction, 'apply_filter', ''),
        'reconType': getattr(config.reconstruction, 'recon_type', None),
        'kFac': getattr(config.reconstruction, 'k_factor', 1),
        'doK0Cor': getattr(config.reconstruction, 'do_k0_correction', False),
        'k0ZeroFill': getattr(config.reconstruction, 'k0_zero_fill', 4),
        'doB0Cor': getattr(config.reconstruction, 'do_b0_correction', False),
        'B0refCh': getattr(config.reconstruction, 'b0_ref_channel', 27),
        'B0corMode': getattr(config.reconstruction, 'b0_cor_mode', 'lin'),
        'acqConf': {'H1offset': getattr(getattr(config, 'acqConf', {}), 'H1offset', 4.7)},
        'mSize': getattr(config.trajectory, 'res', 256)
    }
    return para

def flip_recon(recon_img: np.ndarray) -> np.ndarray:
    """
    Flip the reconstructed image along axis 0 and axis 1 (equivalent to MATLAB flip(img,1), flip(img,2)).
    Args:
        recon_img: numpy array (reconstructed image)
    Returns:
        Flipped image
    """
    recon_img = np.flip(recon_img, axis=0)
    recon_img = np.flip(recon_img, axis=1)
    return recon_img

def rescale_recon_img(recon_img, Nx, Ny, res):
    """
    Flip and rescale the reconstructed image as in the MATLAB pipeline.
    Args:
        recon_img: numpy array (reconstructed image)
        Nx: int, image size x-axis
        Ny: int, image size y-axis
        res: int, reconstruction resolution
    Returns:
        Rescaled and flipped image
    """
    recon_img = flip_recon(recon_img)
    area_ratio = (Nx * Ny) / (res * res)
    bart_ratio = area_ratio * res
    recon_img = recon_img / bart_ratio
    return recon_img

def plot_results(ground_truth: np.ndarray, 
                reconstructed: np.ndarray,
                kspace_data: Optional[np.ndarray] = None,
                trajectory: Optional[np.ndarray] = None,
                save_path: Optional[str] = None,
                show: bool = True) -> None:
    """
    Plot reconstruction results
    
    Args:
        ground_truth: Original image
        reconstructed: Reconstructed image  
        kspace_data: K-space data (optional)
        trajectory: Trajectory data (optional)
        save_path: Path to save plot (optional)
        show: Whether to display plot
    """
    try:
        # Determine subplot layout
        n_plots = 2
        if kspace_data is not None:
            n_plots += 1
        if trajectory is not None:
            n_plots += 1
            
        if n_plots == 2:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        elif n_plots == 3:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
        axes = np.atleast_1d(axes).flatten()
        
        # Plot ground truth
        im1 = axes[0].imshow(ground_truth, cmap='gray')
        axes[0].set_title('Ground Truth (Downscaled)')
        axes[0].axis('equal')
        axes[0].axis('off')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot reconstruction
        im2 = axes[1].imshow(np.abs(reconstructed), cmap='gray')
        axes[1].set_title('BART Rosette NUFFT Reconstruction')
        axes[1].axis('equal') 
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1])
        
        plot_idx = 2
        
        # Plot k-space data if provided
        if kspace_data is not None and plot_idx < len(axes):
            im3 = axes[plot_idx].imshow(np.log(np.abs(kspace_data) + 1e-10), cmap='hot')
            axes[plot_idx].set_title('K-space Data (Log Scale)')
            axes[plot_idx].axis('off')
            plt.colorbar(im3, ax=axes[plot_idx])
            plot_idx += 1
            
        # Plot trajectory if provided
        if trajectory is not None and plot_idx < len(axes):
            axes[plot_idx].plot(np.real(trajectory.flatten()), np.imag(trajectory.flatten()), 
                               'b-', alpha=0.7, linewidth=0.5)
            axes[plot_idx].set_title('Rosette Trajectory')
            axes[plot_idx].set_xlabel('kx')
            axes[plot_idx].set_ylabel('ky')
            axes[plot_idx].axis('equal')
            axes[plot_idx].grid(True, alpha=0.3)
            
        plt.tight_layout()
        
        # Save plot if requested
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to: {save_path}")
            
        # Show plot if requested
        if show:
            plt.show()
        else:
            plt.close()
            
    except Exception as e:
        logger.error(f"Error plotting results: {str(e)}")
        raise

def plot_heatmap(img, save_path, title=None, vmin=None, vmax=None, max_ssim_value=None, show=False):
    plt.figure(figsize=(8,6))
    im1 = plt.imshow(img, cmap='jet', aspect='equal', interpolation='nearest', vmin=vmin, vmax=vmax)
    if title:
        plt.title(title, fontweight='bold')
    plt.xlabel('X pixels')
    plt.ylabel('Y pixels')
    plt.colorbar(im1, shrink=0.8)
    plt.tight_layout()
    plt.savefig(save_path)
    if show:
        plt.show()
    else:
        plt.close()

def save_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Save reconstruction results to files
    
    Args:
        results: Dictionary containing results to save
        output_dir: Output directory path
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each result
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                np.save(os.path.join(output_dir, f"{key}.npy"), value)
                logger.info(f"Saved {key} to {output_dir}")
                
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        raise

def downsample_image(ground_truth_img: np.ndarray, size: tuple = (50, 50)) -> np.ndarray:
    """
    Downsample the ground truth image to the specified size using bilinear interpolation.
    Args:
        ground_truth_img: numpy array, original image
        size: tuple, target size (default (50, 50))
    Returns:
        ground_truth_img_down: downsampled image
    """
    ground_truth_img_down = resize(
        ground_truth_img,
        size,
        order=1,  # bilinear interpolation
        preserve_range=False
    )
    return ground_truth_img_down

def plot_comparison(ground_truth_img_down_norm: np.ndarray, img_reconstructed: np.ndarray, save_path: str = None, show: bool = False):
    """
    Create a 1x2 subplot comparing the normalized downsampled ground truth and the reconstructed image.
    Both subplots use the same colorbar scale (from the ground truth image).
    Args:
        ground_truth_img_down_norm: numpy array, normalized downsampled ground truth image
        img_reconstructed: numpy array, reconstructed image
        save_path: optional path to save the figure
        show: whether to display the plot
    """
    plt.figure(figsize=(10, 5))
    vmin = np.min(ground_truth_img_down_norm)
    vmax = np.max(ground_truth_img_down_norm)

    plt.subplot(1, 2, 1)
    im1 = plt.imshow(ground_truth_img_down_norm, cmap='jet', aspect='equal', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title('Base Image')
    plt.axis('equal')
    plt.axis('tight')
    plt.colorbar(im1, shrink=0.8)

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(np.abs(img_reconstructed), cmap='jet', aspect='equal', interpolation='nearest', vmin=vmin, vmax=vmax)
    plt.title('Reconstructed Image')
    plt.axis('equal')
    plt.axis('tight')
    plt.colorbar(im2, shrink=0.8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()

def print_pipeline_timing_summary(step1_time: float, step2_time: float, step3_time: float, 
                                 step4_time: float, step5_time: float, step6_time: float, 
                                 total_pipeline_time: float, run_gpu: bool) -> None:
    """
    Print a formatted summary of pipeline timing for all steps.
    
    Args:
        step1_time: Time for Step 1 (Image Loading) in seconds
        step2_time: Time for Step 2 (Cartesian k-space) in seconds
        step3_time: Time for Step 3 (Rosette Trajectory) in seconds
        step4_time: Time for Step 4 (k-space Sampling) in seconds
        step5_time: Time for Step 5 (BART Reconstruction) in seconds
        step6_time: Time for Step 6 (Metrics Calculation) in seconds
        total_pipeline_time: Total pipeline time in seconds
    """
    print(f"\n{'='*60}")
    print(f"          PIPELINE TIMING SUMMARY")
    print(f"{'='*60}")
    print(f"Step 1 (Image Loading):           {step1_time:.3f} seconds")
    print(f"Step 2 (Cartesian k-space):       {step2_time:.3f} seconds")
    print(f"Step 3 (Rosette Trajectory):      {step3_time:.3f} seconds")
    print(f"Step 4 (k-space Sampling):        {step4_time:.3f} seconds")
    if run_gpu:
        print(f"Step 5 (BART Recon GPU):          {step5_time:.3f} seconds")
    else:
        print(f"Step 5 (BART Recon NO GPU):       {step5_time:.3f} seconds")
    
    print(f"Step 6 (Metrics Calculation):     {step6_time:.3f} seconds")
    print(f"{'='*60}")
    print(f"TOTAL PIPELINE TIME:              {total_pipeline_time:.3f} seconds")
    print(f"{'='*60}\n")


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