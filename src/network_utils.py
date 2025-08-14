import numpy as np
import matplotlib.pyplot as plt
import sys


# def run_bart_nufft(mrData, kSpaceTrj, para, run_gpu=False):
#     """
#     Python translation of MATLAB runBartRecon4 for BART NUFFT reconstruction.
#     Args:
#         mrData: numpy array, k-space data
#         kSpaceTrj: dict with keys 'kxx', 'kyy'
#         para: dict of parameters
#     Returns:
#         reconMrsi: reconstructed image (numpy array)
#     """

#     # --- Parameter defaults ---
#     para = para.copy()
#     para.setdefault('oprPath', None)
#     para.setdefault('isWaterRef', False)
#     para.setdefault('dbgOn', False)
#     para.setdefault('csReg', 0.05)
#     para.setdefault('csMap', None)
#     para.setdefault('itr', 50)
#     para.setdefault('CGitr', 5)
#     para.setdefault('useNorm', False)
#     para.setdefault('applyFilter', '')
#     para.setdefault('reconType', None)
#     para.setdefault('kFac', 1)
#     para.setdefault('doK0Cor', False)
#     para.setdefault('k0ZeroFill', 4)
#     para.setdefault('doB0Cor', False)
#     para.setdefault('B0refCh', 27)
#     para.setdefault('B0corMode', 'lin')
#     para.setdefault('acqConf', {'H1offset': 4.7})
#     para.setdefault('mSize', mrData.shape[1] if mrData.ndim > 1 else mrData.shape[0])

#     # --- Normalize k-space trajectory ---
#     # print kspace_traj
#     # print(f"Normalizing k-space trajectory with max value: {np.max(np.abs(kSpaceTrj['kxx']))}")
#     kxx = kSpaceTrj['kxx'] / np.max(np.abs(kSpaceTrj['kxx'])) * para['mSize']/2 * para['kFac']
#     kyy = kSpaceTrj['kyy'] / np.max(np.abs(kSpaceTrj['kyy'])) * para['mSize']/2 * para['kFac']

#     # print(f"Normalized k-space trajectory shapes: kxx={kxx.shape}, kyy={kyy.shape}")
#     kxx = kxx.reshape(1, kxx.shape[0], kxx.shape[1])
#     kyy = kyy.reshape(1, kyy.shape[0], kyy.shape[1])

#     # print(f"Added a 3rd dimension to kxx and kyy: kxx={kxx.shape}, kyy={kyy.shape}")
#     kspaceRSI = np.concatenate([kxx, kyy], axis=0)
#     # print(f"kspace RSI shape: {kspaceRSI.shape}")
#     kspaceRSI = np.concatenate([kspaceRSI, np.zeros((1, kxx.shape[1], kxx.shape[2]))], axis=0)
#     # print(f"kspace RSI shape with 3rd axis: {kspaceRSI.shape}")
#     # print(f"Last column of kspaceRSI (should be zeros): {kspaceRSI[-1, :, :]}")

#     # --- Reshape mrData ---
#     # MATLAB: mrData = reshape( mrData, [1 size(mrData,1) size(mrData,2) size(mrData,3) 1 1 1 1 1 1 size(mrData,4)] );
#     # For typical 2D data, this becomes (1, Nx, Ny, 1, 1, 1, 1, 1, 1, 1, 1)
#     shape = [1] + list(mrData.shape) + [1]*(11-len(mrData.shape)-1)
#     # print(f"Shape {shape}")
#     mrData_reshaped = mrData.reshape(shape)
#     mrData2 = np.squeeze(mrData_reshaped)
#     mrData2 = mrData2[np.newaxis, :, :]

#     # --- BART NUFFT reconstruction ---
#     if run_gpu:
#         print(f"Run_gpu = {run_gpu}")
#         reconMrsi = bart(1, 'nufft -i -t -g', kspaceRSI, mrData2)
#     else: 
#         print(f"Run_gpu = {run_gpu}")
#         reconMrsi = bart(1, 'nufft -i -t', kspaceRSI, mrData2)
#     return reconMrsi


def shift_trajectory(kx, ky):
    """
    Shift a closed-loop 2D trajectory by (0, -Ky(Kx=0)).
    
    Parameters
    ----------
    kx : array_like
        Array of kx coordinates.
    ky : array_like
        Array of ky coordinates.
    
    Returns
    -------
    kx_shift : np.ndarray
        Shifted kx coordinates (same as input).
    ky_shift : np.ndarray
        Shifted ky coordinates after translation.
    """
    kx = np.array(kx)
    ky = np.array(ky)
    
    # Find indices where kx == 0 (or closest to zero)
    idx = np.argmin(np.abs(kx))
    ky_at_kx0 = ky[0] #TAKE THE FIRST VALUE AT KX=0
    
    # Apply translation
    kx_shift = kx.copy()
    ky_shift = ky - ky_at_kx0
    
    return kx_shift, ky_shift


def rotate_traj(kx, ky, n_rotation=79):
    """
    Generate rotated versions of a 2D trajectory.

    Parameters
    ----------
    kx : array_like
        1D array of kx coordinates of the trajectory.
    ky : array_like
        1D array of ky coordinates of the trajectory.
    n_rotation : int, optional
        Number of rotated trajectories to generate (default 79).

    Returns
    -------
    kspaceTrj : dict
        Dictionary with:
        'kxx' : np.ndarray of shape (len(kx), n_rotation)
        'kyy' : np.ndarray of shape (len(kx), n_rotation)
    """
    # Load analytical trajectory for debugging
    # first_rosette = np.load('tmp/first_rosette.npy')
    # kx = np.real(first_rosette)
    # ky = np.imag(first_rosette)
    kx = np.array(kx)
    ky = np.array(ky)
    n_points = len(kx)

    # print(f"Number of points in trajectory: {n_points}")
    kxx = np.zeros((n_rotation, n_points))
    kyy = np.zeros((n_rotation, n_points))

    # Rotation angles evenly spaced from 0 to 2π
    angles = np.linspace(0, 2 * np.pi, n_rotation, endpoint=False)

    for i, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        kxx[i, :] = cos_t * kx - sin_t * ky
        kyy[i, :] = sin_t * kx + cos_t * ky

    # print(f"length of kxx: {len(kxx)}"
    #       f"\nlength of kyy: {len(kyy)}"
    #       f"\nshape of kxx: {kxx.shape}"
    #       f"\nshape of kyy: {kyy.shape}")
    # print(f"First point of every trajectory: {kxx[50,:]} \t {kyy[50,:]}")
    kspaceTrj = {'kxx': kxx, 'kyy': kyy}

    # print(f"\n\n\nFirst point of every trajectories: {kxx[0,:]} \t {kyy[0,:]} \n\n\n")


    return kspaceTrj

def complex_traj(kspaceTrj):
    """
    Convert rotated k-space trajectories into a flattened complex vector.

    Parameters
    ----------
    kspaceTrj : dict
        Dictionary with 'kxx' and 'kyy' arrays of shape (n_points, n_rotation).

    Returns
    -------
    ktraj_complex : np.ndarray
        Complex 1D array of shape (n_points * n_rotation, ),
        where each entry is kxx + 1j * kyy.
    """
    kxx = np.array(kspaceTrj['kxx'])
    kyy = np.array(kspaceTrj['kyy'])

    # Create complex matrix
    ktraj_complex = kxx + 1j * kyy

    # Flatten to 1D
    return ktraj_complex.ravel()

def plot_trajs(kspaceTrj):
    """
    Plot all trajectories from the rotated k-space data.

    Parameters
    ----------
    kspaceTrj : dict
        Dictionary with 'kxx' and 'kyy' arrays of shape (n_points, n_rotation).
    """
    kxx = kspaceTrj['kxx']
    kyy = kspaceTrj['kyy']

    plt.figure(figsize=(6, 6))
    for i in range(kxx.shape[1]):
        plt.plot(kxx[:, i], kyy[:, i], lw=0.8)

    plt.xlabel("kx")
    plt.ylabel("ky")
    plt.axis("equal")
    plt.title("Rotated k-space Trajectories")
    plt.grid(True)
    plt.show()



def max_rotated_deriv(dx, dy, ddx, ddy, n_rotation):
    """
    Vectorized computation of the maximum absolute derivative values 
    across all rotations.

    Parameters
    ----------
    dx, dy, ddx, ddy : array_like
        Derivative components of the trajectory (1D arrays of length n_points).
    n_rotation : int
        Number of rotations to consider.

    Returns
    -------
    max_dx, max_dy, max_ddx, max_ddy : float
        Maximum absolute values across all rotations.
    """
    dx = np.array(dx)[:, None]   # shape (n_points, 1)
    dy = np.array(dy)[:, None]
    ddx = np.array(ddx)[:, None]
    ddy = np.array(ddy)[:, None]

    # Rotation angles
    angles = np.linspace(0, 2 * np.pi, n_rotation, endpoint=False)
    cos_t = np.cos(angles)[None, :]  # shape (1, n_rotation)
    sin_t = np.sin(angles)[None, :]

    # Rotate first derivatives
    dx_rot = cos_t * dx - sin_t * dy
    dy_rot = sin_t * dx + cos_t * dy

    # Rotate second derivatives
    ddx_rot = cos_t * ddx - sin_t * ddy
    ddy_rot = sin_t * ddx + cos_t * ddy

    # Compute maxima across both axes
    max_dx_val = np.max(np.abs(dx_rot))
    max_dy_val = np.max(np.abs(dy_rot))
    max_ddx_val = np.max(np.abs(ddx_rot))
    max_ddy_val = np.max(np.abs(ddy_rot))

    return max_dx_val, max_dy_val, max_ddx_val, max_ddy_val


def demo_trajectory_utilities(kx, ky, save_path, run_folder=None, n_rotations=79):
    """
    Apply trajectory utilities (shift, rotate) to kx, ky trajectories and perform image reconstruction
    
    Args:
        kx: numpy array of kx coordinates
        ky: numpy array of ky coordinates  
        save_path: directory to save plots
        run_folder: optional subfolder for organized saving
        n_rotations: number of rotated trajectories to generate
    
    Returns:
        dict: Contains trajectory data and reconstruction results
    """
    import numpy as np
    import os
    import torch
    from src.bart_config import create_default_config
    from src.bart_interpolation_fct import fast_kspace_interpolation_v3
    from src.bart_utils import run_bart_nufft, build_para, rescale_recon_img, plot_comparison
    from src.plots import plot_trajectory_shift_comparison, plot_rotated_trajectories, plot_trajectory_utilities_combined
    
    print("\n=== TRAJECTORY UTILITIES DEMO ===")
    
    # Create save directory
    # if run_folder:
    #     plot_save_dir = os.path.join(save_path, run_folder)
    # else:
    #     plot_save_dir = save_path
    if run_folder:
        plot_save_dir = run_folder  # Use the run_folder directly
    else:
        plot_save_dir = save_path
    os.makedirs(plot_save_dir, exist_ok=True)
    
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
    kSpaceTrj = rotate_traj(kx_shifted, ky_shifted, n_rotation=n_rotations)

    # # print the first 5 elements of the dictionary
    # print(f"   First 5 kxx values: {kSpaceTrj['kxx'][:5, :5]}")
    # print(f"   First 5 kyy values: {kSpaceTrj['kyy'][:5, :5]}")
    # tmp_traj = np.load('tmp/rosette_traj_analytical.npy')
    # tmp_traj = tmp_traj.reshape(128,79)
    # kSpaceTrj = {
    #     'kxx': np.real(tmp_traj),
    #     'kyy': np.imag(tmp_traj)
    # }

        # print the first 5 elements of the dictionary
    # print(f"   First 5 kxx values: {kSpaceTrj['kxx'][:5, :5]}")
    # print(f"   First 5 kyy values: {kSpaceTrj['kyy'][:5, :5]}")


    print(f"   Generated {n_rotations} rotated trajectories")
    
    # 4. Create plots using plotting functions
    print("4. Creating trajectory plots...")
    
    # Plot shift comparison
    shift_plot_path = plot_trajectory_shift_comparison(kx, ky, kx_shifted, ky_shifted, plot_save_dir)
    print(f"   Saved shift comparison to: {shift_plot_path}")
    
    # Plot rotated trajectories
    rotated_plot_path = plot_rotated_trajectories(kSpaceTrj, plot_save_dir)
    print(f"   Saved rotated trajectories to: {rotated_plot_path}")
    
    # Plot combined view
    combined_plot_path = plot_trajectory_utilities_combined(kx, ky, kx_shifted, ky_shifted, kSpaceTrj, plot_save_dir)
    print(f"   Saved combined view to: {combined_plot_path}")
    
    # 5. Prepare for image reconstruction
    print("5. Performing image reconstruction...")
    
    config = create_default_config()
    os.makedirs(config.tmp_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)
    
    # Load k-space data if available
    kspace_file_path = os.path.join(config.tmp_dir, 'kspace_cartesian.npy')
    if os.path.exists(kspace_file_path):
        kspace_data = np.load(kspace_file_path)
        print(f"   Loaded k-space data from {kspace_file_path}")
        print(f"   K-space data shape: {kspace_data.shape}")
        
        # Create complex trajectory
        rosette_traj = complex_traj(kSpaceTrj)
        print(f"   Generated rosette trajectory with shape: {rosette_traj.shape}")
        
        # Interpolate k-space data
        FOV = 224  # Field of View in mm (from run.py)
        kspace_sampled = fast_kspace_interpolation_v3(kspace_data, rosette_traj, FOV)
        
        # Prepare for BART reconstruction
        para = build_para(config)
        kxx = kSpaceTrj['kxx']
        kyy = kSpaceTrj['kyy']
        mr_data = kspace_sampled.reshape(kxx.shape)
        
        # Run BART NUFFT reconstruction
        run_gpu = True if torch.cuda.is_available() else False
        recon_img = run_bart_nufft(mr_data, kSpaceTrj, para, run_gpu)
        
        print(f"   Reconstructed image shape: {recon_img.shape}")
        print(f"   Reconstructed image dtype: {recon_img.dtype}")
        
        # Rescale reconstruction
        RES = 50  # Resolution from run.py
        [Nx, Ny] = kspace_data.shape[0], kspace_data.shape[1]
        recon_img = rescale_recon_img(recon_img, Nx, Ny, RES)
        
        # Save reconstruction
        np.save(os.path.join(config.tmp_dir, 'recon_img.npy'), recon_img)
        
        # Compare with ground truth if available
        ground_truth_img_down_path = os.path.join(config.tmp_dir, 'ground_truth_img_down.npy')
        if os.path.exists(ground_truth_img_down_path):
            ground_truth_img_down = np.load(ground_truth_img_down_path)
            plot_comparison(ground_truth_img_down, recon_img,
                save_path='plots_bart/comparison.png',
                show=False
            )
            print(f"   Saved reconstruction comparison to: plots_bart/comparison.png")
        
        reconstruction_data = {
            'recon_img': recon_img,
            'kspace_sampled': kspace_sampled,
            'rosette_traj': rosette_traj
        }
    else:
        print(f"   Warning: K-space data not found at {kspace_file_path}")
        reconstruction_data = None
    
    print(f"=== TRAJECTORY UTILITIES DEMO COMPLETED ===\n")
    
    # Return all computed data
    return {
        'kx_original': kx,
        'ky_original': ky,
        'kx_shifted': kx_shifted,
        'ky_shifted': ky_shifted,
        'kSpaceTrj': kSpaceTrj,
        'plot_paths': {
            'shift_comparison': shift_plot_path,
            'rotated_trajectories': rotated_plot_path,
            'combined_view': combined_plot_path
        },
        'reconstruction': reconstruction_data
    }