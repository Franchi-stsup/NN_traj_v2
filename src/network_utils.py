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
    ky_at_kx0 = ky[idx]
    
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
    kx = np.array(kx)
    ky = np.array(ky)
    n_points = len(kx)

    kxx = np.zeros((n_points, n_rotation))
    kyy = np.zeros((n_points, n_rotation))

    # Rotation angles evenly spaced from 0 to 2π
    angles = np.linspace(0, 2 * np.pi, n_rotation, endpoint=False)

    for i, theta in enumerate(angles):
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        kxx[:, i] = cos_t * kx - sin_t * ky
        kyy[:, i] = sin_t * kx + cos_t * ky
    kspaceTrj = {'kxx': kxx, 'kyy': kyy}

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