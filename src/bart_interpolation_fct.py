import numpy as np
from scipy.interpolate import griddata, RegularGridInterpolator
from scipy.spatial import cKDTree
import time

def fast_kspace_interpolation_v1(kspace_data, rosette_traj, FoV, Kmax_res):
    """
    Optimized interpolation using smaller grid based on resolution requirements
    """
    print("  Using reduced grid interpolation...")
    
    # Use smaller grid based on resolution requirements
    Kmax_reduced = Kmax_res * 1.2
    
    # Determine grid size based on desired resolution
    # Rule of thumb: grid spacing should be fine enough for your trajectory
    N_reduced = int(2 * Kmax_reduced / (Kmax_res / min(kspace_data.shape)) * 2)
    N_reduced = max(64, min(N_reduced, 512))  # Reasonable bounds
    
    print(f"  Using reduced grid: {N_reduced}x{N_reduced} (vs original {kspace_data.shape})")
    
    # Create reduced grid
    kx_grid = np.linspace(-Kmax_reduced, Kmax_reduced, N_reduced)
    ky_grid = np.linspace(-Kmax_reduced, Kmax_reduced, N_reduced)
    KX_grid, KY_grid = np.meshgrid(kx_grid, ky_grid)
    
    # Interpolate original data onto reduced grid first
    Nx, Ny = kspace_data.shape
    Kmax_orig = Nx / (FoV * 1e-3) / 2
    
    kx_orig = np.linspace(-Kmax_orig, Kmax_orig, Nx)
    ky_orig = np.linspace(-Kmax_orig, Kmax_orig, Ny)
    
    # Use RegularGridInterpolator for the first step (much faster)
    interp_func = RegularGridInterpolator(
        (kx_orig, ky_orig), kspace_data, 
        method='linear', bounds_error=False, fill_value=0
    )
    
    # Sample on reduced grid
    grid_points = np.vstack((KX_grid.flatten(), KY_grid.flatten())).T
    kspace_reduced = interp_func(grid_points).reshape(N_reduced, N_reduced)
    
    # Now interpolate from reduced grid to trajectory
    points = np.vstack((KX_grid.flatten(), KY_grid.flatten())).T
    values = kspace_reduced.flatten()
    
    kx_traj = np.real(rosette_traj)
    ky_traj = np.imag(rosette_traj)
    interp_points = np.vstack((kx_traj, ky_traj)).T
    
    kspace_sampled = griddata(points, values, interp_points, 
                             method='linear', fill_value=0)
    
    return kspace_sampled

def fast_kspace_interpolation_v2(kspace_data, rosette_traj, FoV, Kmax_res):
    """
    Direct RegularGridInterpolator approach (fastest for regular->irregular)
    """
    print("  Using direct RegularGridInterpolator...")
    
    Nx, Ny = kspace_data.shape
    Kmax_orig = Nx / (FoV * 1e-3) / 2
    
    # Original grid coordinates
    kx_orig = np.linspace(-Kmax_orig, Kmax_orig, Nx)
    ky_orig = np.linspace(-Kmax_orig, Kmax_orig, Ny)
    
    # Create interpolator
    interp_func = RegularGridInterpolator(
        (kx_orig, ky_orig), kspace_data,
        method='linear', bounds_error=False, fill_value=0
    )
    
    # Trajectory points
    kx_traj = np.real(rosette_traj)
    ky_traj = np.imag(rosette_traj)
    interp_points = np.vstack((kx_traj, ky_traj)).T
    
    # Direct interpolation
    kspace_sampled = interp_func(interp_points)
    
    return kspace_sampled

def fast_kspace_interpolation_v3(kspace_data, rosette_traj, FoV=224): #, Kmax_res):
    """
    Optimized griddata with spatial filtering
    """
    # print("  Using spatially filtered griddata...")
    
    Nx, Ny = kspace_data.shape
    Kmax_orig = Nx / (FoV * 1e-3) / 2
    # Kmax_reduced = Kmax_res * 1.2
    
    # Create original grid
    kx_orig = np.linspace(-Kmax_orig, Kmax_orig, Nx)
    ky_orig = np.linspace(-Kmax_orig, Kmax_orig, Ny)
    KX_orig, KY_orig = np.meshgrid(kx_orig, ky_orig)
    
    # Trajectory points
    kx_traj = np.real(rosette_traj)
    ky_traj = np.imag(rosette_traj)
    
    # Filter grid points to only those near trajectory (spatial optimization)
    trajectory_points = np.vstack((kx_traj, ky_traj)).T
    grid_points = np.vstack((KX_orig.flatten(), KY_orig.flatten())).T
    
    # Build KDTree for efficient neighbor finding
    tree = cKDTree(grid_points)
    
    # Find grid points within reasonable distance of trajectory
    max_dist = np.sqrt((kx_orig[1] - kx_orig[0])**2 + (ky_orig[1] - ky_orig[0])**2) * 3
    
    # For each trajectory point, find nearby grid points
    nearby_indices = set()
    for traj_point in trajectory_points:
        indices = tree.query_ball_point(traj_point, max_dist)
        nearby_indices.update(indices)
    
    nearby_indices = list(nearby_indices)
    # print(f"  Using {len(nearby_indices)} grid points (vs {len(grid_points)} original)")
    
    # Use only nearby points for interpolation
    filtered_points = grid_points[nearby_indices]
    filtered_values = kspace_data.flatten()[nearby_indices]
    
    interp_points = np.vstack((kx_traj, ky_traj)).T
    kspace_sampled = griddata(filtered_points, filtered_values, interp_points,
                             method='linear', fill_value=0)
    
    return kspace_sampled

def benchmark_interpolation_methods(kspace_data, rosette_traj, FoV, Kmax_res):
    """
    Benchmark different interpolation methods
    """
    methods = [
        ("Original griddata", lambda: original_method(kspace_data, rosette_traj, FoV)),
        ("Reduced grid", lambda: fast_kspace_interpolation_v1(kspace_data, rosette_traj, FoV, Kmax_res)),
        ("RegularGridInterpolator", lambda: fast_kspace_interpolation_v2(kspace_data, rosette_traj, FoV, Kmax_res)),
        ("Spatial filtering", lambda: fast_kspace_interpolation_v3(kspace_data, rosette_traj, FoV, Kmax_res))
    ]
    
    results = {}
    
    for name, method in methods:
        print(f"\nTesting {name}:")
        start_time = time.time()
        try:
            result = method()
            end_time = time.time()
            results[name] = {
                'time': end_time - start_time,
                'result': result,
                'success': True
            }
            print(f"  Time: {end_time - start_time:.3f}s")
            print(f"  Output shape: {result.shape}")
        except Exception as e:
            print(f"  Failed: {e}")
            results[name] = {'success': False, 'error': str(e)}
    
    return results

import torch
import torch.nn.functional as F

@torch.no_grad()
def _compute_kmax_from_fov(H, W, FoV_mm: float):
    # Match your original formula; uses the readout size (W) for Kmax.
    # If you prefer min(H,W), replace W with min(H, W).
    return (W / (FoV_mm * 1e-3)) / 2.0

def fast_kspace_interpolation_v3_torch(
    kspace_data: torch.Tensor,    # (H, W) real or complex64/complex128
    rosette_traj: torch.Tensor,   # (M,) complex tensor: kx + i ky  [in 1/m]
    FoV: float = 224.0,           # FoV in mm (same convention as your NumPy code)
    clamp_to_grid: bool = True,   # clamp coords to [-1, 1] instead of zero-padding outside
):
    """
    Differentiable k-space interpolation using bilinear sampling (grid_sample).

    Args:
        kspace_data: (H, W) tensor. If complex, use complex dtype; if real, it's treated as a single channel.
                     H corresponds to ky (rows), W to kx (cols) on a Cartesian grid.
        rosette_traj: (M,) complex tensor with kx in real part and ky in imag part, in units of 1/m.
        FoV: Field of view in millimeters. Used to compute Kmax so physical k-space units map to grid.
        clamp_to_grid: If True, coordinates outside [-1, 1] are softly clamped (keeps grads).
                       If False, we use padding_mode='zeros' (samples become exactly 0 outside).

    Returns:
        sampled: (M,) tensor matching the dtype (real/complex) of kspace_data.
    """
    if not torch.is_tensor(kspace_data) or not torch.is_tensor(rosette_traj):
        raise TypeError("Inputs must be PyTorch tensors.")

    # Shapes
    H, W = kspace_data.shape[-2:]

    # Compute Kmax (matches your original Nx/FoV/2 formula; using W≡Nx/readout)
    Kmax = _compute_kmax_from_fov(H, W, FoV)

    # Split trajectory (expects complex traj: kx + i ky)
    if not torch.is_complex(rosette_traj):
        raise ValueError("rosette_traj must be a complex tensor with kx in real and ky in imag parts.")
    kx = rosette_traj.real   # (M,)
    ky = rosette_traj.imag   # (M,)

    # Map physical k-space coords to normalized grid_sample coords in [-1, 1].
    # align_corners=True ⇒ -Kmax ↦ -1 and +Kmax ↦ +1 exactly.
    x_norm = kx / Kmax
    y_norm = ky / Kmax

    if clamp_to_grid:
        # Smooth clamp to keep gradients (as opposed to hard zero outside)
        x_norm = torch.clamp(x_norm, -1.0, 1.0)
        y_norm = torch.clamp(y_norm, -1.0, 1.0)

    # Build grid for grid_sample: shape (N=1, Hout=1, Wout=M, 2)
    # grid[..., 0] = x, grid[..., 1] = y
    grid = torch.stack((x_norm, y_norm), dim=-1).view(1, 1, -1, 2)

    # Prepare input as (N=1, C, H, W). Handle complex by 2 channels (Re, Im).
    if torch.is_complex(kspace_data):
        data_2ch = torch.view_as_real(kspace_data)  # (H, W, 2)
        data_2ch = data_2ch.permute(2, 0, 1).unsqueeze(0)  # (1, 2, H, W)
        y = F.grid_sample(
            data_2ch, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True
        )  # (1, 2, 1, M)
        y = y.squeeze(0).squeeze(1).permute(1, 0)  # (M, 2)
        y = y.contiguous()  # Ensure contiguous memory layout for view_as_complex
        sampled = torch.view_as_complex(y)         # (M,)
    else:
        data_1ch = kspace_data.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        y = F.grid_sample(
            data_1ch, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True
        )  # (1, 1, 1, M)
        sampled = y.squeeze(0).squeeze(0).squeeze(0)      # (M,)

    return sampled


def original_method(kspace_data, rosette_traj, FoV):
    """Original method for comparison"""
    Nx, Ny = kspace_data.shape
    Kmax = Nx / (FoV * 1e-3) / 2
    
    kx_grid = np.linspace(-Kmax, Kmax, Nx)
    ky_grid = np.linspace(-Kmax, Kmax, Ny)
    KX_grid, KY_grid = np.meshgrid(kx_grid, ky_grid)
    
    points = np.vstack((KX_grid.flatten(), KY_grid.flatten())).T
    values = kspace_data.flatten()
    
    kx_traj = np.real(rosette_traj)
    ky_traj = np.imag(rosette_traj)
    interp_points = np.vstack((kx_traj, ky_traj)).T
    
    return griddata(points, values, interp_points, method='linear', fill_value=0)

# # Example usage:
# if __name__ == "__main__":
#     # Example parameters
#     Nx = Ny = 256
#     FoV = 240  # mm
    
#     # Create example data
#     kspace_data = np.random.complex128((Nx, Ny))
    
#     # Create example rosette trajectory
#     n_points = 1000
#     t = np.linspace(0, 4*np.pi, n_points)
#     rosette_traj = 50 * np.exp(1j * t) * np.sin(3 * t)
    
#     Kmax_res = 30  # Example resolution limit
    
#     # Run benchmark
#     results = benchmark_interpolation_methods(kspace_data, rosette_traj, FoV, Kmax_res)