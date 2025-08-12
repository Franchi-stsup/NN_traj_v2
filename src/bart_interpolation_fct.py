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

def fast_kspace_interpolation_v3(kspace_data, rosette_traj, FoV): #, Kmax_res):
    """
    Optimized griddata with spatial filtering
    """
    print("  Using spatially filtered griddata...")
    
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
    print(f"  Using {len(nearby_indices)} grid points (vs {len(grid_points)} original)")
    
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