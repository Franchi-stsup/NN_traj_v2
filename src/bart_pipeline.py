"""
Main Pipeline Module for BART NUFFT Rosette Trajectory Reconstruction
===================================================================

This module implements the complete reconstruction pipeline including:
1. Image loading and preprocessing
2. Cartesian k-space sampling via NUFFT
3. Rosette trajectory generation  
4. K-space data sampling along trajectory
5. BART reconstruction
6. Results visualization and saving
"""

import time
import os
import logging
from scipy.ndimage import zoom
from scipy.interpolate import griddata
import numpy as np
import os
from src.bart_config import PipelineConfig, create_default_config
from src.bart_utils import *
from src.bart_interface import bart
from src.bart_metrics import calculate_all_metrics, print_metrics_summary
from src.bart_interpolation_fct import fast_kspace_interpolation_v1, fast_kspace_interpolation_v2, fast_kspace_interpolation_v3

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def run_bart_recon(load_kspace=False, run_gpu=False):
    """
    Implements the MATLAB pipeline in Python:
    1. Load image
    2. Cartesian k-space sampling via NUFFT (or load from file)
    3. Generate rosette trajectory
    4. Sample k-space data along trajectory
    5. BART reconstruction
    
    Args:
        load_kspace (bool): If True, load k-space data from tmp/kspace_cartesian.npy 
                           instead of recomputing it in Step 2
    """
    # Start total pipeline timer
    pipeline_start_time = time.time()
    
    config = create_default_config()
    os.makedirs(config.tmp_dir, exist_ok=True)
    os.makedirs(config.plots_dir, exist_ok=True)

    # --- Step 1: Load image ---
    print(f"\nStep 1: Loading image...")
    step1_start = time.time()
    image = load_image(config.input_image_path)
    [Nx, Ny] = image.shape[0], image.shape[1]  # Assuming square image for simplicity
    np.save(os.path.join(config.tmp_dir, 'image.npy'), image)
    step1_time = time.time() - step1_start

    # --- Step 2: Cartesian k-space sampling via NUFFT ---
    if load_kspace:
        print(f"\nStep 2: Loading k-space data from file...")
        step2_start = time.time()
        kspace_file_path = os.path.join(config.tmp_dir, 'kspace_cartesian.npy')
        if os.path.exists(kspace_file_path):
            kspace_data = np.load(kspace_file_path)
            print(f"  Loaded k-space data from {kspace_file_path}")
            print(f"  K-space data shape: {kspace_data.shape}")
        else:
            print(f"  ❌ K-space file not found at {kspace_file_path}")
            print(f"  Falling back to computing k-space data...")
            traj_cart = create_cartesian_trajectory(Nx, Ny)
            kSpace_Trj_BART, image_in_6D = bart_reformat_nufft(image, traj_cart)
            kspace_data = bart(1, 'nufft -i -t', kSpace_Trj_BART, image_in_6D)
            kspace_data = kspace_data * Nx  # Scale by image size
            np.save(kspace_file_path, kspace_data)
        step2_time = time.time() - step2_start
    else:
        print(f"\nStep 2: Cartesian k-space sampling via NUFFT...")
        step2_start = time.time()
        traj_cart = create_cartesian_trajectory(Nx, Ny)
        kSpace_Trj_BART, image_in_6D = bart_reformat_nufft(image, traj_cart)
        kspace_data = bart(1, 'nufft -i -t', kSpace_Trj_BART, image_in_6D)
        kspace_data = kspace_data * Nx  # Scale by image size
        np.save(os.path.join(config.tmp_dir, 'kspace_cartesian.npy'), kspace_data)
        step2_time = time.time() - step2_start

    # --- Step 3: Generate rosette trajectory ---
    print(f"\nStep 3: Generating rosette trajectory...")
    step3_start = time.time()
    FoV = config.trajectory.fov
    res = config.trajectory.res
    noPaddles = config.trajectory.no_paddles
    points_per_paddle = 100
    o1 = config.trajectory.o1
    o2 = config.trajectory.o2
    rosette_traj = generate_rosette_trajectory(FoV, res, noPaddles, points_per_paddle, o1, o2)
    np.save(os.path.join(config.tmp_dir, 'rosette_traj.npy'), rosette_traj)
    step3_time = time.time() - step3_start


    # --- Step 4: Sample k-space data along Rosette trajectory ---
    print(f"\nStep 4: Sampling k-space data along Rosette trajectory...")
    step4_start = time.time()

    try:
        # print(f"  Nx: {Nx}, Ny: {Ny}, FoV: {FoV}")

        # Kmax = Nx / (FoV * 1e-3) / 2
        # print(f"  Kmax: {Kmax}")

        kx_traj = np.real(rosette_traj)
        ky_traj = np.imag(rosette_traj)

        # kx_grid = np.linspace(-Kmax, Kmax, Nx)
        # ky_grid = np.linspace(-Kmax, Kmax, Ny)
        # # Meshgrid timing 
        # print(f"  Creating meshgrid...")
        # meshgrid_start = time.time()
        # KX_grid, KY_grid = np.meshgrid(kx_grid, ky_grid)
        # print(f"Meshgrid timing {time.time() - meshgrid_start:.3f} seconds")
        # # print(f"  KX_grid shape: {KX_grid.shape}, KY_grid shape: {KY_grid.shape}")
        # print(f"  Interpolating...")
        
        # # Flatten grids and k-space data for griddata
        # points = np.vstack((KX_grid.flatten(), KY_grid.flatten())).T
        # # Extract middle 256x256 points from the grid

        # values = kspace_data.flatten()
        # interp_points = np.vstack((kx_traj, ky_traj)).T
        kspace_time = time.time()
        print(f"  Interpolating k-space data to trajectory points...")
        # Kmax_res = res / (FoV * 1e-3) / 2
        kspace_sampled = fast_kspace_interpolation_v3(kspace_data, rosette_traj, FoV)#, Kmax_res)
        # kspace_sampled = griddata(points, values, interp_points, method='linear', fill_value=0)
        print(f"Interpolation timing {time.time() - kspace_time:.3f} seconds")

        # print(f"Interpolation completed. Sampled k-space shape: {kspace_sampled.shape}")
        
        # benchmark_interp_time = time.time()
        # print(f"  Benchmarking interpolation methods...")
        # benchmark_interp = benchmark_interpolation_methods(kspace_data, rosette_traj, FoV, Kmax_res)
        # print(f"Benchmarking completed in {time.time() - benchmark_interp_time:.3f} seconds")
        # print(f"kspace_sampled characteristics:")
        # print(f"  shape: {kspace_sampled.shape}")
        # print(f"  dtype: {kspace_sampled.dtype}")
        # print(f"  is complex: {np.iscomplexobj(kspace_sampled)}")
        # print(f"  max(abs): {np.max(np.abs(kspace_sampled))}")
        # print(f"  min(abs): {np.min(np.abs(kspace_sampled))}")


    except Exception as e:
        print(f"❌ Error in Step 4: {e}")

    np.save(os.path.join(config.tmp_dir, 'kspace_sampled.npy'), kspace_sampled)
    step4_time = time.time() - step4_start

    # --- Step 5: BART NUFFT reconstruction ---
    print("\nStep 5: BART NUFFT reconstruction...")
    step5_start = time.time()
    # Reshape for BART
    mr_data = kspace_sampled.reshape(kx_traj.shape)
    # print(f"Shape mrData: {mr_data.shape}\n")
    para = build_para(config)

    # Reshape kxx and kyy for BART
    kxx = kx_traj.reshape(points_per_paddle, noPaddles)
    kyy = ky_traj.reshape(points_per_paddle, noPaddles)
    kSpaceTrj = {'kxx': kxx, 'kyy': kyy}

    mr_data = kspace_sampled.reshape(kxx.shape)
    print(f"Shape mrData: {mr_data.shape}\n")


    recon_img = run_bart_nufft(mr_data, kSpaceTrj, para,run_gpu)
    recon_img = rescale_recon_img(recon_img, Nx, Ny, res)
    np.save(os.path.join(config.tmp_dir, 'recon_img.npy'), recon_img)
    step5_time = time.time() - step5_start

    print(f"\nReconstruction completed successfully.")
    
    # --- Step 6: Metrics calculation ---
    print(f"\nStep 6: Calculating metrics...")
    step6_start = time.time()
    ground_truth_img_down = downsample_image(image)
    np.save(os.path.join(config.tmp_dir, 'ground_truth_img_down.npy'), ground_truth_img_down)
    # Calculate all metrics
    metrics = calculate_all_metrics(ground_truth_img_down, recon_img)

    # Print formatted summary
    print_metrics_summary(metrics)
    step6_time = time.time() - step6_start
    
    # Calculate and print total pipeline time
    total_pipeline_time = time.time() - pipeline_start_time

    # Print pipeline timing summary
    print_pipeline_timing_summary(step1_time, step2_time, step3_time, step4_time, 
                                 step5_time, step6_time, total_pipeline_time, run_gpu)
    
    return {
        'image': image,
        'downsampled_image': ground_truth_img_down,
        'kspace_cartesian': kspace_data,
        'rosette_traj': rosette_traj,
        'kspace_sampled': kspace_sampled,
        'recon_img': recon_img
    }


# def run_bart_recon_gpu(load_kspace=False, run_gpu=False):
#     """
#     Implements the MATLAB pipeline in Python:
#     1. Load image
#     2. Cartesian k-space sampling via NUFFT (or load from file)
#     3. Generate rosette trajectory
#     4. Sample k-space data along trajectory
#     5. BART reconstruction
    
#     Args:
#         load_kspace (bool): If True, load k-space data from tmp/kspace_cartesian.npy 
#                            instead of recomputing it in Step 2
#     """
#     # Start total pipeline timer
#     pipeline_start_time = time.time()
    
#     config = create_default_config()
#     os.makedirs(config.tmp_dir, exist_ok=True)
#     os.makedirs(config.plots_dir, exist_ok=True)

#     # --- Step 1: Load image ---
#     print(f"\nStep 1: Loading image...")
#     step1_start = time.time()
#     image = load_image(config.input_image_path)
#     [Nx, Ny] = image.shape[0], image.shape[1]  # Assuming square image for simplicity
#     np.save(os.path.join(config.tmp_dir, 'image.npy'), image)
#     step1_time = time.time() - step1_start

#     # --- Step 2: Cartesian k-space sampling via NUFFT ---
#     if load_kspace:
#         print(f"\nStep 2: Loading k-space data from file...")
#         step2_start = time.time()
#         kspace_file_path = os.path.join(config.tmp_dir, 'kspace_cartesian.npy')
#         if os.path.exists(kspace_file_path):
#             kspace_data = np.load(kspace_file_path)
#             print(f"  Loaded k-space data from {kspace_file_path}")
#             print(f"  K-space data shape: {kspace_data.shape}")
#         else:
#             print(f"  ❌ K-space file not found at {kspace_file_path}")
#             print(f"  Falling back to computing k-space data...")
#             traj_cart = create_cartesian_trajectory(Nx, Ny)
#             kSpace_Trj_BART, image_in_6D = bart_reformat_nufft(image, traj_cart)
#             kspace_data = bart(1, 'nufft -i -t', kSpace_Trj_BART, image_in_6D)
#             kspace_data = kspace_data * Nx  # Scale by image size
#             np.save(kspace_file_path, kspace_data)
#         step2_time = time.time() - step2_start
#     else:
#         print(f"\nStep 2: Cartesian k-space sampling via NUFFT...")
#         step2_start = time.time()
#         traj_cart = create_cartesian_trajectory(Nx, Ny)
#         kSpace_Trj_BART, image_in_6D = bart_reformat_nufft(image, traj_cart)
#         kspace_data = bart(1, 'nufft -i -t', kSpace_Trj_BART, image_in_6D)
#         kspace_data = kspace_data * Nx  # Scale by image size
#         np.save(os.path.join(config.tmp_dir, 'kspace_cartesian.npy'), kspace_data)
#         step2_time = time.time() - step2_start

#     # --- Step 3: Generate rosette trajectory ---
#     print(f"\nStep 3: Generating rosette trajectory...")
#     step3_start = time.time()
#     FoV = config.trajectory.fov
#     res = config.trajectory.res
#     noPaddles = config.trajectory.no_paddles
#     points_per_paddle = 100
#     o1 = config.trajectory.o1
#     o2 = config.trajectory.o2
#     rosette_traj = generate_rosette_trajectory(FoV, res, noPaddles, points_per_paddle, o1, o2)
#     np.save(os.path.join(config.tmp_dir, 'rosette_traj.npy'), rosette_traj)
#     step3_time = time.time() - step3_start


#     # --- Step 4: Sample k-space data along Rosette trajectory ---
#     print(f"\nStep 4: Sampling k-space data along Rosette trajectory...")
#     step4_start = time.time()

#     try:
#         # print(f"  Nx: {Nx}, Ny: {Ny}, FoV: {FoV}")

#         # Kmax = Nx / (FoV * 1e-3) / 2
#         # print(f"  Kmax: {Kmax}")

#         kx_traj = np.real(rosette_traj)
#         ky_traj = np.imag(rosette_traj)

#         # kx_grid = np.linspace(-Kmax, Kmax, Nx)
#         # ky_grid = np.linspace(-Kmax, Kmax, Ny)
#         # # Meshgrid timing 
#         # print(f"  Creating meshgrid...")
#         # meshgrid_start = time.time()
#         # KX_grid, KY_grid = np.meshgrid(kx_grid, ky_grid)
#         # print(f"Meshgrid timing {time.time() - meshgrid_start:.3f} seconds")
#         # # print(f"  KX_grid shape: {KX_grid.shape}, KY_grid shape: {KY_grid.shape}")
#         # print(f"  Interpolating...")
        
#         # # Flatten grids and k-space data for griddata
#         # points = np.vstack((KX_grid.flatten(), KY_grid.flatten())).T
#         # # Extract middle 256x256 points from the grid

#         # values = kspace_data.flatten()
#         # interp_points = np.vstack((kx_traj, ky_traj)).T
#         kspace_time = time.time()
#         print(f"  Interpolating k-space data to trajectory points...")
#         # Kmax_res = res / (FoV * 1e-3) / 2
#         kspace_sampled = fast_kspace_interpolation_v3(kspace_data, rosette_traj, FoV)#, Kmax_res)
#         # kspace_sampled = griddata(points, values, interp_points, method='linear', fill_value=0)
#         print(f"Interpolation timing {time.time() - kspace_time:.3f} seconds")

#         # print(f"Interpolation completed. Sampled k-space shape: {kspace_sampled.shape}")
        
#         # benchmark_interp_time = time.time()
#         # print(f"  Benchmarking interpolation methods...")
#         # benchmark_interp = benchmark_interpolation_methods(kspace_data, rosette_traj, FoV, Kmax_res)
#         # print(f"Benchmarking completed in {time.time() - benchmark_interp_time:.3f} seconds")
#         # print(f"kspace_sampled characteristics:")
#         # print(f"  shape: {kspace_sampled.shape}")
#         # print(f"  dtype: {kspace_sampled.dtype}")
#         # print(f"  is complex: {np.iscomplexobj(kspace_sampled)}")
#         # print(f"  max(abs): {np.max(np.abs(kspace_sampled))}")
#         # print(f"  min(abs): {np.min(np.abs(kspace_sampled))}")


#     except Exception as e:
#         print(f"❌ Error in Step 4: {e}")

#     np.save(os.path.join(config.tmp_dir, 'kspace_sampled.npy'), kspace_sampled)
#     step4_time = time.time() - step4_start

#     # --- Step 5: BART NUFFT reconstruction ---
#     print("\nStep 5: BART NUFFT reconstruction...")
#     step5_start = time.time()
#     # Reshape for BART
#     mr_data = kspace_sampled.reshape(kx_traj.shape)
#     # print(f"Shape mrData: {mr_data.shape}\n")
#     para = build_para(config)

#     # Reshape kxx and kyy for BART
#     kxx = kx_traj.reshape(points_per_paddle, noPaddles)
#     kyy = ky_traj.reshape(points_per_paddle, noPaddles)
#     kSpaceTrj = {'kxx': kxx, 'kyy': kyy}

#     mr_data = kspace_sampled.reshape(kxx.shape)
#     print(f"Shape mrData: {mr_data.shape}\n")


#     recon_img = run_bart_nufft(mr_data, kSpaceTrj, para)
#     recon_img = rescale_recon_img(recon_img, Nx, Ny, res)
#     np.save(os.path.join(config.tmp_dir, 'recon_img.npy'), recon_img)
#     step5_time = time.time() - step5_start

#     print(f"\nReconstruction completed successfully.")
    
#     # --- Step 6: Metrics calculation ---
#     print(f"\nStep 6: Calculating metrics...")
#     step6_start = time.time()
#     ground_truth_img_down = downsample_image(image)
#     np.save(os.path.join(config.tmp_dir, 'ground_truth_img_down.npy'), ground_truth_img_down)
#     # Calculate all metrics
#     metrics = calculate_all_metrics(ground_truth_img_down, recon_img)

#     # Print formatted summary
#     print_metrics_summary(metrics)
#     step6_time = time.time() - step6_start
    
#     # Calculate and print total pipeline time
#     total_pipeline_time = time.time() - pipeline_start_time

#     # Print pipeline timing summary
#     print_pipeline_timing_summary(step1_time, step2_time, step3_time, step4_time, 
#                                  step5_time, step6_time, total_pipeline_time)
    
#     return {
#         'image': image,
#         'downsampled_image': ground_truth_img_down,
#         'kspace_cartesian': kspace_data,
#         'rosette_traj': rosette_traj,
#         'kspace_sampled': kspace_sampled,
#         'recon_img': recon_img
#     }
