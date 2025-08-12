"""
Metrics Module for Image Reconstruction Quality Assessment
========================================================

This module provides various metrics for evaluating the quality of reconstructed images
compared to ground truth images. Includes structural and pixel-wise similarity measures.
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mean_squared_error(ground_truth: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Mean Squared Error (MSE) between ground truth and reconstructed images.
    
    Args:
        ground_truth: Ground truth image array
        reconstructed: Reconstructed image array
        
    Returns:
        float: MSE value (lower is better)
    """
    try:
        # Ensure same shape
        if ground_truth.shape != reconstructed.shape:
            logger.warning(f"Shape mismatch: GT {ground_truth.shape} vs Recon {reconstructed.shape}")
            return float('inf')
        
        # Convert to real values if complex
        gt = np.real(ground_truth) if np.iscomplexobj(ground_truth) else ground_truth
        recon = np.abs(reconstructed) if np.iscomplexobj(reconstructed) else reconstructed
        
        mse = np.mean((gt - recon) ** 2)
        #logger.info(f"MSE: {mse:.6f}")
        return float(mse)
        
    except Exception as e:
        logger.error(f"Error calculating MSE: {e}")
        return float('inf')


def structural_similarity_index(ground_truth: np.ndarray, reconstructed: np.ndarray, 
                              data_range: float = None) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between ground truth and reconstructed images.
    
    Args:
        ground_truth: Ground truth image array
        reconstructed: Reconstructed image array
        data_range: Dynamic range of the data (max - min). If None, computed automatically.
        
    Returns:
        float: SSIM value between -1 and 1 (higher is better, 1 is perfect)
    """
    try:
        # Ensure same shape
        if ground_truth.shape != reconstructed.shape:
            logger.warning(f"Shape mismatch: GT {ground_truth.shape} vs Recon {reconstructed.shape}")
            return -1.0
        
        # Convert to real values if complex
        gt = np.real(ground_truth) if np.iscomplexobj(ground_truth) else ground_truth
        recon = np.abs(reconstructed) if np.iscomplexobj(reconstructed) else reconstructed
        
        # Normalize to [0, 1] range if needed
        gt_norm = (gt - gt.min()) / (gt.max() - gt.min()) if gt.max() != gt.min() else gt
        recon_norm = (recon - recon.min()) / (recon.max() - recon.min()) if recon.max() != recon.min() else recon
        
        if data_range is None:
            data_range = max(gt_norm.max() - gt_norm.min(), recon_norm.max() - recon_norm.min())
        
        ssim_value = ssim(gt_norm, recon_norm, data_range=data_range)
        #logger.info(f"SSIM: {ssim_value:.6f}")
        return float(ssim_value)
        
    except Exception as e:
        logger.error(f"Error calculating SSIM: {e}")
        return -1.0


def pearson_correlation_coefficient(ground_truth: np.ndarray, reconstructed: np.ndarray) -> float:
    """
    Calculate Pearson Correlation Coefficient between ground truth and reconstructed images.
    
    Args:
        ground_truth: Ground truth image array
        reconstructed: Reconstructed image array
        
    Returns:
        float: Pearson correlation coefficient between -1 and 1 (higher is better, 1 is perfect)
    """
    try:
        # Ensure same shape
        if ground_truth.shape != reconstructed.shape:
            logger.warning(f"Shape mismatch: GT {ground_truth.shape} vs Recon {reconstructed.shape}")
            return 0.0
        
        # Convert to real values if complex and flatten
        gt = np.real(ground_truth).flatten() if np.iscomplexobj(ground_truth) else ground_truth.flatten()
        recon = np.abs(reconstructed).flatten() if np.iscomplexobj(reconstructed) else reconstructed.flatten()
        
        # Remove any NaN or infinite values
        mask = np.isfinite(gt) & np.isfinite(recon)
        if not mask.any():
            logger.warning("No finite values found for correlation calculation")
            return 0.0
        
        correlation, _ = pearsonr(gt[mask], recon[mask])
        #logger.info(f"Pearson Correlation: {correlation:.6f}")
        return float(correlation)
        
    except Exception as e:
        logger.error(f"Error calculating Pearson correlation: {e}")
        return 0.0


def calculate_all_metrics(ground_truth: np.ndarray, reconstructed: np.ndarray) -> dict:
    """
    Calculate all available metrics between ground truth and reconstructed images.
    
    Args:
        ground_truth: Ground truth image array
        reconstructed: Reconstructed image array
        
    Returns:
        dict: Dictionary containing all computed metrics
    """
    #logger.info("Calculating all metrics...")
    
    metrics = {
        'MSE': mean_squared_error(ground_truth, reconstructed),
        'SSIM': structural_similarity_index(ground_truth, reconstructed),
        'Pearson_Correlation': pearson_correlation_coefficient(ground_truth, reconstructed)
    }
    
    #logger.info("Metrics calculation completed")
    return metrics


def print_metrics_summary(metrics: dict) -> None:
    """
    Print a formatted summary of all metrics.
    
    Args:
        metrics: Dictionary containing computed metrics
    """
    print("\n" + "="*60)
    print("          RECONSTRUCTION METRICS SUMMARY")
    print("="*60)
    
    # Quality metrics (higher is better)
    print("\nQuality Metrics (Higher is Better):")
    print(f"  SSIM:                {metrics['SSIM']:.6f}")
    print(f"  Pearson Correlation: {metrics['Pearson_Correlation']:.6f}")
    
    # Error metrics (lower is better)
    print("\nError Metrics (Lower is Better):")
    print(f"  MSE:                 {metrics['MSE']:.6f}")
    
    print("="*60)
    print(2 * "\n")