"""
Simplified Linux BART Interface
===============================

This module provides a direct interface to BART (Berkeley Advanced Reconstruction Toolbox)
using the Python bindings. Much simpler than subprocess calls.

Based on the MATLAB pipeline steps:
1. Image loading and preprocessing
2. Cartesian k-space sampling via NUFFT  
3. Rosette trajectory generation
4. K-space data sampling along trajectory
5. BART reconstruction
"""

import os
import sys
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, Union

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add BART Python path
BART_PATH = "/home/fsaint/Desktop/MasterProject_Francesco/MasterProject_Francesco/matlab_scripts/extSrc/bart-0.9.00"
BART_PYTHON_PATH = os.path.join(BART_PATH, "python")

if BART_PYTHON_PATH not in sys.path:
    sys.path.insert(0, BART_PYTHON_PATH)

# Import BART functions
try:
    from bart import bart
    BART_AVAILABLE = True
    logger.info("\n✅ BART Python bindings imported successfully")
except ImportError as e:
    logger.warning(f"❌ Failed to import BART Python bindings: {e}")
    BART_AVAILABLE = False
    
    # Create dummy bart function for fallback
    def bart(*args, **kwargs):
        raise ImportError("BART Python bindings not available")


class BARTError(Exception):

# Only keep essentials for test_bart_connectivity.py

    """Custom exception for BART-related errors"""
    pass

def is_bart_available() -> bool:
    return BART_AVAILABLE

def get_bart_info() -> Dict[str, Any]:
    if BART_AVAILABLE:
        return {
            'available': True,
            'bart_path': BART_PATH,
            'python_path': BART_PYTHON_PATH,
            'version': "bart-0.9.00",
            'method': 'Python Bindings',
            'functional': True
        }
    else:
        return {
            'available': False,
            'bart_path': BART_PATH,
            'python_path': BART_PYTHON_PATH,
            'version': None,
            'method': 'Not Available',
            'functional': False
        }

def test_bart_installation() -> bool:
    print("Testing BART Installation via Python Bindings")
    print("=" * 50)
    info = get_bart_info()
    print(f"BART Available: {info['available']}")
    print(f"BART Path: {info['bart_path']}")
    print(f"Python Path: {info['python_path']}")
    print(f"Version: {info['version']}")
    print(f"Method: {info['method']}")
    print(f"Functional: {info['functional']}")
    if not BART_AVAILABLE:
        print("\n❌ BART Python bindings not available!")
        print("\nTroubleshooting suggestions:")
        print("1. Check if BART Python bindings exist:")
        print(f"   ls -la {BART_PYTHON_PATH}")
        print("2. Check if bart.py is in the Python directory:")
        print(f"   ls -la {BART_PYTHON_PATH}/bart.py")
        print("3. Make sure BART was compiled with Python support")
        print("4. Try importing manually:")
        print(f"   cd {BART_PYTHON_PATH} && python -c 'from bart import bart'")
        return False
    try:
        print("\nTesting basic BART commands...")
        phantom_result = bart(1,'phantom -x256')
        print(f"✅ Phantom created: shape {phantom_result.shape}")
        fft_result = bart(1,'fft 0', phantom_result)
        print(f"✅ FFT completed: shape {fft_result.shape}")
        traj = np.random.randn(3, 1, 100).astype(np.complex64)
        nufft_result = bart(1,'nufft -i', traj, phantom_result)
        print(f"✅ NUFFT completed: shape {nufft_result.shape}")
        print("\n✅ BART installation test passed!")
        return True
    except Exception as e:
        print(f"\n❌ BART installation test failed: {str(e)}")
        return False

def pics(data: np.ndarray, trajectory: Optional[np.ndarray] = None,
         regularization: float = 0.01, iterations: int = 30,
         **kwargs) -> np.ndarray:
    """
    Perform PICS reconstruction using BART
    
    Args:
        data: K-space data
        trajectory: Optional trajectory for non-Cartesian data
        regularization: Regularization parameter
        iterations: Number of iterations
        
    Returns:
        Reconstructed image
    """
    if not BART_AVAILABLE:
        raise BARTError("BART Python bindings not available")
    
    try:
        # Build PICS command
        cmd_parts = ["pics", "-i", str(iterations), "-R", f"T:7:0:{regularization}"]
        
        if trajectory is not None:
            cmd_parts.extend(["-t"])
            cmd = " ".join(cmd_parts)
            result = bart(cmd, trajectory, data)
        else:
            cmd = " ".join(cmd_parts)
            result = bart(cmd, data)
        
        return result
            
    except Exception as e:
        raise BARTError(f"PICS error: {str(e)}")

# Backward compatibility functions
def check_bart_or_exit():
    """Check BART availability or exit"""
    if not is_bart_available():
        print("❌ BART not available. Exiting.")
        exit(1)

def test_phantom_nufft():
    """Simple test: phantom, trajectory, nufft"""
    print("\n--- Simple BART phantom/nufft test ---")
    try:
        import bart
        # Create a sample image, reshape to 6D
        image_in = bart.bart(1, 'phantom -x256')
        image_in_6D = image_in.reshape(1, image_in.shape[0], image_in.shape[1], 1, 1, 1)
        # Generate a radial trajectory
        trajectory = bart.bart(1, 'traj -r -x256 -y256')
        # Call nufft
        kspace_data = bart.bart(1, 'nufft -i -t', trajectory, image_in_6D)
        print("K-space data created successfully.")
        print(f"Shape of kspace_data: {kspace_data.shape}")
        return True
    except Exception as e:
        print(f"❌ Simple phantom/nufft test failed: {e}")
        return False