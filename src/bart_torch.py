import torch
import numpy as np
from src.bart_interface import bart

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Image dimensions
IMAGE_SIZE = 64  # Image width and height (assumes square images)
IMAGE_CHANNELS = 1  # Number of channels (1 for grayscale, 3 for RGB)

# BART NUFFT parameters
BART_6D_DIMS = 6  # BART expects 6D tensors: (batch, height, width, coils, time, slices)
TRAJ_DIMS = 3  # Trajectory dimensions: (kx, ky, kz) where kz=0 for 2D

# Trajectory generation parameters
TRAJ_TYPE = 'radial'  # Type of trajectory: 'radial', 'spiral', 'cartesian'
TRAJ_SIZE = IMAGE_SIZE  # Trajectory sampling grid size

# Data types
IMAGE_DTYPE = torch.complex64  # Complex data type for MR images
TRAJ_DTYPE = torch.complex64  # Complex data type for k-space trajectories

# BART command flags
NUFFT_FORWARD_FLAGS = "nufft -i -t"  # Forward NUFFT with trajectory flag
NUFFT_ADJOINT_FLAGS = "nufft -a -t"  # Adjoint NUFFT with trajectory flag

# Debug options
VERBOSE_BACKWARD = True  # Print debug information during backward pass
VERBOSE_FORWARD = False  # Print debug information during forward pass

# =============================================================================
# BART NUFFT PYTORCH WRAPPER
# =============================================================================

class BartNufft(torch.autograd.Function):
    """
    A custom PyTorch function to wrap the BART NUFFT operator.
    
    This class implements forward and backward passes for non-uniform Fast Fourier Transform (NUFFT)
    operations using the Berkeley Advanced Reconstruction Toolbox (BART). It enables seamless
    integration of BART's NUFFT capabilities into PyTorch neural networks with automatic differentiation.
    
    Features:
    - Forward pass: Image domain -> k-space (NUFFT)
    - Backward pass: k-space -> Image domain (Adjoint NUFFT) 
    - Automatic gradient computation for backpropagation
    - Support for arbitrary k-space trajectories
    - GPU/CPU tensor compatibility
    
    Notes:
    - Input images must be complex-valued (torch.complex64)
    - Trajectories must be BART-compatible format: (3, N, M) complex64
    - BART expects specific dimensional formats for proper operation
    """
    
    @staticmethod
    def forward(ctx, image_tensor, trajectory_tensor):
        """
        Runs the forward NUFFT operation (Image -> k-space).
        
        Args:
            image_tensor (torch.Tensor): Input image tensor with shape (channels, height, width)
                                       Must be complex64 dtype
            trajectory_tensor (torch.Tensor): K-space trajectory coordinates with shape (3, N, M)
                                             where 3 represents (kx, ky, kz) and kz=0 for 2D
                                             Must be complex64 dtype
                                             
        Returns:
            torch.Tensor: K-space data with shape corresponding to trajectory sampling pattern
        """
        # Save trajectory for the backward pass
        ctx.save_for_backward(trajectory_tensor)
        
        # Convert PyTorch tensors to NumPy arrays for BART
        image_np = image_tensor.detach().cpu().numpy()
        traj_np = trajectory_tensor.detach().cpu().numpy()
        
        # BART expects 6D image format: (batch, height, width, coils, time, slices)
        # Reshape from (channels, height, width) to (1, height, width, 1, 1, 1)
        if image_np.ndim == 3 and image_np.shape[0] == IMAGE_CHANNELS:
            image_np = image_np.reshape(1, image_np.shape[1], image_np.shape[2], 1, 1, 1)
        
        if VERBOSE_FORWARD:
            print(f"Forward: image shape: {image_np.shape}")
            print(f"Forward: trajectory shape: {traj_np.shape}")
        
        # Call BART's forward NUFFT: Image -> k-space
        # -i: inverse NUFFT (forward transform from image to k-space)
        # -t: use provided trajectory
        kspace_np = bart(1, NUFFT_FORWARD_FLAGS, traj_np, image_np)
        
        # Convert the result back to a PyTorch tensor
        kspace_tensor = torch.from_numpy(kspace_np).to(image_tensor.device)
        
        return kspace_tensor

    @staticmethod
    def backward(ctx, grad_output_kspace):
        """
        Runs the adjoint NUFFT operation (k-space -> Image) for gradient computation.
        
        Args:
            grad_output_kspace (torch.Tensor): Gradient from subsequent layer in k-space domain
                                              Shape depends on the trajectory sampling pattern
                                              
        Returns:
            tuple: (grad_image_tensor, None) where:
                  - grad_image_tensor: Gradient w.r.t. input image (same shape as forward input)
                  - None: No gradient w.r.t. trajectory (assumed fixed)
        """
        # Retrieve the saved trajectory from forward pass
        trajectory_tensor, = ctx.saved_tensors
        
        # Convert tensors to NumPy arrays for BART
        grad_kspace_np = grad_output_kspace.detach().cpu().numpy()
        traj_np = trajectory_tensor.detach().cpu().numpy()
        
        # Ensure k-space gradient has proper dimensionality for BART
        # BART expects at least 3D: (batch, height, width)
        if grad_kspace_np.ndim == 2:  # (H, W) -> (1, H, W)
            grad_kspace_np = grad_kspace_np[np.newaxis, ...]
        
        if VERBOSE_BACKWARD:
            print(f"Backward: grad_kspace shape: {grad_kspace_np.shape}")
            print(f"Backward: trajectory shape: {traj_np.shape}")
        
        # Call BART's adjoint NUFFT: k-space -> Image
        # -a: adjoint NUFFT (backward transform from k-space to image)
        # -t: use provided trajectory
        grad_image_np = bart(1, NUFFT_ADJOINT_FLAGS, traj_np, grad_kspace_np)
        
        if VERBOSE_BACKWARD:
            print(f"Backward: grad_image shape from BART: {grad_image_np.shape}")
        
        # Convert the gradient back to a PyTorch tensor
        grad_image_tensor = torch.from_numpy(grad_image_np).to(grad_output_kspace.device)
        
        # Ensure gradient tensor matches the original input tensor shape
        # Handle BART's 6D output format: (1, H, W, 1, 1, 1) -> (1, H, W)
        if grad_image_tensor.ndim == BART_6D_DIMS:
            # Squeeze out extra dimensions but preserve batch dimension
            grad_image_tensor = grad_image_tensor.squeeze()
            if grad_image_tensor.ndim == 2:  # (H, W) -> (1, H, W)
                grad_image_tensor = grad_image_tensor.unsqueeze(0)
        elif grad_image_tensor.ndim == 2:  # (H, W) -> (1, H, W)
            grad_image_tensor = grad_image_tensor.unsqueeze(0)
            
        if VERBOSE_BACKWARD:
            print(f"Backward: final grad_image shape: {grad_image_tensor.shape}")
        
        # Return gradients: (grad w.r.t. image, grad w.r.t. trajectory)
        # We don't compute gradients w.r.t. trajectory (assumed fixed), so return None
        return grad_image_tensor, None

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def generate_bart_trajectory(traj_type='radial', size=IMAGE_SIZE):
    """
    Generate a BART-compatible trajectory.
    
    Args:
        traj_type (str): Type of trajectory ('radial', 'spiral', 'cartesian')
        size (int): Grid size for trajectory generation
        
    Returns:
        torch.Tensor: Trajectory tensor with shape (3, size, size) and dtype complex64
    """
    if traj_type == 'radial':
        traj_cmd = f'traj -r -x{size} -y{size}'
    elif traj_type == 'spiral':
        traj_cmd = f'traj -s -x{size} -y{size}'
    elif traj_type == 'cartesian':
        traj_cmd = f'traj -c -x{size} -y{size}'
    else:
        raise ValueError(f"Unsupported trajectory type: {traj_type}")
    
    traj_np = bart(1, traj_cmd)
    return torch.from_numpy(traj_np)

def create_test_image(channels=IMAGE_CHANNELS, height=IMAGE_SIZE, width=IMAGE_SIZE, dtype=IMAGE_DTYPE):
    """
    Create a test image tensor for NUFFT testing.
    
    Args:
        channels (int): Number of image channels
        height (int): Image height
        width (int): Image width  
        dtype (torch.dtype): Data type for the image
        
    Returns:
        torch.Tensor: Random complex image tensor with requires_grad=True
    """
    return torch.randn(channels, height, width, dtype=dtype, requires_grad=True)

# =============================================================================
# EXAMPLE USAGE AND TESTING
# =============================================================================

def test_bart_nufft():
    """
    Test function to demonstrate BART NUFFT PyTorch wrapper functionality.
    
    This function:
    1. Creates a test image and trajectory
    2. Applies forward NUFFT (image -> k-space)
    3. Computes a loss function
    4. Performs backpropagation (adjoint NUFFT for gradients)
    5. Validates gradient computation
    """
    print("Testing BART NUFFT with PyTorch...")
    print("=" * 50)
    
    # Create NUFFT operator handle
    nufft_operator = BartNufft.apply
    
    # Generate test image
    print(f"Creating test image: {IMAGE_CHANNELS}×{IMAGE_SIZE}×{IMAGE_SIZE}")
    image = create_test_image()
    print(f"Image type: {image.dtype}")
    print(f"Image shape: {image.shape}")
    print(f"Image requires grad: {image.requires_grad}")
    
    # Generate BART-compatible trajectory
    print(f"\nGenerating {TRAJ_TYPE} trajectory: {TRAJ_SIZE}×{TRAJ_SIZE}")
    trajectory = generate_bart_trajectory(TRAJ_TYPE, TRAJ_SIZE)
    print(f"Trajectory type: {trajectory.dtype}")
    print(f"Trajectory shape: {trajectory.shape}")
    
    # Forward pass: Image -> k-space
    print(f"\nPerforming forward NUFFT (Image -> k-space)...")
    kspace = nufft_operator(image, trajectory)
    print(f"✅ Forward NUFFT completed")
    print(f"K-space shape: {kspace.shape}")
    print(f"K-space dtype: {kspace.dtype}")
    
    # Compute loss function
    print(f"\nComputing loss function...")
    loss = torch.sum(torch.abs(kspace)**2)
    print(f"Loss value: {loss.item():.6f}")
    
    # Backward pass: k-space -> Image (gradient computation)
    print(f"\nPerforming backward pass (gradient computation)...")
    loss.backward()
    print(f"✅ Backward pass completed")
    
    # Validate gradients
    print(f"\nGradient validation:")
    print(f"Gradient shape: {image.grad.shape}")
    print(f"Gradient dtype: {image.grad.dtype}")
    print(f"Gradient computed: {image.grad is not None}")
    print(f"Gradient norm: {torch.norm(image.grad).item():.6f}")
    
    print("\n" + "=" * 50)
    print("✅ BART NUFFT PyTorch wrapper test completed successfully!")
    
    return {
        'image_shape': image.shape,
        'kspace_shape': kspace.shape,
        'trajectory_shape': trajectory.shape,
        'loss': loss.item(),
        'gradient_norm': torch.norm(image.grad).item()
    }

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the test
    results = test_bart_nufft()