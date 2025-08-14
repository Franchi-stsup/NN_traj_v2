import torch
import numpy as np
from src.bart_interface import bart

class BartNufft(torch.autograd.Function):
    """
    A custom PyTorch function to wrap the BART NUFFT operator.
    """
    
    @staticmethod
    def forward(ctx, image_tensor, trajectory_tensor):
        """
        Runs the forward NUFFT operation (Image -> k-space).
        
        Args:
            image_tensor (torch.Tensor): The input image.
            trajectory_tensor (torch.Tensor): The k-space trajectory coordinates.
        """
        # Save trajectory for the backward pass
        ctx.save_for_backward(trajectory_tensor)
        
        # Convert PyTorch tensors to NumPy arrays for BART
        # BART expects complex data, so ensure input is complex
        image_np = image_tensor.detach().cpu().numpy()
        traj_np = trajectory_tensor.detach().cpu().numpy()
        
        # Call BART's forward NUFFT
        # bart(1, "nufft ...", input, trajectory)
        kspace_np = bart(1, "nufft -i", image_np, traj_np)
        
        # Convert the result back to a PyTorch tensor
        kspace_tensor = torch.from_numpy(kspace_np).to(image_tensor.device)
        
        return kspace_tensor

    @staticmethod
    def backward(ctx, grad_output_kspace):
        """
        Runs the adjoint NUFFT operation (k-space -> Image) for the gradient.
        
        Args:
            grad_output_kspace (torch.Tensor): The gradient from the subsequent layer.
        """
        # Retrieve the saved trajectory
        trajectory_tensor, = ctx.saved_tensors
        
        # Convert tensors to NumPy arrays
        grad_kspace_np = grad_output_kspace.detach().cpu().numpy()
        traj_np = trajectory_tensor.detach().cpu().numpy()
        
        # Call BART's ADJOINT NUFFT using the -a flag
        # This is the key step for calculating the gradient w.r.t the image
        grad_image_np = bart(1, "nufft -a", grad_kspace_np, traj_np)
        
        # Convert the gradient on the image back to a PyTorch tensor
        grad_image_tensor = torch.from_numpy(grad_image_np).to(grad_output_kspace.device)
        
        # The backward pass must return a gradient for each input of forward().
        # We only need the gradient for the image_tensor, not the trajectory.
        return grad_image_tensor, None

# --- Example Usage ---
print("Testing BART NUFFT with PyTorch...")
# Get a convenience handle for the function
nufft_operator = BartNufft.apply

# print("nufft_operator is a callable function:", callable(nufft_operator))
# Create some dummy data
image = torch.randn(1, 128, 128, dtype=torch.complex64, requires_grad=True)

print("image type:", image.dtype)
# Trajectory should have shape (3, N) for 2D, (4, N) for 3D
trajectory = torch.randn(3, 128 * 128, dtype=torch.float32)

print("trajectory type:", trajectory.dtype)
# Apply the NUFFT operator just like any other PyTorch function
kspace = nufft_operator(image, trajectory)

print("NUFFT operation completed. k-space shape:", kspace.shape)
# Calculate some dummy loss on the k-space data
loss = torch.sum(torch.abs(kspace)**2)

print("Loss calculated:", loss.item())
# Backpropagate! PyTorch will automatically call your custom backward method.
loss.backward()

# The gradient is now available in image.grad
print("Gradient calculated for the input image tensor:")
print(image.grad.shape)
print("Gradient is not None:", image.grad is not None)