"""
Neural Network Architecture and Dataset Classes
Contains the CircleCNN model and CircleDataset for circle trajectory reconstruction.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from src.bart_interface import bart # Assuming BART is installed and accessible

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CircleDataset(Dataset):
    """Dataset for circle generation training"""
    
    def __init__(self, num_samples=1000, input_length=128, output_length=128, 
                 radius=1.0, noise_level=0.0):
        """
        Args:
            num_samples: Number of training samples
            input_length: Length of input t-vector (128 for 5ms resolution over 640ms)
            output_length: Length of output vectors (128 for 5ms resolution)
            radius: Circle radius
            noise_level: Noise level for input signals
        """
        self.num_samples = num_samples
        self.input_length = input_length
        self.output_length = output_length
        self.radius = radius
        self.noise_level = noise_level
        
        # Time vectors - both input and output have same temporal resolution
        self.dt = 0.005  # 5ms resolution for both input and output

        self.input_time = torch.linspace(0, 0.64, input_length+1)
        # get rid of the last point to match output length
        self.input_time = self.input_time[:-1]
        self.output_time = torch.linspace(0, 0.64, output_length+1)
        self.output_time = self.output_time[:-1]  # Ensure same length

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate input signal (can be time vector + noise, or more complex signal)
        # For now, using time + random noise as input
        input_signal = self.input_time # + self.noise_level * torch.randn(self.input_length)
        
        # Generate target circle coordinates
        # One full circle over 640ms
        angle = 2 * np.pi * self.output_time / 0.64
        kx_target = self.radius * torch.cos(angle)
        ky_target = self.radius * torch.sin(angle)
        tmpx = kx_target
        tmpy = ky_target

        ky_target = tmpx
        kx_target = tmpy
        ky_target = -ky_target  # Invert ky for the desired output
        
        # Stack kx and ky for output
        target = torch.stack([kx_target, ky_target], dim=0)  # Shape: (2, 128)
        
        return input_signal.float(), target.float()


class CircleCNN(nn.Module):
    """1D CNN for circle generation with equal input/output lengths"""
    
    def __init__(self, input_length=128, output_length=128, hidden_channels=64):
        super(CircleCNN, self).__init__()
        
        self.input_length = input_length
        self.output_length = output_length
        
        # Simplified architecture since input and output are same length
        self.conv_layers = nn.Sequential(
            # First conv block
            nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            # Second conv block
            nn.Conv1d(hidden_channels, hidden_channels*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_channels*2),
            nn.ReLU(),
            
            # Third conv block
            nn.Conv1d(hidden_channels*2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(),
            
            # Output conv layer - maps to 2 channels (kx, ky)
            nn.Conv1d(hidden_channels, 2, kernel_size=1),
        )
        
    def forward(self, x):
        # Input shape: (batch_size, input_length)
        # Add channel dimension for Conv1d
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, input_length)
        
        # Apply convolutional layers
        x = self.conv_layers(x)  # Shape: (batch_size, 2, input_length)
        
        # Output is already in the right format: (batch_size, 2, output_length)
        return x


class CircleCNN_02(nn.Module):
    """Enhanced 1D CNN for circle generation with more parameters and residual connections"""
    
    def __init__(self, input_length=128, output_length=128, base_channels=64):
        super(CircleCNN_02, self).__init__()
        
        self.input_length = input_length
        self.output_length = output_length
        
        # Input projection
        self.input_conv = nn.Sequential(
            nn.Conv1d(1, base_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        
        # Encoder blocks (expand channels)
        self.encoder_block1 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(),
            nn.Conv1d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU()
        )
        
        self.encoder_block2 = nn.Sequential(
            nn.Conv1d(base_channels*2, base_channels*4, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU(),
            nn.Conv1d(base_channels*4, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU()
        )
        
        # Middle processing block (highest resolution)
        self.middle_block = nn.Sequential(
            nn.Conv1d(base_channels*4, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*8),
            nn.ReLU(),
            nn.Conv1d(base_channels*8, base_channels*8, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*8),
            nn.ReLU(),
            nn.Conv1d(base_channels*8, base_channels*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*4),
            nn.ReLU()
        )
        
        # Decoder blocks (reduce channels) with skip connections
        # decoder_block1: input = middle_block(256) + skip2(128) = 384 channels
        self.decoder_block1 = nn.Sequential(
            nn.Conv1d(base_channels*6, base_channels*2, kernel_size=3, padding=1),  # 6*64 = 384 input channels
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU(),
            nn.Conv1d(base_channels*2, base_channels*2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels*2),
            nn.ReLU()
        )
        
        # decoder_block2: input = decoder_block1(128) + skip1(64) = 192 channels
        self.decoder_block2 = nn.Sequential(
            nn.Conv1d(base_channels*3, base_channels, kernel_size=3, padding=1),  # 3*64 = 192 input channels
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        
        # Multi-scale feature extraction (parallel branches)
        # Input to multiscale: decoder_block2(64) + skip1(64) = 128 channels
        self.multiscale_conv1 = nn.Conv1d(base_channels*2, base_channels//2, kernel_size=1, padding=0)
        self.multiscale_conv3 = nn.Conv1d(base_channels*2, base_channels//2, kernel_size=3, padding=1)
        self.multiscale_conv5 = nn.Conv1d(base_channels*2, base_channels//2, kernel_size=5, padding=2)
        self.multiscale_conv7 = nn.Conv1d(base_channels*2, base_channels//2, kernel_size=7, padding=3)
        
        # Feature fusion and output
        # Input to fusion: multiscale(128) + decoder_block2(64) = 192 channels
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(base_channels*3, base_channels, kernel_size=3, padding=1),  # 3*64 = 192 input channels
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.Conv1d(base_channels, base_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_channels//2),
            nn.ReLU()
        )
        
        # Final output layer
        self.output_conv = nn.Conv1d(base_channels//2, 2, kernel_size=1)
        
        # Residual connection weights (learnable parameters)
        self.skip_weight1 = nn.Parameter(torch.ones(1) * 0.5)
        self.skip_weight2 = nn.Parameter(torch.ones(1) * 0.5)
        
    def forward(self, x):
        # Input projection
        x = x.unsqueeze(1)  # (batch, 1, 128)
        x = self.input_conv(x)  # (batch, 64, 128)
        
        # Encoder path with skip connections
        skip1 = x  # (batch, 64, 128) - Save for skip connection
        x = self.encoder_block1(x)  # (batch, 128, 128)
        
        skip2 = x  # (batch, 128, 128) - Save for skip connection  
        x = self.encoder_block2(x)  # (batch, 256, 128)
        
        # Middle processing
        x = self.middle_block(x)  # (batch, 256, 128)
        
        # Decoder path with skip connections
        x = torch.cat([x, skip2], dim=1)  # (batch, 384, 128) = 256 + 128
        x = self.decoder_block1(x)  # (batch, 128, 128)
        
        x = torch.cat([x, skip1], dim=1)  # (batch, 192, 128) = 128 + 64
        x = self.decoder_block2(x)  # (batch, 64, 128)
        
        # Multi-scale processing
        multiscale_input = torch.cat([x, skip1], dim=1)  # (batch, 128, 128) = 64 + 64
        
        ms1 = self.multiscale_conv1(multiscale_input)  # (batch, 32, 128)
        ms3 = self.multiscale_conv3(multiscale_input)  # (batch, 32, 128)
        ms5 = self.multiscale_conv5(multiscale_input)  # (batch, 32, 128)
        ms7 = self.multiscale_conv7(multiscale_input)  # (batch, 32, 128)
        
        # Combine multi-scale features
        multiscale_features = torch.cat([ms1, ms3, ms5, ms7], dim=1)  # (batch, 128, 128)
        
        # Feature fusion
        fused_features = torch.cat([multiscale_features, x], dim=1)  # (batch, 192, 128) = 128 + 64
        x = self.feature_fusion(fused_features)  # (batch, 32, 128)
        
        # Final output
        x = self.output_conv(x)  # (batch, 2, 128)
        
        return x

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


def get_device():
    """Get the appropriate device (GPU if available, otherwise CPU)"""
    return device


def print_device_info():
    """Print information about the device being used"""
    print(f"Using device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
