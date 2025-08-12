"""
Training, Loss Functions, and Model Management
Contains training loops, custom loss functions, and model save/load functionality.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
import threading
import queue
import time
from src.network import CircleCNN, CircleCNN_02, get_device

# Global variables for keyboard handling
keyboard_queue = queue.Queue()
keyboard_thread = None
keyboard_running = False

def keyboard_listener():
    """Background thread to listen for keyboard input"""
    global keyboard_running
    
    while keyboard_running:
        try:
            if os.name == 'nt':  # Windows
                char = input()
                if char.lower() == 'q':
                    keyboard_queue.put('q')
                    break
            else:  # Unix/Linux/Mac
                import select
                import sys
                
                # Use select with a very short timeout to check periodically
                ready, _, _ = select.select([sys.stdin], [], [], 0.2)  # 200ms timeout
                if ready and keyboard_running:  # Double-check keyboard_running
                    try:
                        line = sys.stdin.readline().strip()
                        if line.lower() == 'q':
                            keyboard_queue.put('q')
                            break
                    except (IOError, OSError):
                        pass
                
                # If no input, just continue to check keyboard_running
                        
        except (EOFError, KeyboardInterrupt, OSError):
            break
    
    keyboard_running = False

def start_keyboard_listener():
    """Start the keyboard listener thread"""
    global keyboard_thread, keyboard_running
    if not keyboard_running:
        keyboard_running = True
        keyboard_thread = threading.Thread(target=keyboard_listener, daemon=False)  # Not daemon
        keyboard_thread.start()
        print("Keyboard listener started")

def stop_keyboard_listener():
    """Stop the keyboard listener thread"""
    global keyboard_running, keyboard_thread
    
    if not keyboard_running:
        return  # Already stopped
        
    print("Stopping keyboard listener...")
    keyboard_running = False
    
    # Give the thread a moment to exit gracefully
    if keyboard_thread and keyboard_thread.is_alive():
        keyboard_thread.join(timeout=1.0)  # Wait up to 1 second
        
        if keyboard_thread.is_alive():
            print("Warning: Keyboard thread did not exit cleanly within timeout")
            # Force the program to continue anyway
        
    # Clear any remaining items in the queue
    while not keyboard_queue.empty():
        try:
            keyboard_queue.get_nowait()
        except queue.Empty:
            break
    
    # Reset the thread reference
    keyboard_thread = None
    print("Keyboard listener stopped")

def check_keyboard_interrupt():
    """
    Check for keyboard interrupt (specifically 'q' key press followed by Enter)
    Uses a background thread approach that works reliably across platforms
    """
    try:
        # Check if 'q' was pressed
        char = keyboard_queue.get_nowait()
        return char == 'q'
    except queue.Empty:
        return False

device = get_device()


class SmoothLoss(nn.Module):
    """
    Custom loss function with derivative smoothness penalties
    """
    def __init__(self, mse_weight=1.0, first_deriv_weight=0.0005, second_deriv_weight=0.000, dt=0.005):
        super(SmoothLoss, self).__init__()
        self.mse_weight = mse_weight
        self.first_deriv_weight = first_deriv_weight
        self.second_deriv_weight = second_deriv_weight
        self.dt = dt
        self.mse_loss = nn.MSELoss()
        
    def compute_derivatives_torch(self, tensor):
        """
        Compute first and second derivatives using torch operations
        tensor shape: (batch_size, 2, sequence_length)
        """
        # First derivative using finite differences
        first_deriv = (tensor[:, :, 1:] - tensor[:, :, :-1]) / self.dt
        
        # Second derivative
        second_deriv = (first_deriv[:, :, 1:] - first_deriv[:, :, :-1]) / self.dt
        
        return first_deriv, second_deriv
    
    def forward(self, predicted, target):
        """
        Custom loss with smoothness penalties
        """
        # Standard MSE loss
        mse_loss = self.mse_loss(predicted, target)
        
        # Compute derivatives for both predicted and target
        pred_first_deriv, pred_second_deriv = self.compute_derivatives_torch(predicted)
        target_first_deriv, target_second_deriv = self.compute_derivatives_torch(target)
        
        # First derivative penalty: penalize maximum absolute derivative
        first_deriv_penalty = torch.max(torch.abs(pred_first_deriv))
        
        # Second derivative penalty: penalize maximum absolute second derivative
        second_deriv_penalty = torch.max(torch.abs(pred_second_deriv))
        
        # Optional: Also penalize difference between predicted and target derivatives
        first_deriv_mse = self.mse_loss(pred_first_deriv, target_first_deriv)
        second_deriv_mse = self.mse_loss(pred_second_deriv, target_second_deriv)
        
        # Total loss
        total_loss = (self.mse_weight * mse_loss + 
                     self.first_deriv_weight * (first_deriv_penalty + first_deriv_mse) +
                     self.second_deriv_weight * (second_deriv_penalty + second_deriv_mse))
        
        # Return loss components for monitoring
        loss_components = {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'first_deriv_penalty': first_deriv_penalty,
            'second_deriv_penalty': second_deriv_penalty,
            'first_deriv_mse': first_deriv_mse,
            'second_deriv_mse': second_deriv_mse
        }
        
        return total_loss, loss_components


def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=1e-3, 
                use_smooth_loss=True, model_save_dir='models', use_scheduler=False,
                mse_weight=1.0, first_deriv_weight=0.001, second_deriv_weight=0.000,
                save_params=None):
    """
    Train the CNN model with keyboard interruption support
    
    Args:
        save_params: Dictionary containing parameters for emergency model saving
    """
    
    if use_smooth_loss:
        criterion = SmoothLoss(mse_weight=mse_weight, 
                             first_deriv_weight=first_deriv_weight, 
                             second_deriv_weight=second_deriv_weight)
    else:
        criterion = nn.MSELoss()
        
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Optional scheduler
    if use_scheduler:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
    else:
        scheduler = None
    
    train_losses = []
    val_losses = []
    
    model.train()
    
    # Flag for early termination
    interrupted = False
    
    print("\n=== TRAINING STARTED ===")
    print("Type 'q' and press Enter at any time to interrupt training and save the model")
    print("(The check happens between epochs and every 10 batches)")
    
    # Start keyboard listener
    start_keyboard_listener()
    
    try:
        # Main training loop with progress bar
        with tqdm(total=num_epochs, desc="Training Progress") as pbar:
            for epoch in range(num_epochs):
                # Check for keyboard interrupt at the beginning of each epoch
                if check_keyboard_interrupt():
                    print(f"\n\n=== KEYBOARD INTERRUPT DETECTED ===")
                    print(f"Training stopped at epoch {epoch+1}/{num_epochs}")
                    interrupted = True
                    break
                
                # Training phase
                total_train_loss = 0.0
                train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Train", 
                               leave=False, disable=True)
                
                for batch_idx, (inputs, targets) in enumerate(train_bar):
                    # Check for interrupt every 10 batches for more responsiveness
                    if batch_idx % 10 == 0 and check_keyboard_interrupt():
                        print(f"\n\n=== KEYBOARD INTERRUPT DETECTED ===")
                        print(f"Training stopped at epoch {epoch+1}/{num_epochs}, batch {batch_idx}")
                        interrupted = True
                        break
                    
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    
                    if use_smooth_loss:
                        loss, _ = criterion(outputs, targets)
                    else:
                        loss = criterion(outputs, targets)
                        
                    loss.backward()
                    optimizer.step()
                    
                    total_train_loss += loss.item()
                
                # Break out of epoch loop if interrupted during batch processing
                if interrupted:
                    break
                
                # Validation phase
                model.eval()
                total_val_loss = 0.0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs, targets = inputs.to(device), targets.to(device)
                        outputs = model(inputs)
                        
                        if use_smooth_loss:
                            val_loss, _ = criterion(outputs, targets)
                        else:
                            val_loss = criterion(outputs, targets)
                            
                        total_val_loss += val_loss.item()
                
                model.train()
                if scheduler is not None:
                    scheduler.step()
                
                avg_train_loss = total_train_loss / len(train_loader)
                avg_val_loss = total_val_loss / len(val_loader)
                
                train_losses.append(avg_train_loss)
                val_losses.append(avg_val_loss)
                
                # Update progress bar with loss information
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else learning_rate
                pbar.set_postfix({
                    'Train Loss': f'{avg_train_loss:.6f}',
                    'Val Loss': f'{avg_val_loss:.6f}',
                    'LR': f'{current_lr:.2e}'
                })
                pbar.update(1)
                
                # Print detailed info every 20 epochs
                if (epoch + 1) % 20 == 0:
                    print(f'\nEpoch [{epoch+1}/{num_epochs}], '
                          f'Train Loss: {avg_train_loss:.6f}, '
                          f'Val Loss: {avg_val_loss:.6f}')
    
    finally:
        # Always stop the keyboard listener
        print("Cleaning up keyboard listener...")
        stop_keyboard_listener()
        print("Training function cleanup completed")    # Handle early termination due to keyboard interrupt
    if interrupted:
        print(f"\n=== EMERGENCY MODEL SAVE ===")
        print("Saving interrupted model...")
        
        # Save the model with interrupted status
        if save_params:
            # Update the epoch count to reflect actual training
            save_params['num_epochs'] = len(train_losses)
            save_params['interrupted'] = True
            
            # Create emergency save filename
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            emergency_filename = f"{timestamp}_interrupted_ep{len(train_losses)}.pth"
            
            # Save the model
            torch.save(model.state_dict(), os.path.join(model_save_dir, emergency_filename))
            print(f"Model saved as: {emergency_filename}")
            print(f"Training completed {len(train_losses)} epochs before interruption")
        else:
            # Fallback save without parameters
            emergency_filename = f"interrupted_model_ep{len(train_losses)}.pth"
            torch.save(model.state_dict(), os.path.join(model_save_dir, emergency_filename))
            print(f"Model saved as: {emergency_filename}")
        
        print("=== TRAINING INTERRUPTED BY USER ===")
        
        # Return what we have so far
        return train_losses, val_losses, True  # True indicates interruption
    
    print("\n=== TRAINING COMPLETED NORMALLY ===")
    return train_losses, val_losses, False  # False indicates normal completion


def save_model(model, filepath='circle_cnn_model.pth', save_full=False, save_dir='models', **kwargs):
    """
    Save the trained model with structured filename including parameters
    
    Args:
        model: The trained model to save
        filepath: Base filename (used as fallback)
        save_full: Whether to save full model or just state dict
        save_dir: Directory to save models
        **kwargs: Training parameters (num_epochs, learning_rate, batch_size, etc.)
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Create structured filename if parameters are provided
    if kwargs:
        from datetime import datetime
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Extract common parameters
        params = []
        if 'num_epochs' in kwargs:
            params.append(f"ep{kwargs['num_epochs']}")
        if 'learning_rate' in kwargs:
            params.append(f"lr{kwargs['learning_rate']:.0e}")
        if 'batch_size' in kwargs:
            params.append(f"bs{kwargs['batch_size']}")
        if 'radius' in kwargs:
            params.append(f"r{kwargs['radius']}")
        if 'use_scheduler' in kwargs and kwargs['use_scheduler']:
            params.append("sched")
        if 'use_smooth_loss' in kwargs and kwargs['use_smooth_loss']:
            params.append("smooth")
            # Add smooth loss weights if available
            if 'mse_weight' in kwargs:
                params.append(f"mse{kwargs['mse_weight']}")
            if 'first_deriv_weight' in kwargs:
                params.append(f"fd{kwargs['first_deriv_weight']}")
            if 'second_deriv_weight' in kwargs:
                params.append(f"sd{kwargs['second_deriv_weight']}")
        
        # Add model architecture indicator
        if 'use_cnn_02' in kwargs and kwargs['use_cnn_02']:
            params.append("cnn02")
        
        # Create structured filename (timestamp + parameters only)
        if params:
            structured_name = f"{timestamp}_{'_'.join(params)}.pth"
        else:
            structured_name = f"{timestamp}.pth"
        
        full_path = os.path.join(save_dir, structured_name)
    else:
        # Use provided filepath as fallback
        full_path = os.path.join(save_dir, filepath)
    
    if save_full:
        # Save entire model (larger file, includes architecture)
        torch.save(model, full_path.replace('.pth', '_full.pth'))
        print(f"Full model saved as '{full_path.replace('.pth', '_full.pth')}'")
        return full_path.replace('.pth', '_full.pth')
    else:
        # Save only state dict (smaller file, requires model definition)
        torch.save(model.state_dict(), full_path)
        print(f"Model state dict saved as '{full_path}'")
        return full_path


def load_pretrained_model(filepath='circle_cnn_model.pth', input_length=128, output_length=128, 
                         load_dir='models', use_cnn_02=False):
    """Load a pretrained model"""
    
    # Automatically determine the correct directory based on model architecture
    if use_cnn_02:
        # Override load_dir if it's the default 'models' to use CNN_02 specific directory
        if load_dir == 'models':
            load_dir = 'models_cnn02'
    else:
        # For standard CNN, use regular models directory
        if load_dir == 'models_cnn02':
            load_dir = 'models'
    
    full_path = os.path.join(load_dir, filepath)
    
    # Create model with appropriate architecture
    if use_cnn_02:
        # from network import CircleCNN_02
        model = CircleCNN_02(input_length=input_length, output_length=output_length)
    else:
        model = CircleCNN(input_length=input_length, output_length=output_length)
    
    # Load the state dict
    try:
        pytorch_version = torch.__version__
        major, minor = map(int, pytorch_version.split('.')[:2])
        
        # weights_only parameter was introduced in PyTorch 1.13.0
        if major > 1 or (major == 1 and minor >= 13):
            model.load_state_dict(torch.load(full_path, map_location=device, weights_only=True))
        else:
            model.load_state_dict(torch.load(full_path, map_location=device))
            
        model = model.to(device)
        model.eval()  # Set to evaluation mode
        print(f"Model loaded successfully from '{full_path}'")
        return model
    except FileNotFoundError:
        print(f"Model file '{full_path}' not found!")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def inference_single_sample(model, input_signal):
    """Run inference on a single input sample"""
    model.eval()
    with torch.no_grad():
        if isinstance(input_signal, np.ndarray):
            input_signal = torch.tensor(input_signal, dtype=torch.float32)
        
        # Add batch dimension if needed
        if len(input_signal.shape) == 1:
            input_signal = input_signal.unsqueeze(0)
        
        input_signal = input_signal.to(device)
        prediction = model(input_signal).cpu()
        
        return prediction.squeeze(0)  # Remove batch dimension


def inference_batch(model, input_batch):
    """Run inference on a batch of inputs"""
    model.eval()
    with torch.no_grad():
        if isinstance(input_batch, np.ndarray):
            input_batch = torch.tensor(input_batch, dtype=torch.float32)
        
        input_batch = input_batch.to(device)
        predictions = model(input_batch).cpu()
        
        return predictions


def demo_pretrained_usage(model_path='circle_cnn_model.pth', load_dir='models', use_cnn_02=False):
    """Demonstrate how to use a pretrained model"""
    print("\n=== PRETRAINED MODEL DEMO ===")
    
    # Load pretrained model
    model = load_pretrained_model(model_path, load_dir=load_dir, use_cnn_02=use_cnn_02)
    
    if model is None:
        print("No pretrained model found. Train a model first!")
        return None
    
    # Create some test input (simulated time signal)
    test_input = torch.linspace(0, 0.64, 128) + 0.01 * torch.randn(128)
    
    # Single sample inference
    prediction = inference_single_sample(model, test_input)
    print(f"Prediction shape: {prediction.shape}")  # Should be (2, 128)
    
    # Extract kx and ky
    kx_pred = prediction[0].numpy()
    ky_pred = prediction[1].numpy()
    
    print(f"kx range: [{kx_pred.min():.3f}, {kx_pred.max():.3f}]")
    print(f"ky range: [{ky_pred.min():.3f}, {ky_pred.max():.3f}]")
    
    return model, prediction
