import numpy as np
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from scipy.stats import poisson

# Try importing CuPy
try:
    import cupy as cp
    USE_CUPY = True
except ImportError:
    USE_CUPY = False

# Custom exceptions for SNN-specific errors
class SNNError(Exception):
    """Base exception class for SNN-related errors."""
    def __init__(self, code, message):
        self.code = code
        self.message = f"Error {code}: {message}"
        super().__init__(self.message)

class DimensionError(SNNError):
    """Exception for matrix dimension mismatches."""
    def __init__(self, expected_shape, actual_shape, operation, component=""):
        code = "SNN-001"
        message = f"Dimension mismatch in {component} during {operation}. Expected shape: {expected_shape}, Got: {actual_shape}"
        super().__init__(code, message)

class WeightLoadError(SNNError):
    """Exception for weight loading failures."""
    def __init__(self, filename, error_details):
        code = "SNN-002"
        message = f"Failed to load weights from {filename}: {error_details}"
        super().__init__(code, message)

class ImageLoadError(SNNError):
    """Exception for image loading failures."""
    def __init__(self, filename, error_details):
        code = "SNN-003"
        message = f"Failed to load image from {filename}: {error_details}"
        super().__init__(code, message)

class ComputationError(SNNError):
    """Exception for computation failures."""
    def __init__(self, operation, error_details):
        code = "SNN-004"
        message = f"Computation error during {operation}: {error_details}"
        super().__init__(code, message)

class QueueError(SNNError):
    """Exception for queue-related failures."""
    def __init__(self, queue_type, operation, error_details):
        code = "SNN-005"
        message = f"Queue error in {queue_type} during {operation}: {error_details}"
        super().__init__(code, message)

def verify_dimensions(name, tensor, expected_shape):
    """Verify tensor dimensions match expected shape."""
    if tensor.shape != expected_shape:
        raise DimensionError(expected_shape, tensor.shape, "shape verification", name)

def safe_matrix_operation(operation, *args, **kwargs):
    """Safely perform matrix operations with dimension checking."""
    try:
        result = operation(*args, **kwargs)
        return result
    except ValueError as e:
        raise ComputationError(operation.__name__, str(e))
    except Exception as e:
        raise ComputationError(operation.__name__, f"Unexpected error: {str(e)}")

def load_weights(file_path):
    """Loads weights from a .npz file with error handling."""
    try:
        with np.load(file_path) as data:
            W0 = data["W0"]
            W1 = data["W1"]

            # Verify dimensions
            verify_dimensions("W0", W0, (500, 784))
            verify_dimensions("W1", W1, (10, 500))

            return W0, W1
    except FileNotFoundError:
        raise WeightLoadError(file_path, "File not found")
    except Exception as e:
        raise WeightLoadError(file_path, str(e))

def save_weights(file_path, W0, W1):
    """Saves weights to a .npz file with error handling."""
    try:
        np.savez(file_path, W0=W0, W1=W1)
    except Exception as e:
        raise WeightLoadError(file_path, str(e))

def load_image(file_path):
    """Loads an image, converts it to grayscale, resizes it, and returns it as a NumPy array."""
    try:
        img = Image.open(file_path).convert("L") 
        img = img.resize((28, 28))  
        img_array = np.array(img).reshape(784, 1) / 255.0 

        # Verify shape
        if img_array.shape != (784, 1):
            raise ValueError(f"Unexpected image shape: {img_array.shape}")

        return img_array
    except Exception as e:
        raise ImageLoadError(file_path, str(e))

def save_image(file_path, img_array):
    """Saves a NumPy array as an image file."""
    try:
        img = Image.fromarray((img_array * 255).astype(np.uint8).reshape(28, 28))
        img.save(file_path)
    except Exception as e:
        raise ImageLoadError(file_path, str(e))
    
def get_prediction(output_spikes):
    # Count the number of spikes for each class
    spike_counts = np.sum(output_spikes, axis=1)  # Sum along the time axis

    # Find the class with the maximum spike count
    predicted_class = np.argmax(spike_counts)

    # Calculate confidence (percentage of spikes for the predicted class)
    total_spikes = np.sum(spike_counts)
    confidence = (spike_counts[predicted_class] / total_spikes) * 100 if total_spikes > 0 else 0

    return predicted_class, confidence

def rates(images, W0, W1, parameters):
    """Calculate rates for both layers with shape verification."""
    # Convert to numpy arrays if using CuPy
    if USE_CUPY:
        images = cp.asarray(images)
        W0 = cp.asarray(W0)
        W1 = cp.asarray(W1)

    # Handle both single inputs and batches
    is_batch = len(images.shape) > 1 and images.shape[1] > 1
    if is_batch:
        if images.shape[0] != 784:
            images = images.reshape(784, -1)
    else:
        if len(images.shape) == 1:
            images = images.reshape(-1, 1)
        if images.shape[0] != 784:
            images = images.reshape(784, -1)
        verify_dimensions("images", images, (784, 1))

    verify_dimensions("W0", W0, (500, 784))
    verify_dimensions("W1", W1, (10, 500))

    # Compute rates
    if USE_CUPY:
        R1 = cp.maximum(0, cp.dot(W0, images))
        R2 = cp.maximum(0, cp.dot(W1, R1))
        return cp.asnumpy(R1), cp.asnumpy(R2)
    else:
        R1 = np.maximum(0, W0 @ images)
        R2 = np.maximum(0, W1 @ R1)
        return R1, R2

class PerformanceMonitor:
    """Monitors and logs SNN performance metrics."""
    def __init__(self, window_size=100):
        self.window_size = window_size
        self.spike_history = []
        self.prediction_accuracy = []
        self.computation_times = []
        self.memory_usage = []
        self.last_update = time.time()
        
    def update(self, spikes, prediction, computation_time):
        if spikes is not None:
            self.spike_history.append(float(np.mean(spikes)))
            self.computation_times.append(float(computation_time))
            
            # Keep only recent history
            if len(self.spike_history) > self.window_size:
                self.spike_history = self.spike_history[-self.window_size:]
                self.computation_times = self.computation_times[-self.window_size:]
            
            try:
                import psutil
                self.memory_usage.append(psutil.Process().memory_info().rss / 1024 / 1024)
            except ImportError:
                self.memory_usage.append(0)
            
            self.last_update = time.time()

    def get_stats(self):
        current_time = time.time()
        if not self.spike_history or (current_time - self.last_update) > 5:  # Reset if stale
            return {
                'avg_spike_rate': 0.0,
                'avg_computation_time': 0.0,
                'avg_memory_usage': 0.0,
                'active': False
            }
            
        return {
            'avg_spike_rate': float(np.mean(self.spike_history[-10:])),  # Average of last 10 samples
            'avg_computation_time': float(np.mean(self.computation_times[-10:])),
            'avg_memory_usage': float(np.mean(self.memory_usage[-10:])) if self.memory_usage else 0.0,
            'active': True
        }

class AdaptiveLearning:
    """Implements adaptive learning rate and homeostatic plasticity."""
    def __init__(self, initial_rate=0.01, min_rate=0.001, max_rate=0.1):
        self.learning_rate = initial_rate
        self.min_rate = min_rate
        self.max_rate = max_rate
        self.history = []
        self.target_activity = 0.1  # Target spike rate
        
    def adapt(self, performance_stats):
        """Adjust learning rate based on network performance."""
        current_rate = performance_stats['avg_spike_rate']
        if abs(current_rate - self.target_activity) > 0.05:
            self.learning_rate *= 0.95
        else:
            self.learning_rate *= 1.05
        self.learning_rate = np.clip(self.learning_rate, self.min_rate, self.max_rate)
        return self.learning_rate

class HomeostaticPlasticity:
    """Implements homeostatic plasticity mechanisms."""
    def __init__(self, target_rate=10):
        self.target_rate = target_rate
        self.time_constant = 1000.0  # ms
        self.scaling_factor = 1.0
        
    def adjust_weights(self, W, spike_rates):
        """Scale weights to maintain target firing rate."""
        if USE_CUPY:
            spike_rates = cp.asarray(spike_rates)
            W = cp.asarray(W)
        
        rate_ratio = self.target_rate / (spike_rates.mean() + 1e-6)
        self.scaling_factor *= np.sqrt(rate_ratio)
        adjusted_W = W * self.scaling_factor
        
        if USE_CUPY:
            return cp.asnumpy(adjusted_W)
        return adjusted_W

class BatchProcessor:
    """Handles batch processing of inputs."""
    def __init__(self, batch_size=32):
        self.batch_size = batch_size
        self.current_batch = []
        
    def add_to_batch(self, input_data):
        self.current_batch.append(input_data)
        if len(self.current_batch) >= self.batch_size:
            batch = np.stack(self.current_batch)
            self.current_batch = []
            return batch
        return None

def export_network_visualization(W0, W1, spike_history, filepath):
    """Export network visualization as an interactive HTML file."""
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        # Create weight visualizations
        fig = make_subplots(rows=2, cols=2)
        
        # Add weight matrices
        fig.add_trace(
            go.Heatmap(z=W0, colorscale='Viridis'),
            row=1, col=1
        )
        fig.add_trace(
            go.Heatmap(z=W1, colorscale='Viridis'),
            row=1, col=2
        )
        
        # Add spike history
        fig.add_trace(
            go.Scatter(y=spike_history, mode='lines'),
            row=2, col=1
        )
        
        fig.write_html(filepath)
        return True
    except ImportError:
        print("Plotly is required for interactive visualization export")
        return False

def update_network_parameters(parameters, performance_stats):
    """Dynamically update network parameters based on performance."""
    if performance_stats['avg_computation_time'] > 0.1:
        parameters['batch_size'] = max(1, parameters['batch_size'] - 1)
    
    if performance_stats['avg_spike_rate'] < parameters['target_rate']:
        parameters['gain'] *= 1.1
    elif performance_stats['avg_spike_rate'] > parameters['target_rate'] * 1.5:
        parameters['gain'] *= 0.9
        
    return parameters

def optimize_weights(W0, W1, target_spikes, learning_rate=0.001, max_iterations=1000):
    """Enhanced weight optimization with early stopping."""
    adaptive_learning = AdaptiveLearning(initial_rate=learning_rate)
    homeostatic = HomeostaticPlasticity()
    
    best_loss = float('inf')
    best_W0 = W0.copy()
    best_W1 = W1.copy()
    patience = 20
    patience_counter = 0
    current_lr = learning_rate
    momentum = 0.9
    prev_grad_W0 = 0
    prev_grad_W1 = 0
    
    # Initialize target matrix
    target = np.zeros((10, 1))
    target[target_spikes % 10, 0] = 1.0
    
    # Parameters for optimization
    parameters = {
        "dt": 0.005,
        "gain": 1.0,
        "nrep": 5,
        "batch_size": 1
    }
    
    # Small random noise to break symmetry
    dummy_input = np.random.normal(0, 0.1, (784, 1))
    
    try:
        for i in range(max_iterations):
            # Forward pass
            with np.errstate(all='ignore'):  # Suppress numpy warnings
                R1, R2 = rates(dummy_input, W0, W1, parameters)
                
                # Normalize activations
                R2_max = np.max(R2, axis=0, keepdims=True)
                R2_norm = np.exp(R2 - R2_max)
                R2_sum = np.sum(R2_norm, axis=0, keepdims=True) + 1e-12
                R2_softmax = R2_norm / R2_sum
                
                # Calculate loss
                eps = 1e-12
                ce_loss = -np.sum(target * np.log(R2_softmax + eps))
                reg_loss = 0.0001 * (np.sum(W0**2) + np.sum(W1**2))
                current_loss = ce_loss + reg_loss
                
                # Compute gradients with momentum
                error = R2_softmax - target
                grad_W1 = np.dot(error, R1.T) + 0.0001 * W1
                grad_W0 = np.dot(np.dot(W1.T, error), dummy_input.T) + 0.0001 * W0
                
                # Apply momentum
                grad_W0 = momentum * prev_grad_W0 + (1 - momentum) * grad_W0
                grad_W1 = momentum * prev_grad_W1 + (1 - momentum) * grad_W1
                
                # Store gradients for next iteration
                prev_grad_W0 = grad_W0
                prev_grad_W1 = grad_W1
                
                # Update weights with clipping
                W0 = W0 - current_lr * np.clip(grad_W0, -1, 1)
                W1 = W1 - current_lr * np.clip(grad_W1, -1, 1)
                
                # Normalize weights
                W0 = W0 / (np.std(W0) + eps)
                W1 = W1 / (np.std(W1) + eps)
            
            if i % 10 == 0:
                print(f"Iteration {i}, Loss: {current_loss:.6f}, Learning Rate: {current_lr:.6f}")
            
            if current_loss < best_loss:
                best_loss = current_loss
                best_W0 = W0.copy()
                best_W1 = W1.copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Early stopping at iteration {i} due to no improvement")
                break
            
            # Decay learning rate
            current_lr *= 0.999
            
    except Exception as e:
        print(f"Optimization error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"Final loss: {best_loss:.6f}")
    return best_W0, best_W1

def auto_tune_parameters(parameters, performance_stats):
    """Automatically tune SNN parameters based on performance."""
    if performance_stats['avg_spike_rate'] < 0.1:
        parameters['gain'] *= 1.1
    elif performance_stats['avg_spike_rate'] > 10:
        parameters['gain'] *= 0.9
    
    if performance_stats['avg_computation_time'] > 0.1:
        parameters['nrep'] = max(1, parameters['nrep'] - 1)
    
    return parameters

def record_live(W0, W1, parameters, input_queue, output_queue, param_queue, prediction_history_queue):
    """Continuously generates spikes with comprehensive error handling."""
    prediction_counts = {i: 0 for i in range(10)}  # Initialize counts for each class
    performance_monitor = PerformanceMonitor()  # Initialize performance monitor
    try:
        while True:
            # Get updated parameters
            try:
                updated_params = param_queue.get_nowait()
                parameters["dt"] = updated_params["dt"]
                parameters["gain"] = updated_params["gain"]
            except queue.Empty:
                pass

            # Get input image
            try:
                images = input_queue.get(timeout=1)
            except queue.Empty:
                images = np.random.poisson(lam=2, size=(784, 1))

            if images is None:
                break
                
            # Convert to CuPy array if USE_CUPY is True
            if USE_CUPY:
                if not isinstance(images, cp.ndarray):
                    images = cp.asarray(images)
                if not isinstance(W0, cp.ndarray):
                    W0 = cp.asarray(W0)
                if not isinstance(W1, cp.ndarray):
                    W1 = cp.asarray(W1)

            # Calculate rates safely
            start_time = time.time()
            R1, R2 = rates(images, W0, W1, parameters)
            
            # Convert R1 and R2 to NumPy arrays if using CuPy
            if USE_CUPY:
                if isinstance(R1, cp.ndarray):
                    R1 = cp.asnumpy(R1)
                if isinstance(R2, cp.ndarray):
                    R2 = cp.asnumpy(R2)

            # Generate spikes
            mu_1 = parameters["dt"] * parameters["gain"] * R1
            mu_2 = parameters["dt"] * parameters["gain"] * R2

            mu_1 = safe_matrix_operation(np.tile, mu_1, (1, parameters["nrep"]))
            mu_2 = safe_matrix_operation(np.tile, mu_2, (1, parameters["nrep"]))

            S_1 = np.random.poisson(lam=mu_1)
            S_2 = np.random.poisson(lam=mu_2)

            # Verify output dimensions
            verify_dimensions("S_1", S_1, (500, parameters["nrep"]))
            verify_dimensions("S_2", S_2, (10, parameters["nrep"]))
            
            # Get prediction and update prediction counts
            predicted_class, _ = get_prediction(S_2)
            prediction_counts[predicted_class] += 1
            
            # Send updated prediction counts to the GUI
            prediction_history_queue.put(prediction_counts)

            computation_time = time.time() - start_time
            performance_monitor.update(S_2, predicted_class, computation_time)

            output_queue.put((S_2, R1, S_1, R2, time.time()))

            # Auto-tune parameters based on performance
            parameters = auto_tune_parameters(parameters, performance_monitor.get_stats())

    except SNNError as e:
        print(f"\n{e.message}")
        if 'images' in locals():
            print(f"Current shapes - images: {images.shape}")
        print(f"W0: {W0.shape}, W1: {W1.shape}")
    except Exception as e:
        print(f"\nUnhandled error: {str(e)}")
        import traceback
        traceback.print_exc()

def safe_normalize(array, eps=1e-8):
    """Safely normalize an array to [0, 1] range with epsilon to prevent divide by zero."""
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
    array_min = array.min()
    array_max = array.max()
    denominator = array_max - array_min
    if denominator < eps:
        return np.zeros_like(array)
    normalized = (array - array_min) / (denominator + eps)
    return normalized

def advanced_metrics(spike_data):
    """Compute extended metrics for spikes."""
    avg_spikes = spike_data.mean()
    std_spikes = spike_data.std()
    return {
        "avg_spikes": avg_spikes,
        "std_spikes": std_spikes
    }

def debug_print(message: str):
    """Print debug messages to the debug terminal."""
    print(message)  # Ensure this is redirected to the GUI's debug terminal

def calculate_spike_rate(spikes, elapsed_time):
    """Calculate spike rate given spike data and elapsed time."""
    total_spikes = np.sum(spikes)
    return total_spikes / elapsed_time if elapsed_time > 0 else 0
