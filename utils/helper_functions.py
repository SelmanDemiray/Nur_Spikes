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

def load_image(file_path):
    """Loads an image, converts it to grayscale, resizes it, and returns it as a NumPy array."""
    try:
        img = Image.open(file_path).convert("L")  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to 28x28
        img_array = np.array(img).reshape(784, 1) / 255.0  # Normalize to 0-1

        # Verify shape
        if img_array.shape != (784, 1):
            raise ValueError(f"Unexpected image shape: {img_array.shape}")

        return img_array
    except Exception as e:
        raise ImageLoadError(file_path, str(e))

def rates(images, W0, W1, parameters):
    """Calculate rates for both layers with shape verification."""
    # Convert to numpy arrays if using CuPy
    if USE_CUPY:
        images = cp.asarray(images)
        W0 = cp.asarray(W0)
        W1 = cp.asarray(W1)

    # Ensure images is 2D with shape (784, 1)
    if len(images.shape) == 1:
        images = images.reshape(-1, 1)
    if images.shape[0] != 784:
        images = images.reshape(784, -1)

    # Verify shapes before computation
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

def record_live(W0, W1, parameters, input_queue, output_queue, param_queue):
    """Continuously generates spikes with comprehensive error handling."""
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

            output_queue.put((S_2, R1, S_1, R2, time.time()))

    except SNNError as e:
        print(f"\n{e.message}")
        if 'images' in locals():
            print(f"Current shapes - images: {images.shape}")
        print(f"W0: {W0.shape}, W1: {W1.shape}")
    except Exception as e:
        print(f"\nUnhandled error: {str(e)}")
        import traceback
        traceback.print_exc()