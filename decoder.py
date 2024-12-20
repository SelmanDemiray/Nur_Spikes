import argparse
import gzip
import json
import logging
import os
import queue
import shutil
import threading
import time
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import requests
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, UnidentifiedImageError
from scipy.special import logsumexp
from scipy.stats import kurtosis, poisson, skew
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

# Try importing CuPy
try:
    import cupy as cp

    USE_CUPY = True
except ImportError:
    USE_CUPY = False

from utils.helper_functions import (
    confusion,
    loglikelihood,
    logprior,
    posaverage,
    posterior,
    rates,
    record,
    download_extract_mnist,
    spikes,
    load_mnist_images,
    load_cifar10_images,
)

# Configure logging
logging.basicConfig(
    filename="snn_experiment.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# Log which library is being used
logging.info(f"{'CuPy' if USE_CUPY else 'NumPy'} is being used.")

# Initialize W0 and W1 with original initialization
default_W0 = np.random.normal(0, 1, size=(500, 784))
default_W1 = np.random.uniform(-1, 1, size=(10, 500))


def analyze_mat_weights(file_path):
    """
    Analyzes a .npz file containing weights.

    Args:
        file_path: Path to the .npz file.

    Returns:
        None. Prints detailed information about the weights.
    """
    try:
        with np.load(file_path) as data:  # Use np.load for .npz files
            for var_name in data.files:
                weight_data = data[var_name]
                analyze_weight_data(var_name, weight_data)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return


def analyze_weight_data(var_name, weight_data):
    """
    Analyzes a single weight variable.

    Args:
        var_name: Name of the weight variable.
        weight_data: Numpy array containing the weight data.

    Returns:
        None. Prints detailed information about the weight data.
    """
    # Basic information
    print(f"    - {var_name}:")
    print(f"        - Shape: {weight_data.shape}")
    print(f"        - Size: {weight_data.size} elements")
    print(f"        - Data type: {weight_data.dtype}")

    # Statistical properties
    print(f"        - Mean: {np.mean(weight_data):.4f}")
    print(f"        - Standard deviation: {np.std(weight_data):.4f}")
    print(f"        - Minimum: {np.min(weight_data):.4f}")
    print(f"        - Maximum: {np.max(weight_data):.4f}")
    percentiles = np.percentile(
        weight_data, [0, 1, 5, 25, 50, 75, 95, 99, 100]
    )
    print(
        f"        - Percentiles (0th, 1st, 5th, 25th, 50th, 75th, 95th, 99th, 100th): {percentiles}"
    )
    print(f"        - Skewness: {skew(weight_data.flatten()):.4f}")
    print(f"        - Kurtosis: {kurtosis(weight_data.flatten()):.4f}")

    # Sparsity
    if np.issubdtype(weight_data.dtype, np.integer) or np.issubdtype(
        weight_data.dtype, np.floating
    ):
        zero_count = np.count_nonzero(weight_data == 0)
        sparsity = zero_count / weight_data.size
        print(f"        - Sparsity: {sparsity:.2%}")

    # Histogram
    plt.hist(weight_data.flatten(), bins=50)
    plt.title(f"Histogram of weights for {var_name}")
    plt.xlabel("Weight Value")
    plt.ylabel("Frequency")
    plt.show()

    # Weight distribution for each neuron/filter (for 2D weight data)
    if len(weight_data.shape) == 2:
        plt.figure(figsize=(10, 5))
        for i in range(
            min(weight_data.shape[0], 50)
        ):  # Limit to 50 neurons for visualization
            plt.plot(weight_data[i, :], label=f"Neuron {i+1}")
        plt.title(f"Weight Distribution for Each Neuron ({var_name})")
        plt.xlabel("Input Feature")
        plt.ylabel("Weight Value")
        plt.legend(loc="upper right", ncol=2, fontsize="small")
        plt.show()


def load_weights(file_path):
    """Loads weights from a .npz file."""
    try:
        with np.load(file_path) as data:
            W0 = data["W0"]
            W1 = data["W1"]
        return W0, W1
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Error loading weights: {e}")
        raise


def show_cm(cm, vis="on"):
    """Plots a confusion matrix."""
    if not (isinstance(cm, np.ndarray) and cm.shape[0] == cm.shape[1]):
        logging.warning(
            "Invalid confusion matrix provided. Skipping visualization."
        )
        return

    num_categories = cm.shape[0]
    plt.figure()
    plt.imshow(100 * cm, cmap="jet")
    plt.xticks(np.arange(num_categories), np.arange(num_categories))
    plt.yticks(np.arange(num_categories), np.arange(num_categories))
    plt.xlabel("Decoded image category")
    plt.ylabel("Presented image category")
    cbar = plt.colorbar()
    cbar.set_label("Categorization frequency (%)")
    plt.set_cmap("jet")

    if vis == "on":
        plt.show()


def show_images(images, category=None, num_cols=10):
    """Displays images from a dataset."""
    if not isinstance(images, np.ndarray):
        logging.warning("Invalid image data provided. Skipping visualization.")
        return

    if category is not None:
        if not (
            isinstance(category, int)
            and 0 <= category <= (images.shape[1] / 100) - 1
        ):
            logging.warning("Invalid category value. Displaying all images.")
            category = None

    if category is not None:
        num_images = 100
        start_index = category * 100
        end_index = (category + 1) * 100
        images = images[:, start_index:end_index]
    else:
        num_images = images.shape[1]

    num_rows = int(np.ceil(num_images / num_cols))
    plt.figure()
    plt.set_cmap("gray")

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(
            images[:, i].reshape(
                int(np.sqrt(images.shape[0])), int(np.sqrt(images.shape[0]))
            ).T
        )
        plt.axis("equal")
        plt.axis("off")

    if category is not None:
        plt.suptitle(f"Category {category} images")
    else:
        plt.suptitle("All images")
    plt.show()


def show_weights(weights, num_cols=25):
    """Displays all the receptive field structures as gray images."""
    if not isinstance(weights, np.ndarray):
        logging.warning(
            "Invalid weight data provided. Skipping visualization."
        )
        return

    num_fields = weights.shape[0]
    num_rows = int(np.ceil(num_fields / num_cols))
    plt.figure()
    plt.set_cmap("gray")

    for i in range(num_fields):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(
            weights[i, :].reshape(
                int(np.sqrt(weights.shape[1])), int(np.sqrt(weights.shape[1]))
            ).T
        )
        plt.axis("equal")
        plt.axis("off")

    plt.suptitle("Receptive fields")
    plt.show()


def show_spikes(activity, parameters):
    """Displays a raster plot of spike activity."""
    plt.figure()

    # Optimized plotting: Collect all spike times and plot once per neuron
    for neuron_idx in range(activity.shape[0]):
        spike_times = (
            np.where(activity[neuron_idx, :, :] == 1)[0] * parameters["dt"]
        )
        plt.plot(
            spike_times,
            np.ones_like(spike_times) * neuron_idx,
            "|k",
            markersize=1,
        )  # Smaller marker size

    plt.xlabel("Time (s)")
    plt.ylabel("Neuron Index")
    plt.title("Spike Raster Plot")
    plt.show()


def show_spike_histogram(activity, parameters):
    """Displays a histogram of spike counts across neurons."""
    plt.figure()
    all_spike_counts = np.sum(
        activity, axis=(1, 2)
    )  # Sum across images and repetitions
    plt.hist(all_spike_counts, bins=20)
    plt.xlabel("Spike Count")
    plt.ylabel("Number of Neurons")
    plt.title("Spike Count Histogram")
    plt.show()


def show_spike_rate_plot(activity, parameters):
    """Displays a plot of spike rates over time, averaged across neurons."""
    plt.figure()
    spike_rates = np.mean(activity, axis=0)  # Average across neurons
    time_axis = np.arange(activity.shape[1]) * parameters["dt"]
    for rep in range(activity.shape[2]):
        plt.plot(time_axis, spike_rates[:, rep])
    plt.xlabel("Time (s)")
    plt.ylabel("Average Spike Rate")
    plt.title("Average Spike Rate Over Time")
    plt.show()


def show_neuron_heatmap(activity, parameters):
    """Displays a heatmap of spike activity for each neuron."""
    plt.figure()
    plt.imshow(
        np.sum(activity, axis=2),
        cmap="hot",
        interpolation="nearest",
        aspect="auto",
    )
    plt.xlabel("Image Index")
    plt.ylabel("Neuron Index")
    plt.title("Neuron Spike Activity Heatmap")
    plt.colorbar()
    plt.show()


def load_images(image_paths):
    """Loads images from a list of file paths."""
    try:
        # Assuming all images have the same size and mode
        return np.array(
            [
                np.array(Image.open(path).convert("L")).flatten()
                for path in image_paths
            ]
        ).T
    except (FileNotFoundError, UnidentifiedImageError) as e:
        logging.warning(f"Error loading images: {e}")
        return np.array([])  # Return an empty array


def calculate_category_priors(labels):
    """Calculates category priors based on label frequencies."""
    if not isinstance(labels, np.ndarray):
        logging.warning("Invalid label data provided. Using uniform priors.")
        return np.full(10, 1 / 10)

    label_counts = Counter(labels)
    num_images = len(labels)
    return np.array(
        [label_counts[i] / num_images for i in range(len(label_counts))]
    )


def run_experiment(images, labels, parameters, spike_formats, W0, W1):
    """Runs a single decoding experiment."""
    cp = calculate_category_priors(labels)

    logging.info("Beginning decoding experiment:")
    logging.info(f"Parameters: {parameters}")

    activity = record(images, W0, W1, parameters)

    for spike_format in spike_formats:
        if spike_format == "raster":
            show_spikes(activity, parameters)
        elif spike_format == "histogram":
            show_spike_histogram(activity, parameters)
        elif spike_format == "rate":
            show_spike_rate_plot(activity, parameters)
        elif spike_format == "heatmap":
            show_neuron_heatmap(activity, parameters)

    ll = loglikelihood(activity, images, W0, W1, parameters)
    lp = logprior(cp, images.shape[1])
    pos = posterior(ll, lp)
    cm, ind = confusion(pos)
    pa = posaverage(images, pos, 10)

    # Ensure the 'figures' directory exists
    os.makedirs("figures", exist_ok=True)
    filename = f"figures/case_dt{parameters['dt']:.3f}_gain{parameters['gain']:.1f}_nreps{parameters['nrep']}"

    show_cm(cm, vis="off")
    try:
        plt.savefig(f"{filename}.png", dpi=300)
    except Exception as e:
        logging.warning(f"Error saving figure: {e}")
    plt.close()

    show_images(images)
    show_images(pa)

    logging.info("Finished decoding experiment.")


def record_live(W0, W1, parameters, input_queue, output_queue, param_queue):
    """Continuously generates spikes for live decoding with parameter updates."""
    try:
        while True:
            # Get updated parameters from the queue
            try:
                updated_params = param_queue.get_nowait()
                parameters["dt"] = updated_params["dt"]
                parameters["gain"] = updated_params["gain"]
            except queue.Empty:
                pass  # No new parameters, continue with the current ones

            try:
                images = input_queue.get(
                    timeout=1
                )  # Try to get image from queue with a timeout
            except queue.Empty:
                # Replace random input with Poisson-distributed noise
                images = np.random.poisson(
                    lam=2, size=(784, 1)
                )  # Increased lam to 2

            if images is None:
                break

            # Calculate activities for both layers
            R1, R2 = rates(images, W0, W1, parameters)
            print(f"Shape of R1: {R1.shape}")
            print(f"R1 values (first 5): {R1[:5, 0]}") # Print first 5 values
            print(f"Shape of R2: {R2.shape}")
            print(f"R2 values (first 5): {R2[:5, 0]}")

            R2_positive = np.maximum(R2, 0)
            print(f"Shape of R2_positive: {R2_positive.shape}")
            print(f"R2_positive values (first 5): {R2_positive[:5, 0]}")

            # Ensure that mu has the correct shape for broadcasting
            mu = parameters["dt"] * parameters["gain"] * R2_positive
            mu = np.tile(mu, (1, parameters["nrep"]))
            print(f"Shape of mu: {mu.shape}")
            print(f"mu values (first 5): {mu[:5, 0]}")

            # Check if mu is all zeros or very close to zero
            if np.allclose(mu, 0):
                print("WARNING: mu is (close to) zero, no spikes will be generated!")

            S = poisson.rvs(
                mu=mu, size=(R2_positive.shape[0], parameters["nrep"])
            )  # Generate spikes
            print(f"Shape of S: {S.shape}")
            print(f"S values (first 5): {S[:5, 0]}")

            # Put activities, spikes, and timestamp into the queue
            output_queue.put((S, R1, R2_positive, time.time()))

    except Exception as e:
        logging.error(f"Error in record_live thread: {e}")


def run_experiment_live(default_W0, default_W1, parameters, record_live):
    """
    Runs a live decoding experiment with a GUI.
    """

    # Initialize weights and input as random
    W0 = default_W0.copy()
    W1 = default_W1.copy()
    static_input = np.random.poisson(lam=0.5, size=(784, 1))
    weights_loaded = False
    image_loaded = False

    input_queue = queue.Queue()
    output_queue = queue.Queue()
    param_queue = queue.Queue()  # Create parameter update queue

    root = tk.Tk()
    root.title("Live SNN Experiment")

    # Add a label to display the used library
    library_label = tk.Label(
        root, text=f"Using {'CuPy' if USE_CUPY else 'NumPy'}"
    )
    library_label.pack()

    # Display initial parameters
    params_label = tk.Label(root, text=f"Parameters: {parameters}")
    params_label.pack()

    # Create two neuron windows for separate layers
    neuron_window_1 = tk.Toplevel(root)
    neuron_window_1.title("Neuron Activity - Layer 1")
    neuron_window_2 = tk.Toplevel(root)
    neuron_window_2.title("Neuron Activity - Layer 2")

    # Calculate canvas dimensions based on the number of neurons in each layer
    num_neurons_1 = default_W0.shape[0]
    grid_size_1 = int(np.ceil(np.sqrt(num_neurons_1)))
    canvas_width_1 = grid_size_1 * 50
    canvas_height_1 = grid_size_1 * 50

    num_neurons_2 = default_W1.shape[0]
    grid_size_2 = int(np.ceil(np.sqrt(num_neurons_2)))
    canvas_width_2 = grid_size_2 * 50
    canvas_height_2 = grid_size_2 * 50

    neuron_canvas_1 = tk.Canvas(
        neuron_window_1, width=canvas_width_1, height=canvas_height_1
    )
    neuron_canvas_1.pack()
    neuron_canvas_2 = tk.Canvas(
        neuron_window_2, width=canvas_width_2, height=canvas_height_2
    )
    neuron_canvas_2.pack()

    neuron_rects_1 = []
    for i in range(num_neurons_1):
        row = i // grid_size_1
        col = i % grid_size_1
        x1 = col * 50
        y1 = row * 50
        x2 = x1 + 40
        y2 = y1 + 40
        rect = neuron_canvas_1.create_rectangle(x1, y1, x2, y2, fill="blue")
        neuron_rects_1.append(rect)

    neuron_rects_2 = []
    for i in range(num_neurons_2):
        row = i // grid_size_2
        col = i % grid_size_2
        x1 = col * 50
        y1 = row * 50
        x2 = x1 + 40
        y2 = y1 + 40
        rect = neuron_canvas_2.create_rectangle(x1, y1, x2, y2, fill="blue")
        neuron_rects_2.append(rect)

    spike_count_label = tk.Label(root, text="Spike Count: 0")
    spike_count_label.pack()
    total_spike_count_label = tk.Label(root, text="Total Spike Count: 0")
    total_spike_count_label.pack()
    spike_rate_label = tk.Label(root, text="Spike Rate: 0 Hz")
    spike_rate_label.pack()

    # Create raster plot figures and canvases for both layers
    fig1, ax1 = plt.subplots()
    raster_plot1 = FigureCanvasTkAgg(fig1, master=root)
    raster_plot1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Neuron Index")
    ax1.set_title("Live Spike Raster Plot - Layer 1")

    fig2, ax2 = plt.subplots()
    raster_plot2 = FigureCanvasTkAgg(fig2, master=root)
    raster_plot2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Neuron Index")
    ax2.set_title("Live Spike Raster Plot - Layer 2")

    image_frame = tk.Frame(root)
    image_frame.pack()
    image_label = tk.Label(image_frame, text="Using Random Noise Input")
    image_label.pack()

    def upload_image():
        nonlocal static_input, image_loaded
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            try:
                image = Image.open(file_path).convert("L")
                image = image.resize((28, 28))
                static_input = (
                    np.array(image).flatten().reshape(784, 1) / 255.0
                )
                image_label.config(text=file_path)
                image_loaded = True
            except Exception as e:
                logging.warning(f"Error loading image: {e}")
                image_label.config(text="Error loading image")
        else:
            image_label.config(text="No Image Loaded")

    def remove_image():
        nonlocal static_input, image_loaded
        static_input = np.random.poisson(lam=0.5, size=(784, 1))
        image_loaded = False
        image_label.config(text="Using Random Noise Input")

    # Add this function to restore the static image
    def add_static_image():
        nonlocal static_input, image_loaded
        static_input = np.random.poisson(lam=2, size=(784, 1))
        image_loaded = False  # Keep it as noise input, but use the static_input
        image_label.config(text="Using Static Random Image")

    upload_button = tk.Button(root, text="Upload Image", command=upload_image)
    upload_button.pack()

    remove_button = tk.Button(root, text="Remove Image", command=remove_image)
    remove_button.pack()

    # Add the button to restore the static image
    add_static_button = tk.Button(root, text="Add Static Image", command=add_static_image)
    add_static_button.pack()

    # Create sliders for dt and gain
    dt_label = ttk.Label(root, text="Time Step (dt):")
    dt_label.pack(padx=5, pady=5)
    dt_var = tk.DoubleVar(value=parameters["dt"])
    dt_slider = ttk.Scale(
        root,
        from_=0.001,
        to=0.01,
        orient="horizontal",
        variable=dt_var,
    )
    dt_slider.pack(padx=5, pady=5)

    gain_label = ttk.Label(root, text="Gain:")
    gain_label.pack(padx=5, pady=5)
    gain_var = tk.DoubleVar(value=parameters["gain"])
    gain_slider = ttk.Scale(
        root,
        from_=0.1,
        to=5.0,  # Increased gain range
        orient="horizontal",
        variable=gain_var,
    )
    gain_slider.pack(padx=5, pady=5)

    # Function to load weights from file
    def load_weights_from_file():
        nonlocal W0, W1, weights_loaded
        file_path = filedialog.askopenfilename(
            filetypes=[("Numpy files", "*.npz")]
        )
        if file_path:
            try:
                W0, W1 = load_weights(file_path)
                weights_loaded = True
                print(f"Loaded weights from {file_path}")
            except Exception as e:
                weights_loaded = False
                logging.error(f"Error loading weights: {e}")
                print(f"Error loading weights: {e}")

    def remove_weights():
        nonlocal W0, W1, weights_loaded
        W0 = default_W0.copy()
        W1 = default_W1.copy()
        weights_loaded = False
        print("Using random generated weights")

        # Reset neuron colors to blue
        for rect in neuron_rects_1:
            neuron_canvas_1.itemconfig(rect, fill="blue")
        for rect in neuron_rects_2:
            neuron_canvas_2.itemconfig(rect, fill="blue")

    # Buttons to switch between weight options

    weights_label = tk.Label(root, text="Using Random Generated Weights")
    weights_label.pack()

    def use_random_weights():
        nonlocal W0, W1, weights_loaded
        W0 = default_W0.copy()
        W1 = default_W1.copy()
        weights_loaded = False
        weights_label.config(text="Using Random Generated Weights")
        print("Using random generated weights")

    def use_trained_weights():
        nonlocal weights_loaded
        load_weights_from_file()
        if weights_loaded:
            weights_label.config(text="Using Trained Weights")

    random_button = tk.Button(
        root,
        text="Use Random Generated Weights",
        command=use_random_weights,
    )
    random_button.pack()

    trained_button = tk.Button(
        root,
        text="Use Trained Weights",
        command=use_trained_weights,
    )
    trained_button.pack()

    remove_weights_button = tk.Button(
        root, text="Remove Weights", command=remove_weights
    )
    remove_weights_button.pack()

    def update_parameters():
        """Update hyperparameters in the queue."""
        param_queue.put(
            {
                "dt": dt_var.get(),
                "gain": gain_var.get(),
            }
        )
        root.after(1000, update_parameters)  # Update every 1 second

    # Start parameter update loop
    update_parameters()

    # Ensure record_live uses the modified spikes function
    record_thread = threading.Thread(
        target=record_live,
        args=(
            W0,
            W1,
            parameters,
            input_queue,
            output_queue,
            param_queue,
        ),
    )
    record_thread.daemon = True
    record_thread.start()

    start_time = time.time()
    total_spikes = 0
    last_spike_count = 0

    def update_plot():
        nonlocal start_time, total_spikes, image_loaded, last_spike_count, static_input
        try:
            if not output_queue.empty():
                activity, R1, R2, timestamp = output_queue.get()
                elapsed_time = timestamp - start_time

                spikes_this_frame = np.sum(activity)
                total_spikes += spikes_this_frame

                spike_count_label.config(
                    text=f"Spike Count: {spikes_this_frame}"
                )
                total_spike_count_label.config(
                    text=f"Total Spike Count: {total_spikes}"
                )

                spike_rate = (
                    total_spikes / elapsed_time if elapsed_time > 0 else 0
                )
                spike_rate_label.config(
                    text=f"Spike Rate: {spike_rate:.2f} Hz"
                )

                # Update neuron colors in both canvases (Layer 1)
                R1_normalized = (R1 - np.min(R1)) / (np.max(R1) - np.min(R1))  # Normalize R1
                for neuron_idx, rect in enumerate(neuron_rects_1):
                    if neuron_idx < R1.shape[0]:
                        color_value = int(R1_normalized[neuron_idx, 0] * 255)
                        hex_color = "#{:02x}{:02x}{:02x}".format(color_value, 0, 255 - color_value)
                        neuron_canvas_1.itemconfig(rect, fill=hex_color)

                # Update neuron colors in both canvases (Layer 2)
                for neuron_idx, rect in enumerate(neuron_rects_2):
                    if neuron_idx < activity.shape[0]:  # Use activity.shape[0] for layer 2
                        spike_sum = np.sum(activity[neuron_idx, :])  # Check spikes in current time step
                        neuron_canvas_2.itemconfig(
                            rect,
                            fill="red" if spike_sum > 0 else "blue",
                        )

                # Update raster plots for both layers
                ax1.clear()
                ax2.clear()

                # Layer 1 (R1)
                for neuron_idx in range(R1.shape[0]):
                    spike_times_1 = np.where(R1[neuron_idx, :] > 0)[0] * parameters["dt"]
                    if len(spike_times_1) > 0:
                        ax1.plot(
                            spike_times_1 + elapsed_time,
                            np.ones_like(spike_times_1) * neuron_idx,
                            "|k",
                            markersize=3,
                        )

                # Layer 2 (activity)
                for neuron_idx in range(activity.shape[0]):
                    spike_times_2 = np.where(activity[neuron_idx, :] == 1)[0] * parameters["dt"]
                    if len(spike_times_2) > 0:
                        ax2.plot(
                            spike_times_2 + elapsed_time,
                            np.ones_like(spike_times_2) * neuron_idx,
                            "|k",
                            markersize=3,
                        )

                ax1.set_xlabel("Time (s)")
                ax1.set_ylabel("Neuron Index")
                ax1.set_title("Live Spike Raster Plot - Layer 1")
                ax2.set_xlabel("Time (s)")
                ax2.set_ylabel("Neuron Index")
                ax2.set_title("Live Spike Raster Plot - Layer 2")

                raster_plot1.draw()
                raster_plot2.draw()
                last_spike_count = spikes_this_frame

                # Provide input to the record_live thread based on image_loaded flag
                if image_loaded:
                    input_queue.put(static_input)  # Use the loaded image
                else:
                    # Use Poisson noise
                    input_queue.put(np.random.poisson(lam=2, size=(784, 1)))

        except KeyboardInterrupt:
            logging.info("Stopping experiment...")
            input_queue.put(None)  # Signal the thread to stop
            record_thread.join()
            root.destroy()
            return

        finally:
            root.after(100, update_plot)

    root.after(100, update_plot)
    root.mainloop()


def cli_interaction(W0, W1):
    """Handles command-line interaction for the experiment."""

    while True:
        try:
            # Get parameters from the user
            dt = float(input("Enter dt (e.g., 0.005): "))
            gain = float(input("Enter gain (e.g., 1.0): "))
            nrep = int(input("Enter nrep (e.g., 1): "))
            spike_format = input(
                "Enter spike format ('raster', 'histogram', 'rate', 'heatmap'): "
            )

            # Validate input
            if (
                dt <= 0
                or gain <= 0
                or nrep <= 0
                or spike_format
                not in ["raster", "histogram", "rate", "heatmap"]
            ):
                print(
                    "Invalid input. Please provide positive values for dt, gain, and nrep, and a valid spike format."
                )
                continue

            parameters = {"dt": dt, "gain": gain, "nrep": nrep}

            # Load images and labels
            images, labels = load_mnist_images("data")

            # Run the experiment
            run_experiment(images, labels, parameters, [spike_format], W0, W1)

            # Ask if the user wants to run another experiment
            another_experiment = input(
                "Run another experiment? (y/n): "
            ).lower()
            if another_experiment != "y":
                break

        except ValueError:
            print("Invalid input. Please enter valid numbers.")
        except KeyboardInterrupt:
            print("\nExperiment interrupted.")
            break


def main():
    # Analyze the weights before running the experiment
    mat_file_path = "trained_model.npz"  # Or get the path from user input
    analyze_mat_weights(mat_file_path)

    parser = argparse.ArgumentParser(
        description="Run neural decoding experiments."
    )
    parser.add_argument(
        "--dataset",
        choices=["mnist", "cifar10", "other"],
        default="mnist",
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_dir",
        default="data",
        help="Directory to store data",
    )
    parser.add_argument(
        "--spike_format",
        choices=["raster", "histogram", "rate", "heatmap"],
        default="raster",
        help="Spike visualization format",
    )
    parser.add_argument(
        "--params_file",
        default="params.json",
        help="JSON file with parameters",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--live",
        "-l",
        action="store_true",
        help="Run in live mode",
    )
    parser.add_argument(
        "--activate",
        "-a",
        action="store_true",
        help="Activate with custom input",
    )
    args = parser.parse_args()  # Parse arguments here

    if args.interactive:
        cli_interaction(
            default_W0, default_W1
        )  # Use default weights for interactive mode
    else:
        if args.dataset == "mnist":
            download_extract_mnist(args.data_dir)
            images, labels = load_mnist_images(args.data_dir, kind="train")
            images_test, labels_test = load_mnist_images(args.data_dir, kind="test")
        elif args.dataset == "cifar10":
            images, labels = load_cifar10_images(args.data_dir)
        elif args.dataset == "other":
            image_paths = []
            images = load_images(image_paths)
            labels = []

        try:
            with open(args.params_file, "r") as f:
                params_config = json.load(f)
            parameters = params_config["parameters"]
            dts = params_config.get("dts", [0.005, 0.01])
            gains = params_config.get("gains", [0.8, 1.2])
            nreps = params_config.get("nreps", [1, 2])
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error loading parameters: {e}")
            return

        cp = calculate_category_priors(labels)

        if args.live:
            parameters["nrep"] = 5 # Increased nrep
            logging.info("Beginning live decoding experiment:")
            run_experiment_live(
                default_W0, default_W1, parameters, record_live
            )  # Pass default weights
            logging.info("Finished live decoding experiment.")
        elif args.activate:
            logging.info("Activating neural population with custom input:")
            # run_experiment_activate(W0, W1, parameters)
            # This function is not defined, needs implementation
            logging.info("Finished neural population activation.")
        else:
            logging.info("Beginning decoding experiments:")
            for dt in dts:
                for gain in gains:
                    for nrep in nreps:
                        logging.info(
                            f"Running case with dt = {dt:.3f} s, gain = {gain:.1f}, nreps = {nrep}..."
                        )
                        parameters["dt"] = dt
                        parameters["gain"] = gain
                        parameters["nrep"] = nrep

                        activity = record(
                            images, default_W0, default_W1, parameters
                        )  # Use default weights

                        spike_formats = [args.spike_format]  # Use specified format

                        if args.spike_format == "raster":
                            show_spikes(activity, parameters)
                        elif args.spike_format == "histogram":
                            show_spike_histogram(activity, parameters)
                        elif args.spike_format == "rate":
                            show_spike_rate_plot(activity, parameters)
                        elif args.spike_format == "heatmap":
                            show_neuron_heatmap(activity, parameters)

                        ll = loglikelihood(
                            activity,
                            images,
                            default_W0,
                            default_W1,
                            parameters,
                        )  # Use default weights
                        lp = logprior(cp, images.shape[1])
                        pos = posterior(ll, lp)
                        cm, ind = confusion(pos)
                        pa = posaverage(images, pos, 10)

                        # Ensure the 'figures' directory exists
                        os.makedirs("figures", exist_ok=True)
                        filename = (
                            f"figures/case_dt{dt:.3f}_gain{gain:.1f}_nreps{nrep}"
                        )

                        show_cm(cm, vis="off")
                        try:
                            plt.savefig(f"{filename}.png", dpi=300)
                        except Exception as e:
                            logging.warning(f"Error saving figure: {e}")
                        plt.close()

                        show_images(images)
                        show_images(pa)

            logging.info("Finished decoding experiments.")


if __name__ == "__main__":
    main()