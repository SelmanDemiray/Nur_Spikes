import numpy as np
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from scipy.stats import poisson

# Import from helper_functions
from utils.helper_functions import (
    USE_CUPY,
    load_weights,
    load_image,
    record_live,
    verify_dimensions,
    safe_matrix_operation,
    save_weights,
    save_image,
    AdaptiveLearning,
    HomeostaticPlasticity,
    BatchProcessor,
    PerformanceMonitor,
    export_network_visualization,
    optimize_weights,
    auto_tune_parameters,
    safe_normalize,
    advanced_metrics,
    debug_print
)

if USE_CUPY:
    import cupy as cp

def run_experiment_live(parameters):
    """Runs a live decoding experiment with GUI visualization."""
    # Add new parameters
    parameters.update({
        "batch_size": 32,
        "target_rate": 10,
        "adaptive_learning": True,
        "use_homeostatic": True,
        "max_dynamic_layers": 10  # Ensure max_dynamic_layers is included
    })
    
    # Initialize adaptive learning and homeostatic plasticity
    adaptive_learning = AdaptiveLearning()
    homeostatic = HomeostaticPlasticity(target_rate=parameters["target_rate"])
    batch_processor = BatchProcessor(batch_size=parameters["batch_size"])
    
    # Initialize with default weights and image
    W0 = np.random.normal(0, 1, size=(500, 784))
    W1 = np.random.uniform(-1, 1, size=(10, 500))
    if USE_CUPY:
        W0 = cp.asarray(W0)
        W1 = cp.asarray(W1)
    print("Initial shape of W0:", W0.shape)

    # Set up queues
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    param_queue = queue.Queue()
    prediction_history_queue = queue.Queue()

    if USE_CUPY:
        static_input = cp.random.poisson(lam=0.5, size=(784, 1))
    else:
        static_input = np.random.poisson(lam=0.5, size=(784, 1))

    weights_loaded = True
    image_loaded = True

    # Put the default image into the input queue
    input_queue.put(static_input)

    # Create main window
    root = tk.Tk()
    root.title("Live SNN Experiment")

    # Add library indicator
    library_label = tk.Label(root, text=f"Using {'CuPy' if USE_CUPY else 'NumPy'}")
    library_label.grid(row=0, column=0, columnspan=4)

    # --- Create neuron visualization windows ---
    neuron_window_0 = tk.Toplevel(root)
    neuron_window_0.title("Neuron Activity - Layer 0")
    neuron_window_1 = tk.Toplevel(root)
    neuron_window_1.title("Neuron Activity - Layer 1")

    # --- Set up visualization canvases ---
    num_neurons_0 = W0.shape[0]
    grid_size_0 = int(np.ceil(np.sqrt(num_neurons_0)))
    canvas_width_0 = grid_size_0 * 50
    canvas_height_0 = grid_size_0 * 50

    num_neurons_1 = W1.shape[0]
    grid_size_1 = int(np.ceil(np.sqrt(num_neurons_1)))
    canvas_width_1 = grid_size_1 * 50
    canvas_height_1 = grid_size_1 * 50

    neuron_canvas_0 = tk.Canvas(neuron_window_0, width=canvas_width_0, height=canvas_height_0)
    neuron_canvas_0.pack()
    neuron_canvas_1 = tk.Canvas(neuron_window_1, width=canvas_width_1, height=canvas_height_1)
    neuron_canvas_1.pack()

    # --- Create neuron rectangles for layer 0 ---
    neuron_rects_0 = []
    for i in range(num_neurons_0):
        row = i // grid_size_0
        col = i % grid_size_0
        rect = neuron_canvas_0.create_rectangle(
            col * 50, row * 50,
            col * 50 + 40, row * 50 + 40,
            fill="blue"
        )
        neuron_rects_0.append(rect)

    # --- Create neuron rectangles for layer 1 ---
    neuron_rects_1 = []
    for i in range(num_neurons_1):
        row = i // grid_size_1
        col = i % grid_size_1
        rect = neuron_canvas_1.create_rectangle(
            col * 50, row * 50,
            col * 50 + 40, row * 50 + 40,
            fill="blue"
        )
        neuron_rects_1.append(rect)

    # --- Create spike statistics labels and rate change graphs---
    spike_count_label_0 = tk.Label(root, text="Layer 0 Spike Count: 0")
    spike_count_label_0.grid(row=3, column=0, sticky="w")

    total_spike_count_label_0 = tk.Label(root, text="Layer 0 Total Spike Count: 0")
    total_spike_count_label_0.grid(row=4, column=0, sticky="w")

    spike_rate_label_0 = tk.Label(root, text="Layer 0 Spike Rate: 0 Hz")
    spike_rate_label_0.grid(row=5, column=0, sticky="w")

    spike_count_label_1 = tk.Label(root, text="Layer 1 Spike Count: 0")
    spike_count_label_1.grid(row=3, column=3, sticky="e")

    total_spike_count_label_1 = tk.Label(root, text="Layer 1 Total Spike Count: 0")
    total_spike_count_label_1.grid(row=4, column=3, sticky="e")

    spike_rate_label_1 = tk.Label(root, text="Layer 1 Spike Rate: 0 Hz")
    spike_rate_label_1.grid(row=5, column=3, sticky="e")

    # --- Create prediction label ---
    prediction_label = tk.Label(root, text="Prediction: None (Confidence: 0.00%)")
    prediction_label.grid(row=2, column=0, columnspan=4)

    # --- Create figures and canvases for the rate change graphs ---
    fig_rate_0, ax_rate_0 = plt.subplots(figsize=(4, 2)) 
    canvas_rate_0 = FigureCanvasTkAgg(fig_rate_0, master=root)
    canvas_rate_0.get_tk_widget().grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
    ax_rate_0.set_title("Layer 0 Rate of Change")
    ax_rate_0.set_xlabel("Time (s)")
    ax_rate_0.set_ylabel("Rate of Change (Hz/s)")

    fig_rate_1, ax_rate_1 = plt.subplots(figsize=(4, 2)) 
    canvas_rate_1 = FigureCanvasTkAgg(fig_rate_1, master=root)
    canvas_rate_1.get_tk_widget().grid(row=1, column=2, columnspan=2, padx=5, pady=5, sticky="ew")
    ax_rate_1.set_title("Layer 1 Rate of Change")
    ax_rate_1.set_xlabel("Time (s)")
    ax_rate_1.set_ylabel("Rate of Change (Hz/s)")

    # --- Create raster plots ---
    fig0, ax0 = plt.subplots()
    raster_plot0 = FigureCanvasTkAgg(fig0, master=root)
    raster_plot0.get_tk_widget().grid(row=6, column=0, columnspan=2, padx=5, pady=5)
    ax0.set_title("Live Spike Raster Plot - Layer 0")

    fig1, ax1 = plt.subplots()
    raster_plot1 = FigureCanvasTkAgg(fig1, master=root)
    raster_plot1.get_tk_widget().grid(row=6, column=2, columnspan=2, padx=5, pady=5)
    ax1.set_title("Live Spike Raster Plot - Layer 1")

    # --- Create input and weight visualization frames ---
    input_frame = tk.Frame(root)
    input_frame.grid(row=10, column=0, columnspan=2, pady=5)

    weight_frame = tk.Frame(root)
    weight_frame.grid(row=10, column=2, columnspan=2, pady=5)

    # --- Create labels to display information about input and weight ---
    input_label = tk.Label(input_frame, text="Current Input:")
    input_label.pack()

    weight_label = tk.Label(weight_frame, text="Current Weight (W0):")
    weight_label.pack()

    # --- Create canvases to display input image and weight matrix ---
    input_canvas = tk.Canvas(input_frame, width=140, height=140) # Increased canvas size
    input_canvas.pack()

    weight_canvas = tk.Canvas(weight_frame, width=280, height=280)
    weight_canvas.pack()

    # --- Button Functions ---
    def load_weights_button():
        nonlocal W0, W1, weights_loaded, neuron_canvas_0, neuron_canvas_1, neuron_rects_0, neuron_rects_1, num_neurons_0, num_neurons_1, grid_size_0, grid_size_1, canvas_width_0, canvas_height_0, canvas_width_1, canvas_height_1

        filepath = filedialog.askopenfilename(filetypes=[("NumPy files", "*.npz")])
        if filepath:
            try:
                W0, W1 = load_weights(filepath)
                if USE_CUPY:
                    W0 = cp.asarray(W0)
                    W1 = cp.asarray(W1)
                print("Shape of W0 after loading:", W0.shape)
                weights_loaded = True

                # Update the size of Layer 0 visualization based on new weights
                num_neurons_0 = W0.shape[0]
                grid_size_0 = int(np.ceil(np.sqrt(num_neurons_0)))
                canvas_width_0 = grid_size_0 * 50
                canvas_height_0 = grid_size_0 * 50

                # Clear and update neuron canvas and rectangles for Layer 0
                neuron_canvas_0.delete("all")
                neuron_canvas_0.config(width=canvas_width_0, height=canvas_height_0)
                neuron_rects_0.clear()
                for i in range(num_neurons_0):
                    row = i // grid_size_0
                    col = i % grid_size_0
                    rect = neuron_canvas_0.create_rectangle(
                        col * 50, row * 50,
                        col * 50 + 40, row * 50 + 40,
                        fill="blue"
                    )
                    neuron_rects_0.append(rect)

                # Update the size of Layer 1 visualization based on new weights
                num_neurons_1 = W1.shape[0]
                grid_size_1 = int(np.ceil(np.sqrt(num_neurons_1)))
                canvas_width_1 = grid_size_1 * 50
                canvas_height_1 = grid_size_1 * 50

                # Clear and update neuron canvas and rectangles for Layer 1
                neuron_canvas_1.delete("all")
                neuron_canvas_1.config(width=canvas_width_1, height=canvas_height_1)
                neuron_rects_1.clear()
                for i in range(num_neurons_1):
                    row = i // grid_size_1
                    col = i % grid_size_1
                    rect = neuron_canvas_1.create_rectangle(
                        col * 50, row * 50,
                        col * 50 + 40, row * 50 + 40,
                        fill="blue"
                    )
                    neuron_rects_1.append(rect)
            except Exception as e:
                print(f"Error loading weights: {e}")

    def load_image_button():
        nonlocal static_input, image_loaded
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.png *.jpg *.jpeg")])
        if filepath:
            try:
                static_input = load_image(filepath)
                if USE_CUPY:
                    static_input = cp.asarray(static_input)
                image_loaded = True
                input_queue.put(static_input)
            except Exception as e:
                print(f"Error loading image: {e}")

    def add_default_image():
        nonlocal static_input, image_loaded
        if USE_CUPY:
            static_input = cp.random.poisson(lam=0.5, size=(784, 1))
        else:
            static_input = np.random.poisson(lam=0.5, size=(784, 1))
        image_loaded = True
        input_queue.put(static_input)

    def remove_default_image():
        nonlocal image_loaded
        image_loaded = False

    def add_default_weights():
        nonlocal W0, W1, weights_loaded, neuron_canvas_0, neuron_canvas_1, neuron_rects_0, neuron_rects_1, num_neurons_0, num_neurons_1, grid_size_0, grid_size_1, canvas_width_0, canvas_height_0, canvas_width_1, canvas_height_1
        W0 = np.random.normal(0, 1, size=(500, 784))
        W1 = np.random.uniform(-1, 1, size=(10, 500))
        if USE_CUPY:
            W0 = cp.asarray(W0)
            W1 = cp.asarray(W1)
        weights_loaded = True

        # Update the size of Layer 0 visualization
        num_neurons_0 = W0.shape[0]
        grid_size_0 = int(np.ceil(np.sqrt(num_neurons_0)))
        canvas_width_0 = grid_size_0 * 50
        canvas_height_0 = grid_size_0 * 50

        # Update neuron canvas and rectangles for Layer 0
        neuron_canvas_0.delete("all")
        neuron_canvas_0.config(width=canvas_width_0, height=canvas_height_0)
        neuron_rects_0.clear()
        for i in range(num_neurons_0):
            row = i // grid_size_0
            col = i % grid_size_0
            rect = neuron_canvas_0.create_rectangle(
                col * 50, row * 50,
                col * 50 + 40, row * 50 + 40,
                fill="blue"
            )
            neuron_rects_0.append(rect)

        # Update the size of Layer 1 visualization
        num_neurons_1 = W1.shape[0]
        grid_size_1 = int(np.ceil(np.sqrt(num_neurons_1)))
        canvas_width_1 = grid_size_1 * 50
        canvas_height_1 = grid_size_1 * 50

        # Update neuron canvas and rectangles for Layer 1
        neuron_canvas_1.delete("all")
        neuron_canvas_1.config(width=canvas_width_1, height=canvas_height_1)
        neuron_rects_1.clear()
        for i in range(num_neurons_1):
            row = i // grid_size_1
            col = i % grid_size_1
            rect = neuron_canvas_1.create_rectangle(
                col * 50, row * 50,
                col * 50 + 40, row * 50 + 40,
                fill="blue"
            )
            neuron_rects_1.append(rect)

    def remove_default_weights():
        nonlocal weights_loaded
        weights_loaded = False

    def save_weights_button():
        nonlocal W0, W1
        filepath = filedialog.asksaveasfilename(defaultextension=".npz", filetypes=[("NumPy files", "*.npz")])
        if filepath:
            try:
                save_weights(filepath, W0, W1)
                print(f"Weights saved to {filepath}")
            except Exception as e:
                print(f"Error saving weights: {e}")

    def save_image_button():
        nonlocal static_input
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
        if filepath:
            try:
                save_image(filepath, static_input)
                print(f"Image saved to {filepath}")
            except Exception as e:
                print(f"Error saving image: {e}")

    # --- GUI Layout ---
    button_frame = tk.Frame(root)
    button_frame.grid(row=9, column=0, columnspan=4, pady=10)

    # Row 1: Custom Weights and Image
    load_weights_btn = tk.Button(button_frame, text="Custom Weights", command=load_weights_button)
    load_weights_btn.grid(row=0, column=0, padx=5, pady=5)

    load_image_btn = tk.Button(button_frame, text="Custom Image", command=load_image_button)
    load_image_btn.grid(row=0, column=1, padx=5, pady=5)

    # Row 2: Default Image
    add_default_image_btn = tk.Button(button_frame, text="Default Image", command=add_default_image)
    add_default_image_btn.grid(row=1, column=0, padx=5, pady=5)

    remove_default_image_btn = tk.Button(button_frame, text="Remove Default Image", command=remove_default_image)
    remove_default_image_btn.grid(row=1, column=1, padx=5, pady=5)

    # Row 2: Save Image
    save_image_btn = tk.Button(button_frame, text="Save Image", command=save_image_button)
    save_image_btn.grid(row=1, column=2, padx=5, pady=5)

    # Row 3: Default Weights
    add_default_weights_btn = tk.Button(button_frame, text="Default Weights", command=add_default_weights)
    add_default_weights_btn.grid(row=2, column=0, padx=5, pady=5)

    remove_default_weights_btn = tk.Button(button_frame, text="Remove Default Weights", command=remove_default_weights)
    remove_default_weights_btn.grid(row=2, column=1, padx=5, pady=5)

    # Row 3: Save Weights
    save_weights_btn = tk.Button(button_frame, text="Save Weights", command=save_weights_button)
    save_weights_btn.grid(row=2, column=2, padx=5, pady=5)

    # --- Create parameter sliders ---
    dt_var = tk.DoubleVar(value=parameters["dt"])
    gain_var = tk.DoubleVar(value=parameters["gain"])

    dt_label = ttk.Label(root, text="Time Step (dt):")
    dt_label.grid(row=7, column=0, sticky="w", pady=2)
    dt_scale = ttk.Scale(root, from_=0.001, to=0.01, variable=dt_var, orient="horizontal")
    dt_scale.grid(row=8, column=0, padx=5, pady=2, sticky="ew")

    gain_label = ttk.Label(root, text="Gain:")
    gain_label.grid(row=7, column=3, sticky="w", pady=2)
    gain_scale = ttk.Scale(root, from_=0.1, to=5.0, variable=gain_var, orient="horizontal")
    gain_scale.grid(row=8, column=3, padx=5, pady=2, sticky="ew")

    def update_parameters():
        param_queue.put({"dt": dt_var.get(), "gain": gain_var.get()})
        root.after(1000, update_parameters)

    # Start parameter updates
    update_parameters()

    # Start recording thread
    record_thread = threading.Thread(
        target=record_live,
        args=(W0, W1, parameters, input_queue, output_queue, param_queue, prediction_history_queue),
        daemon=True
    )
    record_thread.start()

    # --- Prediction History Table ---
    prediction_table_frame = tk.Frame(root)
    prediction_table_frame.grid(row=11, column=0, columnspan=4, pady=5)

    prediction_table_label = tk.Label(prediction_table_frame, text="Prediction History:")
    prediction_table_label.pack()

    prediction_table = ttk.Treeview(prediction_table_frame, columns=("Label", "Count"), show="headings")
    prediction_table.heading("Label", text="Label")
    prediction_table.heading("Count", text="Count")
    prediction_table.pack()

    # --- Regular ANN Prediction ---
    ann_prediction_frame = tk.Frame(root)
    ann_prediction_frame.grid(row=12, column=0, columnspan=4, pady=5)

    ann_prediction_label = tk.Label(ann_prediction_frame, text="Regular ANN Prediction: None")
    ann_prediction_label.pack()

    def regular_ann_forward_pass(W0_np, W1_np, input_np, interval=3):
        """Performs a regular ANN forward pass periodically."""
        while True:
            # Convert to NumPy arrays for regular ANN
            if USE_CUPY:
                W0_np_ann = cp.asnumpy(W0_np)
                W1_np_ann = cp.asnumpy(W1_np)
                input_np_ann = cp.asnumpy(input_np)
            else:
                W0_np_ann = W0_np
                W1_np_ann = W1_np
                input_np_ann = input_np

            # Flatten the input image
            input_np_ann = input_np_ann.flatten()

            # Perform forward pass
            layer1_output = np.dot(W0_np_ann, input_np_ann)
            layer1_output = np.maximum(0, layer1_output)  # ReLU activation

            layer2_output = np.dot(W1_np_ann, layer1_output)
            prediction = np.argmax(layer2_output)

            # Update the label in the main thread
            root.after(0, ann_prediction_label.config, {"text": f"Regular ANN Prediction: {prediction}"})

            time.sleep(interval)

    # Thread for regular ANN prediction
    ann_thread = threading.Thread(target=regular_ann_forward_pass, args=(W0, W1, static_input), daemon=True)
    ann_thread.start()

    # Initialize timing variables and rate change lists
    start_time = time.time()
    total_spikes_0 = 0
    total_spikes_1 = 0
    rate_changes_0 = [] 
    rate_changes_1 = []
    time_points_0 = []
    time_points_1 = []

    # Function to calculate rate of change
    def calculate_rate_of_change(rate_changes, time_points):
        if len(rate_changes) < 2:
            return 0
        # Calculate the change in rate over the change in time
        delta_rate = rate_changes[-1] - rate_changes[-2]
        delta_time = time_points[-1] - time_points[-2]
        if delta_time == 0:
            return 0
        return delta_rate / delta_time

    # Function to get prediction from output
    def get_prediction(output_spikes):
        # Count the number of spikes for each class
        spike_counts = np.sum(output_spikes, axis=1)  

        # Find the class with the maximum spike count
        predicted_class = np.argmax(spike_counts)

        # Calculate confidence (percentage of spikes for the predicted class)
        total_spikes = np.sum(spike_counts)
        confidence = (spike_counts[predicted_class] / total_spikes) * 100 if total_spikes > 0 else 0

        return predicted_class, confidence

    def update_prediction_table():
        """Updates the prediction table with the current prediction history."""
        # Clear existing entries
        for row in prediction_table.get_children():
            prediction_table.delete(row)

        # Update with new entries
        try:
            prediction_counts = prediction_history_queue.get_nowait()
            for label, count in prediction_counts.items():
                prediction_table.insert("", tk.END, values=(label, count))
        except queue.Empty:
            pass

        # Schedule the next update
        root.after(500, update_prediction_table)  # Update every 500 ms

    dynamic_layers = []

    def add_new_layer(W_prev, layer_index):
        """Add a new layer based on previous layer's weights."""
        new_layer_size = max(1, W_prev.shape[0] // 2)  # Ensure non-zero size
        W_new = np.random.uniform(-1, 1, size=(new_layer_size, W_prev.shape[0]))
        if USE_CUPY:
            W_new = cp.asarray(W_new)
        dynamic_layers.append(W_new)
        return W_new

    def update_dynamic_layers(r2=None):
        """Update dynamic layers based on spike activity."""
        if len(dynamic_layers) == 0 or r2 is None:
            return

        MAX_DYNAMIC_LAYERS = parameters["max_dynamic_layers"]  # Use parameters dictionary

        for i, W in enumerate(dynamic_layers):
            if i == 0:
                prev_layer_output = static_input
                if prev_layer_output.shape[0] != W.shape[1]:
                    prev_layer_output = prev_layer_output[:W.shape[1], :]
            else:
                prev_layer_output = dynamic_layers[i-1]

            # Ensure prev_layer_output is a CuPy array if USE_CUPY is True
            if USE_CUPY and not isinstance(prev_layer_output, cp.ndarray):
                prev_layer_output = cp.asarray(prev_layer_output)

            # Debug prints to check dimensions
            debug_print(f"Layer {i + 2}: W.shape = {W.shape}, prev_layer_output.shape = {prev_layer_output.shape}")

            # Add shape verification
            if W.shape[1] != prev_layer_output.shape[0]:
                debug_print(f"Shape mismatch: W has {W.shape[1]} columns but prev_layer_output has {prev_layer_output.shape[0]} rows.")
                try:
                    if prev_layer_output.size == W.shape[1]:
                        prev_layer_output = prev_layer_output.flatten()
                    else:
                        debug_print("Shape mismatch persists. Skipping layer.")
                        continue
                except Exception as e:
                    debug_print(f"Failed to reshape prev_layer_output: {e}")
                    continue  # Skip updating this layer

            # Calculate new layer activity
            if USE_CUPY:
                R_new = cp.dot(W, prev_layer_output)
            else:
                R_new = np.dot(W, prev_layer_output)

            # Generate spikes from R_new using Poisson spiking
            if USE_CUPY:
                spikes = cp.random.poisson(lam=R_new)
                spike_rate = max(cp.mean(spikes), 0)
            else:
                spikes = poisson.rvs(mu=R_new)
                spike_rate = max(np.mean(spikes), 0)
            debug_print(f"Layer {i + 2} Spike Rate: {spike_rate}")

            spike_rate = float(spike_rate) if not np.isnan(spike_rate) else 0.0

            if spike_rate > parameters["spike_threshold"]:
                if len(dynamic_layers) < MAX_DYNAMIC_LAYERS:
                    add_new_layer(W, i + 2)
                    debug_print(f"Adding new layer {i + 3} as spike rate {spike_rate} exceeds threshold.")
                else:
                    debug_print("Maximum number of dynamic layers reached. No new layers will be added.")

    def visualize_dynamic_layers():
        """Visualize dynamic layers in the GUI."""
        for i, W in enumerate(dynamic_layers):
            layer_window = tk.Toplevel(root)
            layer_window.title(f"Neuron Activity - Layer {i + 2}")
            num_neurons = W.shape[0]
            grid_size = int(np.ceil(np.sqrt(num_neurons)))
            canvas_width = grid_size * 50
            canvas_height = grid_size * 50
            layer_canvas = tk.Canvas(layer_window, width=canvas_width, height=canvas_height)
            layer_canvas.pack()
            neuron_rects = []
            for j in range(num_neurons):
                row = j // grid_size
                col = j % grid_size
                rect = layer_canvas.create_rectangle(
                    col * 50, row * 50,
                    col * 50 + 40, row * 50 + 40,
                    fill="blue"
                )
                neuron_rects.append(rect)

    # Initialize the first dynamic layer immediately
    add_new_layer(W1, 2)

    def calculate_spike_rate(spikes, elapsed_time):
        if elapsed_time <= 0:
            return 0
        return float(np.sum(spikes)) / elapsed_time

    def update_plot():
        nonlocal W0, W1
        nonlocal start_time, total_spikes_0, total_spikes_1, image_loaded, static_input, rate_changes_0, rate_changes_1
        
        # Initialize spike rates
        spike_rate_0 = 0
        spike_rate_1 = 0
        elapsed_time = 0
        
        try:
            try:
                data = output_queue.get()
                if len(data) == 5:
                    S2, R1, S1, R2, timestamp = data
                elif len(data) == 4:
                    S2, R1, S1, timestamp = data
                    R2 = None
                else:
                    raise ValueError("Unexpected number of items in output_queue")
            except ValueError as ve:
                print(f"Error unpacking output_queue: {ve}")
                S2, R1, S1, R2, timestamp = None, None, None, None, None

            elapsed_time = timestamp - start_time
            if elapsed_time <= 0:
                elapsed_time = 1e-6  # Prevent division by zero
            
            # Update performance monitor
            computation_time = time.time() - timestamp
            perf_monitor.update(S2, None, computation_time)
            
            # Calculate spike rates using helper function
            spike_rate_0 = calculate_spike_rate(S1, elapsed_time)
            spike_rate_1 = calculate_spike_rate(S2, elapsed_time)
            
            total_spikes_0 += np.sum(S1)
            total_spikes_1 += np.sum(S2)
            
            # Update spike statistics for Layer 0
            spike_count_label_0.config(text=f"Layer 0 Spike Count: {np.sum(S1)}")
            total_spike_count_label_0.config(text=f"Layer 0 Total Spike Count: {total_spikes_0}")
            spike_rate_label_0.config(text=f"Layer 0 Spike Rate: {spike_rate_0:.2f} Hz")

            # Update spike statistics for Layer 1
            spike_count_label_1.config(text=f"Layer 1 Spike Count: {np.sum(S2)}")
            total_spike_count_label_1.config(text=f"Layer 1 Total Spike Count: {total_spikes_1}")
            spike_rate_label_1.config(text=f"Layer 1 Spike Rate: {spike_rate_1:.2f} Hz")

            # Update Layer 2 spike rate in performance monitor
            perf_monitor.layer_2_spike_rate = spike_rate_1
            
            # Update rate change lists for both layers
            if elapsed_time > 0:
                rate_changes_0.append(spike_rate_0)
                time_points_0.append(elapsed_time)
                rate_changes_1.append(spike_rate_1)
                time_points_1.append(elapsed_time)

            # Calculate rate of change for both layers
            roc_0 = calculate_rate_of_change(rate_changes_0, time_points_0)
            roc_1 = calculate_rate_of_change(rate_changes_1, time_points_1)

            # Update the rate of change graphs
            ax_rate_0.clear()
            ax_rate_0.plot(time_points_0, rate_changes_0, color='blue')
            ax_rate_0.set_title(f"Layer 0 Rate of Change: {roc_0:.2f} Hz/s")
            ax_rate_0.set_xlabel("Time (s)")
            ax_rate_0.set_ylabel("Rate (Hz)")

            ax_rate_1.clear()
            ax_rate_1.plot(time_points_1, rate_changes_1, color='red')
            ax_rate_1.set_title(f"Layer 1 Rate of Change: {roc_1:.2f} Hz/s")
            ax_rate_1.set_xlabel("Time (s)")
            ax_rate_1.set_ylabel("Rate (Hz)")
            
            canvas_rate_0.draw()
            canvas_rate_1.draw()

            # Update neuron visualizations for layer 0
            if R1 is not None and R1.size > 0:
                R1_normalized = safe_normalize(R1)
                for i, rect in enumerate(neuron_rects_0):
                    if i < R1.shape[0]:
                        color_value = int(R1_normalized[i, 0] * 255)
                        hex_color = "#{:02x}{:02x}{:02x}".format(color_value, 0, 255 - color_value)
                        neuron_canvas_0.itemconfig(rect, fill=hex_color)

            # Update neuron visualizations for layer 1
            for i, rect in enumerate(neuron_rects_1):
                if i < S2.shape[0]:
                    neuron_canvas_1.itemconfig(
                        rect,
                        fill="red" if np.sum(S2[i, :]) > 0 else "blue"
                    )

            # Update raster plots
            ax0.clear()
            ax1.clear()

            # Plot Layer 0 activity
            for i in range(S1.shape[0]):
                spike_times = np.where(S1[i, :] > 0)[0] * parameters["dt"]
                if len(spike_times) > 0:
                    ax0.plot(
                        spike_times + elapsed_time,
                        np.ones_like(spike_times) * i,
                        "|k",
                        markersize=3
                    )

            # Plot Layer 1 activity
            for i in range(S2.shape[0]):
                spike_times = np.where(S2[i, :] == 1)[0] * parameters["dt"]
                if len(spike_times) > 0:
                    ax1.plot(
                        spike_times + elapsed_time,
                        np.ones_like(spike_times) * i,
                        "|k",
                        markersize=3
                    )

            ax0.set_xlabel("Time (s)")
            ax0.set_ylabel("Neuron Index")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Neuron Index")

            raster_plot0.draw()
            raster_plot1.draw()
            
            # Get prediction and confidence from output spikes (S2)
            predicted_class, confidence = get_prediction(S2)

            # Update prediction label if confidence is 40% or more
            if confidence >= 40:
                prediction_label.config(text=f"Prediction: {predicted_class} (Confidence: {confidence:.2f}%)")
            else:
                prediction_label.config(text="Prediction: Uncertain (Confidence < 40%)")

            # Update input and weight visualizations
            if image_loaded and static_input is not None:
                # Reshape the input for display
                if USE_CUPY:
                    input_image = cp.asnumpy(static_input).reshape(28, 28) * 255
                else:
                    input_image = static_input.reshape(28, 28) * 255

                # Convert to a PIL Image and resize for display
                input_image = Image.fromarray(input_image.astype(np.uint8))
                # Resize the image to fit the canvas (e.g., 5 times the original size)
                resized_image = input_image.resize((140, 140), Image.NEAREST)
                # Convert the resized image to a PhotoImage and display it on the canvas
                input_photo = ImageTk.PhotoImage(resized_image)
                input_canvas.create_image(0, 0, anchor=tk.NW, image=input_photo)
                input_canvas.image = input_photo 


            # Update weight visualization with proper scaling
            if weights_loaded and W0 is not None:
                try:
                    # Convert to NumPy if needed
                    w0_display = cp.asnumpy(W0) if USE_CUPY else W0.copy()
                    
                    # Normalize weights for visualization
                    w0_display = safe_normalize(w0_display)
                    if w0_display is not None:
                        # Scale to 0-255 range with proper size
                        W0_resized = np.kron(w0_display, np.ones((5, 5))) * 255
                        W0_resized = np.clip(W0_resized, 0, 255).astype(np.uint8)
                        
                        # Convert to PIL Image
                        W0_image = Image.fromarray(W0_resized)
                        W0_photo = ImageTk.PhotoImage(W0_image)
                        
                        # Clear previous image and display new one
                        weight_canvas.delete("all")
                        weight_canvas.create_image(0, 0, anchor=tk.NW, image=W0_photo)
                        weight_canvas.image = W0_photo  # Keep reference
                except Exception as e:
                    print(f"Weight visualization error: {e}")

            # Determine the next input based on button presses or default behavior
            if image_loaded:
                input_queue.put(static_input)  
            elif not weights_loaded:
                if USE_CUPY:
                    input_queue.put(cp.random.poisson(lam=2, size=(784, 1)))
                else:
                    input_queue.put(np.random.poisson(lam=2, size=(784, 1)))  

            else:
                if USE_CUPY:
                    input_queue.put(cp.random.poisson(lam=0.5, size=(784, 1)))
                else:
                    input_queue.put(np.random.poisson(lam=0.5, size=(784, 1))) 
        
            # Apply homeostatic plasticity if enabled
            if parameters["use_homeostatic"] and elapsed_time > 0:
                W0 = homeostatic.adjust_weights(W0, spike_rate_0)
                W1 = homeostatic.adjust_weights(W1, spike_rate_1)
        
            # Update learning rate if adaptive learning is enabled
            if parameters["adaptive_learning"]:
                learning_rate = adaptive_learning.adapt({
                    'avg_spike_rate': spike_rate_1
                })
                parameters['learning_rate'] = learning_rate

            # Update dynamic layers
            if R2 is not None:
                update_dynamic_layers(R2)
            else:
                print("Warning: R2 is None. Skipping update_dynamic_layers.")

        except Exception as e:
            print(f"Error in update_plot: {e}")
            import traceback
            traceback.print_exc()

        finally:
            root.after(50, update_plot)  # Faster update rate

    def update_performance_display():
        try:
            stats = perf_monitor.get_stats()
            if stats['active']:
                spike_rate_label.config(text=f"Avg Spike Rate: {stats['avg_spike_rate']:.2f} Hz")
                compute_time_label.config(text=f"Avg Compute Time: {stats['avg_computation_time']*1000:.1f} ms")
                memory_label.config(text=f"Memory Usage: {stats['avg_memory_usage']:.1f} MB")
            else:
                spike_rate_label.config(text="Avg Spike Rate: No data")
                compute_time_label.config(text="Avg Compute Time: No data")
                memory_label.config(text="Memory Usage: No data")
        except Exception as e:
            print(f"Performance display error: {e}")
        
        root.after(500, update_performance_display)  # Update twice per second

    # Start the update loops
    root.after(0, update_prediction_table)
    root.after(100, update_plot)
    
    # Add performance monitor
    perf_monitor = PerformanceMonitor()
    
    # Add performance display
    perf_frame = tk.Frame(root)
    perf_frame.grid(row=13, column=0, columnspan=4, pady=5)
    
    perf_label = tk.Label(perf_frame, text="Performance Metrics:")
    perf_label.pack()
    
    spike_rate_label = tk.Label(perf_frame, text="Avg Spike Rate: 0 Hz")
    spike_rate_label.pack()
    
    compute_time_label = tk.Label(perf_frame, text="Avg Compute Time: 0 ms")
    compute_time_label.pack()
    
    memory_label = tk.Label(perf_frame, text="Memory Usage: 0 MB")
    memory_label.pack()
    
    def optimize_button():
        nonlocal W0, W1
        try:
            W0, W1 = optimize_weights(W0, W1, target_spikes=10)
            print("Weights optimized")
        except Exception as e:
            print(f"Optimization error: {e}")

    def auto_tune_button():
        nonlocal parameters
        stats = perf_monitor.get_stats()
        parameters = auto_tune_parameters(parameters, stats)
        print("Parameters auto-tuned")
        
    # Add optimization buttons
    optimize_btn = tk.Button(button_frame, text="Optimize Weights", command=optimize_button)
    optimize_btn.grid(row=3, column=0, padx=5, pady=5)
    
    auto_tune_btn = tk.Button(button_frame, text="Auto-tune Params", command=auto_tune_button)
    auto_tune_btn.grid(row=3, column=1, padx=5, pady=5)
    
    def update_performance_display():
        stats = perf_monitor.get_stats()
        spike_rate_label.config(text=f"Avg Spike Rate: {stats['avg_spike_rate']:.2f} Hz")
        compute_time_label.config(text=f"Avg Compute Time: {stats['avg_computation_time']*1000:.1f} ms")
        memory_label.config(text=f"Memory Usage: {stats['avg_memory_usage']:.1f} MB")
        root.after(1000, update_performance_display)
    
    # Start performance updates
    update_performance_display()
    
    def export_visualization():
        filepath = filedialog.asksaveasfilename(
            defaultextension=".html",
            filetypes=[("HTML files", "*.html")]
        )
        if filepath:
            if export_network_visualization(W0, W1, rate_changes_1, filepath):
                print(f"Network visualization exported to {filepath}")
            else:
                print("Failed to export visualization")
    
    # Add export visualization button
    export_viz_btn = tk.Button(button_frame, text="Export Visualization", command=export_visualization)
    export_viz_btn.grid(row=3, column=2, padx=5, pady=5)
    
    # Add debug terminal
    debug_frame = tk.Frame(root)
    debug_frame.grid(row=14, column=0, columnspan=4, pady=5, sticky="nsew")
    
    debug_label = tk.Label(debug_frame, text="Debug Terminal:")
    debug_label.pack()
    
    debug_text = tk.Text(debug_frame, height=10, width=60)
    debug_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    
    scrollbar = tk.Scrollbar(debug_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    debug_text.config(yscrollcommand=scrollbar.set)
    scrollbar.config(command=debug_text.yview)
    
    # Create debug print function
    def debug_print(message):
        debug_text.insert(tk.END, f"{message}\n")
        debug_text.see(tk.END)
    
    # Redirect print statements to debug terminal
    import sys
    class DebugPrinter:
        def write(self, text):
            if text.strip():  # Only process non-empty strings
                root.after(0, debug_print, text)
        def flush(self):
            pass
    
    sys.stdout = DebugPrinter()
    
    # Add clear button for debug terminal
    clear_debug_btn = tk.Button(debug_frame, text="Clear Debug", 
                              command=lambda: debug_text.delete(1.0, tk.END))
    clear_debug_btn.pack(side=tk.BOTTOM)
    
    # Add copy button for debug terminal
    def copy_debug():
        root.clipboard_clear()
        root.clipboard_append(debug_text.get(1.0, tk.END))
        root.update()
        debug_print("Debug text copied to clipboard")
        
    copy_debug_btn = tk.Button(debug_frame, text="Copy Debug", command=copy_debug)
    copy_debug_btn.pack(side=tk.BOTTOM)
    
    if parameters.get("enable_advanced_metrics", False):
        metrics_frame = tk.Frame(root)
        metrics_frame.grid(row=15, column=0, columnspan=4, pady=5)
        metrics_label = tk.Label(metrics_frame, text="Advanced Metrics:")
        metrics_label.pack()
        metrics_value_label = tk.Label(metrics_frame, text="N/A")
        metrics_value_label.pack()

        def update_advanced_metrics():
            if not output_queue.empty():
                S2, _, S1, _, _ = output_queue.get()
                adv = advanced_metrics(S2)
                text_val = f"Average Spikes: {adv['avg_spikes']:.2f}"
                if parameters.get("enable_stddev_metrics", False):
                    text_val += f" | Std Dev: {adv['std_spikes']:.2f}"
                metrics_value_label.config(text=text_val)
            root.after(1000, update_advanced_metrics)

        update_advanced_metrics()
    
    # Add button to visualize dynamic layers
    visualize_layers_btn = tk.Button(button_frame, text="Visualize Layers", command=visualize_dynamic_layers)
    visualize_layers_btn.grid(row=4, column=0, padx=5, pady=5)

    root.mainloop()

if __name__ == "__main__":
    # Default parameters
    parameters = {
        "dt": 0.005,
        "gain": 1.0,
        "nrep": 5,
        "spike_threshold": 0.3  # Threshold for adding new layers
    }

    # Run the live experiment
    run_experiment_live(parameters)