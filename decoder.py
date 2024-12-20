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

# Import from helper_functions
from utils.helper_functions import (
    USE_CUPY,
    load_weights,
    load_image,
    record_live,
    verify_dimensions,
    safe_matrix_operation
)

if USE_CUPY:
    import cupy as cp

def run_experiment_live(parameters):
    """Runs a live decoding experiment with GUI visualization."""
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

    # --- Create spike statistics labels ---
    spike_count_label_0 = tk.Label(root, text="Layer 0 Spike Count: 0")
    spike_count_label_0.grid(row=1, column=0, sticky="w")

    total_spike_count_label_0 = tk.Label(root, text="Layer 0 Total Spike Count: 0")
    total_spike_count_label_0.grid(row=2, column=0, sticky="w")

    spike_rate_label_0 = tk.Label(root, text="Layer 0 Spike Rate: 0 Hz")
    spike_rate_label_0.grid(row=3, column=0, sticky="w")

    spike_count_label_1 = tk.Label(root, text="Layer 1 Spike Count: 0")
    spike_count_label_1.grid(row=1, column=3, sticky="e")

    total_spike_count_label_1 = tk.Label(root, text="Layer 1 Total Spike Count: 0")
    total_spike_count_label_1.grid(row=2, column=3, sticky="e")

    spike_rate_label_1 = tk.Label(root, text="Layer 1 Spike Rate: 0 Hz")
    spike_rate_label_1.grid(row=3, column=3, sticky="e")

    # --- Create raster plots ---
    fig0, ax0 = plt.subplots()
    raster_plot0 = FigureCanvasTkAgg(fig0, master=root)
    raster_plot0.get_tk_widget().grid(row=4, column=0, columnspan=2, padx=5, pady=5)
    ax0.set_title("Live Spike Raster Plot - Layer 0")

    fig1, ax1 = plt.subplots()
    raster_plot1 = FigureCanvasTkAgg(fig1, master=root)
    raster_plot1.get_tk_widget().grid(row=4, column=2, columnspan=2, padx=5, pady=5)
    ax1.set_title("Live Spike Raster Plot - Layer 1")

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

    # --- GUI Layout ---
    button_frame = tk.Frame(root)
    button_frame.grid(row=5, column=0, columnspan=4, pady=10)

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

    # Row 3: Default Weights
    add_default_weights_btn = tk.Button(button_frame, text="Default Weights", command=add_default_weights)
    add_default_weights_btn.grid(row=2, column=0, padx=5, pady=5)

    remove_default_weights_btn = tk.Button(button_frame, text="Remove Default Weights", command=remove_default_weights)
    remove_default_weights_btn.grid(row=2, column=1, padx=5, pady=5)

    # --- Create parameter sliders ---
    dt_var = tk.DoubleVar(value=parameters["dt"])
    gain_var = tk.DoubleVar(value=parameters["gain"])

    dt_label = ttk.Label(root, text="Time Step (dt):")
    dt_label.grid(row=6, column=0, sticky="w", pady=2)
    dt_scale = ttk.Scale(root, from_=0.001, to=0.01, variable=dt_var, orient="horizontal")
    dt_scale.grid(row=7, column=0, padx=5, pady=2, sticky="ew")

    gain_label = ttk.Label(root, text="Gain:")
    gain_label.grid(row=6, column=3, sticky="w", pady=2)
    gain_scale = ttk.Scale(root, from_=0.1, to=5.0, variable=gain_var, orient="horizontal")
    gain_scale.grid(row=7, column=3, padx=5, pady=2, sticky="ew")

    def update_parameters():
        param_queue.put({"dt": dt_var.get(), "gain": gain_var.get()})
        root.after(1000, update_parameters)

    # Start parameter updates
    update_parameters()

    # Start recording thread
    record_thread = threading.Thread(
        target=record_live,
        args=(W0, W1, parameters, input_queue, output_queue, param_queue),
        daemon=True
    )
    record_thread.start()

    # Initialize timing variables
    start_time = time.time()
    total_spikes_0 = 0
    total_spikes_1 = 0

    def update_plot():
        nonlocal start_time, total_spikes_0, total_spikes_1, image_loaded, static_input
        if not output_queue.empty():
            activity, R1, S1, R2, timestamp = output_queue.get()
            elapsed_time = timestamp - start_time

            # Update spike statistics for Layer 0
            spikes_this_frame_0 = np.sum(S1)
            total_spikes_0 += spikes_this_frame_0
            spike_count_label_0.config(text=f"Layer 0 Spike Count: {spikes_this_frame_0}")
            total_spike_count_label_0.config(text=f"Layer 0 Total Spike Count: {total_spikes_0}")
            spike_rate_0 = total_spikes_0 / elapsed_time if elapsed_time > 0 else 0
            spike_rate_label_0.config(text=f"Layer 0 Spike Rate: {spike_rate_0:.2f} Hz")

            # Update spike statistics for Layer 1
            spikes_this_frame_1 = np.sum(activity)  # activity corresponds to S2 (Layer 1)
            total_spikes_1 += spikes_this_frame_1
            spike_count_label_1.config(text=f"Layer 1 Spike Count: {spikes_this_frame_1}")
            total_spike_count_label_1.config(text=f"Layer 1 Total Spike Count: {total_spikes_1}")
            spike_rate_1 = total_spikes_1 / elapsed_time if elapsed_time > 0 else 0
            spike_rate_label_1.config(text=f"Layer 1 Spike Rate: {spike_rate_1:.2f} Hz")

            # Update neuron visualizations for layer 0
            R1_normalized = (R1 - np.min(R1)) / (np.max(R1) - np.min(R1))
            for i, rect in enumerate(neuron_rects_0):
                if i < R1.shape[0]:
                    color_value = int(R1_normalized[i, 0] * 255)
                    hex_color = "#{:02x}{:02x}{:02x}".format(color_value, 0, 255 - color_value)
                    neuron_canvas_0.itemconfig(rect, fill=hex_color)

            # Update neuron visualizations for layer 1
            for i, rect in enumerate(neuron_rects_1):
                if i < activity.shape[0]:
                    neuron_canvas_1.itemconfig(
                        rect,
                        fill="red" if np.sum(activity[i, :]) > 0 else "blue"
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
            for i in range(activity.shape[0]):
                spike_times = np.where(activity[i, :] == 1)[0] * parameters["dt"]
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

            # Determine the next input based on button presses or default behavior
            if image_loaded:
                input_queue.put(static_input)  # Use the current static input (default or loaded)
            elif not weights_loaded:
                if USE_CUPY:
                    input_queue.put(cp.random.poisson(lam=2, size=(784, 1)))
                else:
                    input_queue.put(np.random.poisson(lam=2, size=(784, 1)))  # No image, no weights, use lam=2

            else:
                if USE_CUPY:
                    input_queue.put(cp.random.poisson(lam=0.5, size=(784, 1)))
                else:
                    input_queue.put(np.random.poisson(lam=0.5, size=(784, 1)))  # No image, but weights, use lam=0.5

        root.after(100, update_plot)

    # Start the update loop
    root.after(100, update_plot)
    root.mainloop()

if __name__ == "__main__":
    # Default parameters
    parameters = {
        "dt": 0.005,
        "gain": 1.0,
        "nrep": 5
    }

    # Run the live experiment
    run_experiment_live(parameters)