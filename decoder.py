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

def load_weights(file_path):
    """Loads weights from a .npz file."""
    try:
        with np.load(file_path) as data:
            W0 = data["W0"]
            W1 = data["W1"]
        return W0, W1
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise

def rates(images, W0, W1, parameters):
    """Calculate rates for both layers."""
    R1 = np.maximum(0, W0 @ images)
    R2 = np.maximum(0, W1 @ R1)
    return R1, R2

def record_live(W0, W1, parameters, input_queue, output_queue, param_queue):
    """Continuously generates spikes for live decoding with parameter updates."""
    try:
        while True:
            # Get updated parameters
            try:
                updated_params = param_queue.get_nowait()
                parameters["dt"] = updated_params["dt"]
                parameters["gain"] = updated_params["gain"]
            except queue.Empty:
                pass

            try:
                images = input_queue.get(timeout=1)
            except queue.Empty:
                images = np.random.poisson(lam=2, size=(784, 1))

            if images is None:
                break

            # Calculate activities for both layers
            R1, R2 = rates(images, W0, W1, parameters)
            R2_positive = np.maximum(R2, 0)
            
            # Generate spikes using Poisson distribution
            mu = parameters["dt"] * parameters["gain"] * R2_positive
            mu = np.tile(mu, (1, parameters["nrep"]))
            S = poisson.rvs(mu=mu, size=(R2_positive.shape[0], parameters["nrep"]))
            
            output_queue.put((S, R1, R2_positive, time.time()))

    except Exception as e:
        print(f"Error in record_live thread: {e}")

def run_experiment_live(W0, W1, parameters):
    """Runs a live decoding experiment with GUI visualization."""
    static_input = np.random.poisson(lam=0.5, size=(784, 1))
    weights_loaded = False
    image_loaded = False

    # Set up queues
    input_queue = queue.Queue()
    output_queue = queue.Queue()
    param_queue = queue.Queue()

    # Create main window
    root = tk.Tk()
    root.title("Live SNN Experiment")

    # Add library indicator
    library_label = tk.Label(root, text=f"Using {'CuPy' if USE_CUPY else 'NumPy'}")
    library_label.pack()

    # Create neuron visualization windows
    neuron_window_1 = tk.Toplevel(root)
    neuron_window_1.title("Neuron Activity - Layer 1")
    neuron_window_2 = tk.Toplevel(root)
    neuron_window_2.title("Neuron Activity - Layer 2")

    # Set up visualization canvases
    num_neurons_1 = W0.shape[0]
    grid_size_1 = int(np.ceil(np.sqrt(num_neurons_1)))
    canvas_width_1 = grid_size_1 * 50
    canvas_height_1 = grid_size_1 * 50

    num_neurons_2 = W1.shape[0]
    grid_size_2 = int(np.ceil(np.sqrt(num_neurons_2)))
    canvas_width_2 = grid_size_2 * 50
    canvas_height_2 = grid_size_2 * 50

    neuron_canvas_1 = tk.Canvas(neuron_window_1, width=canvas_width_1, height=canvas_height_1)
    neuron_canvas_1.pack()
    neuron_canvas_2 = tk.Canvas(neuron_window_2, width=canvas_width_2, height=canvas_height_2)
    neuron_canvas_2.pack()

    # Create neuron rectangles
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

    neuron_rects_2 = []
    for i in range(num_neurons_2):
        row = i // grid_size_2
        col = i % grid_size_2
        rect = neuron_canvas_2.create_rectangle(
            col * 50, row * 50,
            col * 50 + 40, row * 50 + 40,
            fill="blue"
        )
        neuron_rects_2.append(rect)

    # Create spike statistics labels
    spike_count_label = tk.Label(root, text="Spike Count: 0")
    spike_count_label.pack()
    total_spike_count_label = tk.Label(root, text="Total Spike Count: 0")
    total_spike_count_label.pack()
    spike_rate_label = tk.Label(root, text="Spike Rate: 0 Hz")
    spike_rate_label.pack()

    # Create raster plots
    fig1, ax1 = plt.subplots()
    raster_plot1 = FigureCanvasTkAgg(fig1, master=root)
    raster_plot1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    ax1.set_title("Live Spike Raster Plot - Layer 1")

    fig2, ax2 = plt.subplots()
    raster_plot2 = FigureCanvasTkAgg(fig2, master=root)
    raster_plot2.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=1)
    ax2.set_title("Live Spike Raster Plot - Layer 2")

    # Create parameter sliders
    dt_var = tk.DoubleVar(value=parameters["dt"])
    gain_var = tk.DoubleVar(value=parameters["gain"])

    ttk.Label(root, text="Time Step (dt):").pack()
    ttk.Scale(root, from_=0.001, to=0.01, variable=dt_var, orient="horizontal").pack()

    ttk.Label(root, text="Gain:").pack()
    ttk.Scale(root, from_=0.1, to=5.0, variable=gain_var, orient="horizontal").pack()

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
    total_spikes = 0

    def update_plot():
        nonlocal start_time, total_spikes, image_loaded, static_input
        if not output_queue.empty():
            activity, R1, R2, timestamp = output_queue.get()
            elapsed_time = timestamp - start_time

            # Update spike statistics
            spikes_this_frame = np.sum(activity)
            total_spikes += spikes_this_frame
            
            spike_count_label.config(text=f"Spike Count: {spikes_this_frame}")
            total_spike_count_label.config(text=f"Total Spike Count: {total_spikes}")
            
            spike_rate = total_spikes / elapsed_time if elapsed_time > 0 else 0
            spike_rate_label.config(text=f"Spike Rate: {spike_rate:.2f} Hz")

            # Update neuron visualizations
            R1_normalized = (R1 - np.min(R1)) / (np.max(R1) - np.min(R1))
            for i, rect in enumerate(neuron_rects_1):
                if i < R1.shape[0]:
                    color_value = int(R1_normalized[i, 0] * 255)
                    hex_color = "#{:02x}{:02x}{:02x}".format(color_value, 0, 255 - color_value)
                    neuron_canvas_1.itemconfig(rect, fill=hex_color)

            for i, rect in enumerate(neuron_rects_2):
                if i < activity.shape[0]:
                    neuron_canvas_2.itemconfig(
                        rect,
                        fill="red" if np.sum(activity[i, :]) > 0 else "blue"
                    )

            # Update raster plots
            ax1.clear()
            ax2.clear()

            # Plot Layer 1 activity
            for i in range(R1.shape[0]):
                spike_times = np.where(R1[i, :] > 0)[0] * parameters["dt"]
                if len(spike_times) > 0:
                    ax1.plot(
                        spike_times + elapsed_time,
                        np.ones_like(spike_times) * i,
                        "|k",
                        markersize=3
                    )

            # Plot Layer 2 activity
            for i in range(activity.shape[0]):
                spike_times = np.where(activity[i, :] == 1)[0] * parameters["dt"]
                if len(spike_times) > 0:
                    ax2.plot(
                        spike_times + elapsed_time,
                        np.ones_like(spike_times) * i,
                        "|k",
                        markersize=3
                    )

            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Neuron Index")
            ax2.set_xlabel("Time (s)")
            ax2.set_ylabel("Neuron Index")

            raster_plot1.draw()
            raster_plot2.draw()

            # Queue next input
            input_queue.put(static_input if image_loaded else np.random.poisson(lam=2, size=(784, 1)))

        root.after(100, update_plot)

    # Start the update loop
    root.after(100, update_plot)
    root.mainloop()

if __name__ == "__main__":
    # Initialize with random weights
    W0 = np.random.normal(0, 1, size=(500, 784))
    W1 = np.random.uniform(-1, 1, size=(10, 500))
    
    # Set default parameters
    parameters = {
        "dt": 0.005,
        "gain": 1.0,
        "nrep": 5
    }
    
    # Run the live experiment
    run_experiment_live(W0, W1, parameters)