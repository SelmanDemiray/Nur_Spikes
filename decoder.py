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

    # Row 3: Default Weights
    add_default_weights_btn = tk.Button(button_frame, text="Default Weights", command=add_default_weights)
    add_default_weights_btn.grid(row=2, column=0, padx=5, pady=5)

    remove_default_weights_btn = tk.Button(button_frame, text="Remove Default Weights", command=remove_default_weights)
    remove_default_weights_btn.grid(row=2, column=1, padx=5, pady=5)

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

    def update_plot():
        nonlocal start_time, total_spikes_0, total_spikes_1, image_loaded, static_input, rate_changes_0, rate_changes_1
        if not output_queue.empty():
            S2, R1, S1, R2, timestamp = output_queue.get() #activity corresponds with S2
            elapsed_time = timestamp - start_time

            # Update spike statistics for Layer 0
            spikes_this_frame_0 = np.sum(S1)
            total_spikes_0 += spikes_this_frame_0
            spike_count_label_0.config(text=f"Layer 0 Spike Count: {spikes_this_frame_0}")
            total_spike_count_label_0.config(text=f"Layer 0 Total Spike Count: {total_spikes_0}")
            spike_rate_0 = total_spikes_0 / elapsed_time if elapsed_time > 0 else 0
            spike_rate_label_0.config(text=f"Layer 0 Spike Rate: {spike_rate_0:.2f} Hz")

            # Update spike statistics for Layer 1
            spikes_this_frame_1 = np.sum(S2)  
            total_spikes_1 += spikes_this_frame_1
            spike_count_label_1.config(text=f"Layer 1 Spike Count: {spikes_this_frame_1}")
            total_spike_count_label_1.config(text=f"Layer 1 Total Spike Count: {total_spikes_1}")
            spike_rate_1 = total_spikes_1 / elapsed_time if elapsed_time > 0 else 0
            spike_rate_label_1.config(text=f"Layer 1 Spike Rate: {spike_rate_1:.2f} Hz")

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
            R1_normalized = (R1 - np.min(R1)) / (np.max(R1) - np.min(R1))
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


            if weights_loaded and W0 is not None:
                # Normalize the weights for visualization
                if USE_CUPY:
                    W0_normalized = cp.asnumpy(W0)
                else:
                    W0_normalized = W0
                W0_normalized = (W0_normalized - np.min(W0_normalized)) / (np.max(W0_normalized) - np.min(W0_normalized))
                W0_resized = np.kron(W0_normalized, np.ones((5, 5))) * 255

                # Convert to a PIL Image
                W0_image = Image.fromarray(W0_resized.astype(np.uint8))
                W0_photo = ImageTk.PhotoImage(W0_image)

                # Display on the canvas
                weight_canvas.create_image(0, 0, anchor=tk.NW, image=W0_photo)
                weight_canvas.image = W0_photo  

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

        root.after(100, update_plot)

    # Start the update loops
    root.after(0, update_prediction_table)
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