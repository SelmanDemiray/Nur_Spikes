# decoder.py
import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import requests
import os
import gzip
import shutil
import argparse
import json
from collections import Counter
import time
import threading
import queue
import tkinter as tk
from tkinter import filedialog
import scipy.io as sio
import logging

from utils.helper_functions import (confusion, loglikelihood, linrectify,
                                   logprior, posaverage, posterior, rates,
                                   record, download_extract_mnist, spikes,
                                   load_mnist_images, load_cifar10_images)

# Configure logging
logging.basicConfig(filename='snn_experiment.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def load_weights(file_path):
    """Loads weights from a .mat file."""
    try:
        mat_data = sio.loadmat(file_path)
        return mat_data['weights']
    except (FileNotFoundError, KeyError) as e:
        logging.error(f"Error loading weights: {e}")
        raise


# Load the weights
weights = load_weights(r'D:\SNN\dev_main\Nur_Spikes\weights.mat')


def show_cm(cm, vis='on'):
    """Plots a confusion matrix."""
    if not (isinstance(cm, np.ndarray) and cm.shape[0] == cm.shape[1]):
        logging.warning(
            "Invalid confusion matrix provided. Skipping visualization.")
        return

    num_categories = cm.shape[0]
    plt.figure()
    plt.imshow(100 * cm, cmap='jet')
    plt.xticks(np.arange(num_categories), np.arange(num_categories))
    plt.yticks(np.arange(num_categories), np.arange(num_categories))
    plt.xlabel('Decoded image category')
    plt.ylabel('Presented image category')
    cbar = plt.colorbar()
    cbar.set_label('Categorization frequency (%)')
    plt.set_cmap('jet')

    if vis == 'on':
        plt.show()


def show_images(images, category=None, num_cols=10):
    """Displays images from a dataset."""
    if not isinstance(images, np.ndarray):
        logging.warning(
            "Invalid image data provided. Skipping visualization.")
        return

    if category is not None:
        if not (isinstance(category, int)
                and 0 <= category <= (images.shape[1] / 100) - 1):
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
    plt.set_cmap('gray')

    for i in range(num_images):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(
            images[:, i].reshape(int(np.sqrt(images.shape[0])),
                                 int(np.sqrt(images.shape[0]))).T)
        plt.axis('equal')
        plt.axis('off')

    if category is not None:
        plt.suptitle(f"Category {category} images")
    else:
        plt.suptitle("All images")
    plt.show()


def show_weights(weights, num_cols=25):
    """Displays all the receptive field structures as gray images."""
    if not isinstance(weights, np.ndarray):
        logging.warning(
            "Invalid weight data provided. Skipping visualization.")
        return

    num_fields = weights.shape[0]
    num_rows = int(np.ceil(num_fields / num_cols))
    plt.figure()
    plt.set_cmap('gray')

    for i in range(num_fields):
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(
            weights[i, :].reshape(int(np.sqrt(weights.shape[1])),
                                  int(np.sqrt(weights.shape[1]))).T)
        plt.axis('equal')
        plt.axis('off')

    plt.suptitle("Receptive fields")
    plt.show()


def show_spikes(activity, parameters):
    """Displays a raster plot of spike activity."""
    plt.figure()

    # Optimized plotting: Collect all spike times and plot once per neuron
    for neuron_idx in range(activity.shape[0]):
        spike_times = np.where(activity[neuron_idx, :, :] == 1)[0] * parameters[
            'dt']
        plt.plot(spike_times,
                 np.ones_like(spike_times) * neuron_idx,
                 '|k',
                 markersize=1)  # Smaller marker size

    plt.xlabel('Time (s)')
    plt.ylabel('Neuron Index')
    plt.title('Spike Raster Plot')
    plt.show()


def show_spike_histogram(activity, parameters):
    """Displays a histogram of spike counts across neurons."""
    plt.figure()
    all_spike_counts = np.sum(
        activity, axis=(1, 2))  # Sum across images and repetitions
    plt.hist(all_spike_counts, bins=20)
    plt.xlabel('Spike Count')
    plt.ylabel('Number of Neurons')
    plt.title('Spike Count Histogram')
    plt.show()


def show_spike_rate_plot(activity, parameters):
    """Displays a plot of spike rates over time, averaged across neurons."""
    plt.figure()
    spike_rates = np.mean(activity, axis=0)  # Average across neurons
    time_axis = np.arange(activity.shape[1]) * parameters['dt']
    for rep in range(activity.shape[2]):
        plt.plot(time_axis, spike_rates[:, rep])
    plt.xlabel('Time (s)')
    plt.ylabel('Average Spike Rate')
    plt.title('Average Spike Rate Over Time')
    plt.show()


def show_neuron_heatmap(activity, parameters):
    """Displays a heatmap of spike activity for each neuron."""
    plt.figure()
    plt.imshow(np.sum(activity, axis=2),
               cmap='hot',
               interpolation='nearest',
               aspect='auto')
    plt.xlabel('Image Index')
    plt.ylabel('Neuron Index')
    plt.title('Neuron Spike Activity Heatmap')
    plt.colorbar()
    plt.show()


def load_images(image_paths):
    """Loads images from a list of file paths."""
    try:
        # Assuming all images have the same size and mode
        return np.array([
            np.array(Image.open(path).convert('L')).flatten()
            for path in image_paths
        ]).T
    except (FileNotFoundError, PIL.UnidentifiedImageError) as e:
        logging.warning(f"Error loading images: {e}")
        return np.array([])  # Return an empty array


def calculate_category_priors(labels):
    """Calculates category priors based on label frequencies."""
    if not isinstance(labels, np.ndarray):
        logging.warning("Invalid label data provided. Using uniform priors.")
        return np.full(10, 1 / 10)

    label_counts = Counter(labels)
    num_images = len(labels)
    return np.array([
        label_counts[i] / num_images for i in range(len(label_counts))
    ])


def run_experiment(images, labels, parameters, spike_formats, weights):
    """Runs a single decoding experiment."""
    cp = calculate_category_priors(labels)

    logging.info("Beginning decoding experiment:")
    logging.info(f"Parameters: {parameters}")

    activity = record(images, weights, parameters)

    for spike_format in spike_formats:
        if spike_format == "raster":
            show_spikes(activity, parameters)
        elif spike_format == "histogram":
            show_spike_histogram(activity, parameters)
        elif spike_format == "rate":
            show_spike_rate_plot(activity, parameters)
        elif spike_format == "heatmap":
            show_neuron_heatmap(activity, parameters)

    ll = loglikelihood(activity, images, weights, parameters)
    lp = logprior(cp, images.shape[1])
    pos = posterior(ll, lp)
    cm, ind = confusion(pos)
    pa = posaverage(images, pos, 10)

    filename = f"figures/case_dt{parameters['dt']:.3f}_gain{parameters['gain']:.1f}_nreps{parameters['nrep']}"
    show_cm(cm, vis='off')
    try:
        plt.savefig(f"{filename}.png", dpi=300)
    except Exception as e:
        logging.warning(f"Error saving figure: {e}")
    plt.close()

    show_images(images)
    show_images(pa)

    logging.info("Finished decoding experiment.")


def record_live(weights, parameters, input_queue, output_queue):
    """Continuously generates spikes for live decoding."""
    try:
        while True:
            images = input_queue.get()
            if images is None:
                break

            activity = spikes(images, weights, parameters)
            output_queue.put((activity, time.time()))
    except Exception as e:
        logging.error(f"Error in record_live thread: {e}")


def run_experiment_live(weights, parameters):
    """Runs a live decoding experiment with a GUI."""
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    root = tk.Tk()
    root.title("Live SNN Experiment")

    neuron_window = tk.Toplevel(root)
    neuron_window.title("Neuron Activity")

    # Calculate canvas dimensions based on the number of neurons
    num_neurons = weights.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_neurons)))
    canvas_width = grid_size * 50
    canvas_height = grid_size * 50

    neuron_canvas = tk.Canvas(neuron_window, width=canvas_width, height=canvas_height)
    neuron_canvas.pack()

    neuron_rects = []
    for i in range(num_neurons):
        row = i // grid_size
        col = i % grid_size
        x1 = col * 50
        y1 = row * 50
        x2 = x1 + 40
        y2 = y1 + 40
        rect = neuron_canvas.create_rectangle(x1, y1, x2, y2, fill="blue")
        neuron_rects.append(rect)

    spike_count_label = tk.Label(root, text="Spike Count: 0")
    spike_count_label.pack()
    spike_rate_label = tk.Label(root, text="Spike Rate: 0")
    spike_rate_label.pack()

    fig, ax = plt.subplots()
    line, = ax.plot([], [], '|k')  # Initialize line object for raster plot
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron Index')
    ax.set_title('Live Spike Raster Plot')
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack()

    image_frame = tk.Frame(root)
    image_frame.pack()
    image_label = tk.Label(image_frame, text="No Image Loaded")
    image_label.pack()

    static_input = np.random.rand(784, 1)

    def upload_image():
        nonlocal static_input
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif")])
        if file_path:
            try:
                image = Image.open(file_path).convert('L')
                image = image.resize((28, 28))
                static_input = np.array(image).flatten().reshape(784, 1)
                image_label.config(text=file_path)
            except Exception as e:
                logging.warning(f"Error loading image: {e}")
                image_label.config(text="Error loading image")
        else:
            image_label.config(text="No Image Loaded")

    upload_button = tk.Button(root, text="Upload Image", command=upload_image)
    upload_button.pack()

    record_thread = threading.Thread(target=record_live,
                                     args=(weights, parameters,
                                           input_queue, output_queue))
    record_thread.daemon = True
    record_thread.start()

    start_time = time.time()

    def update_plot():
        nonlocal start_time
        try:
            while not output_queue.empty():  # Process all available data
                activity, timestamp = output_queue.get()
                elapsed_time = timestamp - start_time

                total_spikes = np.sum(activity)
                spike_count_label.config(text=f"Spike Count: {total_spikes}")

                spike_rate = total_spikes / elapsed_time if elapsed_time > 0 else 0
                spike_rate_label.config(text=f"Spike Rate: {spike_rate:.2f}")

                # Update neuron colors more efficiently
                for neuron_idx, rect in enumerate(neuron_rects):
                    neuron_canvas.itemconfig(rect, fill="red" if np.any(activity[neuron_idx, :] == 1) else "blue")

                # Plot spikes
                for neuron_idx in range(activity.shape[0]):
                    spike_times = np.where(activity[neuron_idx, :] == 1)[0] * parameters['dt']
                    if len(spike_times) > 0:  # Only plot if there are spikes
                        ax.plot(spike_times + elapsed_time,
                                np.ones_like(spike_times) * neuron_idx,
                                '|k')

                ax.relim()
                ax.autoscale_view(True, True, True)
                canvas.draw()

            input_queue.put(static_input)

        except KeyboardInterrupt:
            logging.info("Stopping experiment...")
            input_queue.put(None)  # Signal the thread to stop
            record_thread.join()  # Wait for the thread to finish
            root.destroy()
            return

        finally:
            root.after(100, update_plot)  # Reschedule update

    root.after(100, update_plot)
    root.mainloop()

def run_experiment_activate(weights, parameters):
    """Activates the neural population with custom input."""
    input_queue = queue.Queue()
    output_queue = queue.Queue()

    record_thread = threading.Thread(target=record_live,
                                       args=(weights, parameters,
                                             input_queue, output_queue))
    record_thread.daemon = True
    record_thread.start()

    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot([], [], '|k')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Neuron Index')
    ax.set_title('Neural Population Activity')

    start_time = time.time()

    try:
        while True:
            input_str = input(
                "Enter input values (comma-separated, or 'q' to quit): ")
            if input_str.lower() == 'q':
                break

            try:
                input_values = [
                    float(val.strip()) for val in input_str.split(",")
                ]
                num_pixels = weights.shape[
                    1]  # Get the number of pixels from weights
                new_images = np.array(input_values).reshape(
                    num_pixels, 1)  # Reshape
                input_queue.put(new_images)
            except ValueError:
                print(
                    "Invalid input format. Please enter comma-separated numbers."
                )
                continue

            if not output_queue.empty():
                activity, timestamp = output_queue.get()
                elapsed_time = timestamp - start_time

                for neuron_idx in range(activity.shape[0]):
                    spike_times = np.where(
                        activity[neuron_idx, :] == 1)[0] * parameters['dt']
                    ax.plot(spike_times + elapsed_time,
                            np.ones_like(spike_times) * neuron_idx,
                            '|k')

                ax.relim()
                ax.autoscale_view(True, True, True)
                fig.canvas.flush_events()

            time.sleep(0.1)

    except KeyboardInterrupt:
        logging.info("Stopping experiment...")
    finally:
        input_queue.put(None)
        record_thread.join()
        plt.ioff()
        plt.show()


def cli_interaction(weights):
    """Provides a command-line interface for user interaction."""
    print("Welcome to the Neural Decoding Experiment!")

    dataset_choice = input(
        "Select dataset (1: MNIST, 2: CIFAR-10, 3: Other): ")
    while dataset_choice not in ["1", "2", "3"]:
        print("Invalid choice. Please enter 1, 2, or 3.")
        dataset_choice = input(
            "Select dataset (1: MNIST, 2: CIFAR-10, 3: Other): ")

    if dataset_choice == "1":
        dataset = "mnist"
        data_dir = "data"
        download_extract_mnist(data_dir)
        images, labels = load_mnist_images(data_dir)
    elif dataset_choice == "2":
        dataset = "cifar10"
        data_dir = "data"
        images, labels = load_cifar10_images(data_dir)
    else:
        dataset = "other"
        image_paths = []
        images = load_images(image_paths)
        labels = []

    spike_formats = []
    while True:
        spike_format = input(
            "Select spike format (1: Raster plot, 2: Histogram, 3: Rate plot, 4: Heatmap, 5: Done): "
        )
        while spike_format not in ["1", "2", "3", "4", "5"]:
            print("Invalid choice. Please enter 1, 2, 3, 4, or 5.")
            spike_format = input(
                "Select spike format (1: Raster plot, 2: Histogram, 3: Rate plot, 4: Heatmap, 5: Done): "
            )
        if spike_format == "5":
            break
        spike_formats.append({
            "1": "raster",
            "2": "histogram",
            "3": "rate",
            "4": "heatmap"
        }[spike_format])

    parameters = {}
    while True:
        try:
            parameters['dt'] = float(
                input("Enter time step (dt) (positive value): "))
            if parameters['dt'] <= 0:
                raise ValueError("Time step must be a positive value.")
            parameters['gain'] = float(
                input("Enter neuron gain (positive value): "))
            if parameters['gain'] <= 0:
                raise ValueError("Gain must be a positive value.")
            parameters['nrep'] = int(
                input("Enter number of repetitions (positive integer): "))
            if parameters['nrep'] <= 0:
                raise ValueError(
                    "Number of repetitions must be a positive integer.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    run_experiment(images, labels, parameters, spike_formats, weights)


def main():
    parser = argparse.ArgumentParser(
        description="Run neural decoding experiments.")
    parser.add_argument("--dataset",
                        choices=["mnist", "cifar10", "other"],
                        default="mnist",
                        help="Dataset to use")
    parser.add_argument("--data_dir",
                        default="data",
                        help="Directory to store data")
    parser.add_argument(
        "--spike_format",
        choices=["raster", "histogram", "rate", "heatmap"],
        default="raster",
        help="Spike visualization format")
    parser.add_argument("--params_file",
                        default="params.json",
                        help="JSON file with parameters")
    parser.add_argument("--interactive",
                        "-i",
                        action="store_true",
                        help="Run in interactive mode")
    parser.add_argument("--live",
                        "-l",
                        action="store_true",
                        help="Run in live mode")
    parser.add_argument("--activate",
                        "-a",
                        action="store_true",
                        help="Activate with custom input")
    args = parser.parse_args()

    if args.interactive:
        cli_interaction(weights)
    else:
        if args.dataset == "mnist":
            download_extract_mnist(args.data_dir)
            images, labels = load_mnist_images(args.data_dir)
        elif args.dataset == "cifar10":
            images, labels = load_cifar10_images(args.data_dir)
        elif args.dataset == "other":
            image_paths = []
            images = load_images(image_paths)
            labels = []

        try:
            with open(args.params_file, 'r') as f:
                params_config = json.load(f)
            parameters = params_config['parameters']
            dts = params_config.get('dts', [0.005, 0.01])
            gains = params_config.get('gains', [0.8, 1.2])
            nreps = params_config.get('nreps', [1, 2])
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logging.error(f"Error loading parameters: {e}")
            return

        cp = calculate_category_priors(labels)

        if args.live:
            logging.info("Beginning live decoding experiment:")
            run_experiment_live(weights, parameters)
            logging.info("Finished live decoding experiment.")
        elif args.activate:
            logging.info("Activating neural population with custom input:")
            run_experiment_activate(weights, parameters)
            logging.info("Finished neural population activation.")
        else:
            logging.info("Beginning decoding experiments:")
            for dt in dts:
                for gain in gains:
                    for nrep in nreps:
                        logging.info(
                            f"Running case with dt = {dt:.3f} s, gain = {gain:.1f}, nreps = {nrep}..."
                        )
                        parameters['dt'] = dt
                        parameters['gain'] = gain
                        parameters['nrep'] = nrep

                        activity = record(images, weights, parameters)

                        spike_formats = [
                            args.spike_format
                        ]  # Use specified format

                        if args.spike_format == "raster":
                            show_spikes(activity, parameters)
                        elif args.spike_format == "histogram":
                            show_spike_histogram(activity, parameters)
                        elif args.spike_format == "rate":
                            show_spike_rate_plot(activity, parameters)
                        elif args.spike_format == "heatmap":
                            show_neuron_heatmap(activity, parameters)

                        ll = loglikelihood(activity, images, weights,
                                           parameters)
                        lp = logprior(cp, images.shape[1])
                        pos = posterior(ll, lp)
                        cm, ind = confusion(pos)
                        pa = posaverage(images, pos, 10)

                        filename = f"figures/case_dt{dt:.3f}_gain{gain:.1f}_nreps{nrep}"
                        show_cm(cm, vis='off')
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