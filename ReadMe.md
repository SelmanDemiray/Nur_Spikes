## Neural Decoding Experiment Script

This script simulates neural decoding experiments using different image datasets and parameters. It provides a flexible framework for exploring how populations of neurons represent and decode visual information.

### Features

*   **Dataset Flexibility:**
    *   Built-in support for MNIST and CIFAR-10 datasets with automatic download and extraction.
    *   Option to use custom datasets by providing image files and labels.
*   **Parameter Control:**
    *   Controllable simulation parameters: time step (`dt`), neuron gain (`gain`), and number of repetitions (`nrep`).
    *   Parameter exploration: Easily test different parameter combinations.
*   **Spike Visualization:**
    *   Raster plots: Visualize spike trains of individual neurons over time.
    *   Histograms: Analyze the distribution of spike counts across neurons.
    *   Rate plots: Visualize average spike rates over time.
    *   Heatmaps: Visualize spike activity patterns across neurons and images.
*   **Decoding Analysis:**
    *   Confusion matrix: Evaluate the accuracy of image classification based on neural activity.
    *   Posterior-averaged images: Visualize the decoded representation of images.
*   **Command-line Interface:**
    *   Easy-to-use CLI for setting dataset, data directory, spike visualization format, and parameter file.
*   **Live Decoding:**
    *   Run live decoding experiments with continuous input and real-time spike visualization.
*   **Custom Input:**
    *   Activate the neural population with custom input and visualize spike activity in real-time.
*   **Extensible Codebase:**
    *   Modular functions and clear documentation make it easy to extend and modify the code for specific research needs.


### Live Decoding with Different Inputs

<details>
  <summary> Raster plot of neurons firing to different stimuli in live experiment mode </summary>

  ![Raster plot](path/to/your/raster_plot.png) 

  This raster plot shows the spiking activity of a population of neurons in response to different stimuli presented in live decoding mode. Each row represents a different neuron, and each dot represents an action potential (spike). The x-axis represents time, and the y-axis represents the neuron index. 

  You can observe how different neurons exhibit distinct firing patterns to different stimuli, reflecting their selectivity and encoding properties. This dynamic representation of neural activity is crucial for understanding how the brain processes information in real-time.
</details>

<details>
  <summary>Neuroscience Behind the Raster Plot</summary>

  **Neural Encoding and Decoding:**

  *  **Encoding:** The process by which neurons transform external stimuli into patterns of electrical activity. Different stimuli elicit distinct patterns of spikes in the neural population.
  *  **Decoding:** The process of interpreting these spike patterns to reconstruct the original stimulus or extract information about it.

  **Raster Plots and Neural Activity:**

  * Raster plots provide a visual representation of the temporal dynamics of neural activity.
  * The timing and frequency of spikes are crucial for encoding and decoding information in the brain.
  * Different neurons may respond selectively to different features of a stimulus, creating a distributed representation.

  **Live Decoding:**

  *  Live decoding experiments allow researchers to observe and analyze neural activity in real-time as stimuli are presented.
  *  This provides insights into the dynamic processing of information in the brain and can be used to develop brain-computer interfaces and other neurotechnologies.
</details>

### Usage

To run the script, use the following command:

```bash
python decoder.py [arguments]
```

**Arguments:**

*   `--dataset`: Specifies the dataset to use. Options are:
    *   `mnist`: Downloads and uses the MNIST dataset.
    *   `cifar10`: Downloads and uses the CIFAR-10 dataset.
    *   `other`: Uses a custom dataset (you need to implement loading logic in the code). Default: `mnist`
*   `--data_dir`: Specifies the directory to store downloaded data. Default: `data`
*   `--spike_format`: Specifies the format for visualizing spike activity. Options are: `raster`, `histogram`, `rate`, `heatmap`. Default: `raster`
*   `--params_file`: Specifies the JSON file containing experiment parameters. Default: `params.json`
*   `--interactive`, `-i`: Run in interactive mode.
*   `--live`, `-l`: Run in live mode with a GUI for image input.
*   `--activate`, `-a`: Activate the neural population with custom input.

**Example Commands:**

To run the script with the MNIST dataset and raster plots:

```bash
python decoder.py --dataset mnist --spike_format raster
```

To run the script with the CIFAR-10 dataset and histograms:

```bash
python decoder.py --dataset cifar10 --spike_format histogram
```

To run the script with a custom dataset (you need to implement loading logic in the code):

```bash
python decoder.py --dataset other --data_dir my_data --params_file my_params.json
```

### Configuration File (`params.json`)

The `params.json` file contains the parameters for the experiments. Here's an example:

```json
{
  "parameters": {
    "dt": 0.01,
    "gain": 1.0,
    "nrep": 5
  },
  "dts": [0.005, 0.01],
  "gains": [0.8, 1.2],
  "nreps": [1, 2]
}
```

*   `parameters`: Default values for the simulation time step (`dt`), neuron gain (`gain`), and number of repetitions (`nrep`).
*   `dts`, `gains`, `nreps`: Lists of values to test for each parameter.

### Output

The script generates the following output:

*   Spike activity visualizations: Raster plots, histograms, rate plots, or heatmaps, displayed during the experiment.
*   Confusion matrices: Saved as PNG images in the `figures` directory.
*   Posterior-averaged images: Displayed after each experiment.

### Requirements

*   Python 3.6 or higher
*   NumPy
*   SciPy
*   Matplotlib
*   Pillow (PIL)
*   Requests

To install the necessary packages, run:

```bash
pip install numpy scipy matplotlib pillow requests
```

### Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request on the GitHub repository.

### License

This project is licensed under the MIT License - see the LICENSE file for details.
```

**Remember to replace `path/to/your/raster_plot.png` with the actual path to your raster plot image.** 

This updated README.md file includes the raster plot image, expandable sections for detailed explanations, and clear formatting for better readability. This will make your project more engaging and informative for users.
