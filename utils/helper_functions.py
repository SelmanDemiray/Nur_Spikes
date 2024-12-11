# helper_functions.py
import numpy as np
from scipy.stats import poisson
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from PIL import Image
import requests
import os
import gzip
import shutil


def linrectify(X):
    """Performs linear rectification on input."""
    return np.maximum(X, 0)


def spikes(images, weights, parameters):
    """Generates spikes for the population of neurons."""
    assert isinstance(images, np.ndarray), "Invalid 'images' array"
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"
    assert 'dt' in parameters and parameters[
        'dt'] > 0, "Invalid 'dt' parameter"
    assert 'gain' in parameters and parameters[
        'gain'] > 0, "Invalid 'gain' parameter"

    R = rates(images, weights, parameters)

    # Use a smaller size for the random values
    S = poisson.rvs(mu=parameters['dt'] * R,
                    size=R.shape)  
    return S


def confusion(pos):
    """Generates a confusion matrix."""
    num_images = pos.shape[1]
    num_categories = int(num_images / 100)

    cm = np.zeros((num_categories, num_categories))
    ind = np.zeros(num_images, dtype=int)

    mapind = np.argmax(pos, axis=0)
    mapcat = mapind // 100
    for c in range(num_categories):
        cm[c, :] = np.bincount(mapcat[c * 100:(c + 1) * 100],
                                minlength=num_categories)
    ind = mapind

    cm = cm / cm.sum(axis=1, keepdims=True)
    return cm, ind


def loglikelihood(activity, image_embeddings, weights, parameters):
    """Calculates the log likelihood of the activity."""
    assert 'dt' in parameters and parameters[
        'dt'] > 0, "Invalid 'dt' parameter"
    assert 'gain' in parameters and parameters[
        'gain'] > 0, "Invalid 'gain' parameter"
    assert 'nrep' in parameters and parameters[
        'nrep'] > 0, "Invalid 'nrep' parameter"

    assert isinstance(activity, np.ndarray), "Invalid 'activity' array"
    assert isinstance(image_embeddings,
                      np.ndarray), "Invalid 'image_embeddings' array"
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"

    meanact = np.ceil(np.mean(activity, axis=2)).astype(int)
    LL = np.zeros((image_embeddings.shape[1], image_embeddings.shape[1]))
    R = rates(image_embeddings, weights, parameters) * parameters['dt']

    A = np.tile(meanact[:, :, np.newaxis], (1, 1, image_embeddings.shape[1]))
    PA = poisson.pmf(A, R[:, :, np.newaxis])
    LPA = np.log(PA + 1e-10)
    LL = np.sum(LPA, axis=0)

    return LL


def logprior(cp, num_images):
    """Returns the log prior for the images."""
    assert isinstance(cp, np.ndarray) and np.all(
        cp > 0) and np.isclose(np.sum(cp), 1.0), "Invalid 'cp' array"

    num_categories = len(cp)
    images_per_category = num_images // num_categories

    LP = np.repeat(np.log(cp / images_per_category), images_per_category)

    return LP


def posaverage(images, pos, navg):
    """Returns the average of images weighted by the posterior."""
    assert isinstance(images, np.ndarray), "Invalid 'images' array"
    assert isinstance(pos, np.ndarray) and pos.shape == (
        images.shape[1], images.shape[1]), "Invalid 'pos' array"
    assert isinstance(navg, int) and 0 < navg <= images.shape[
        1], "Invalid 'navg' value"

    pa = np.zeros(images.shape)
    ipos = np.argsort(pos, axis=0)[::-1]
    spos = np.take_along_axis(pos, ipos, axis=0)

    for i in range(images.shape[1]):
        topimages = images[:, ipos[:navg, i]]
        pa[:, i] = np.sum(topimages * spos[:navg, i].reshape(1, -1), axis=1)
        pa[:, i] = pa[:, i] / np.max(pa[:, i])

    return pa


def posterior(LL, LP):
    """Calculates the posterior probability."""
    assert isinstance(LL, np.ndarray) and LL.shape[
        0] == LL.shape[1], "Invalid 'LL' array"
    assert isinstance(LP, np.ndarray) and LP.shape[
        0] == LL.shape[1], "Invalid 'LP' array"

    LPOS = LL + LP.reshape(1, -1)
    POS = np.exp(LPOS - logsumexp(LPOS, axis=0, keepdims=True))
    return POS


def rates(images, weights, parameters):
    """Calculates the firing rates of the neurons."""
    assert isinstance(images, np.ndarray), "Invalid 'images' array"
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"
    assert 'dt' in parameters and parameters[
        'dt'] > 0, "Invalid 'dt' parameter"
    assert 'gain' in parameters and parameters[
        'gain'] > 0, "Invalid 'gain' parameter"

    R = weights @ images
    R = linrectify(R) * parameters['gain']
    return R


def record(images, weights, parameters):
    """Simulates an experimental recording session."""
    assert isinstance(images, np.ndarray), "Invalid 'images' array"
    assert isinstance(weights, np.ndarray), "Invalid 'weights' array"
    assert 'dt' in parameters and parameters[
        'dt'] > 0, "Invalid 'dt' parameter"
    assert 'gain' in parameters and parameters[
        'gain'] > 0, "Invalid 'gain' parameter"
    assert 'nrep' in parameters and parameters[
        'nrep'] > 0, "Invalid 'nrep' parameter"

    activity = np.zeros((weights.shape[0], images.shape[1],
                        parameters['nrep']))

    for n in range(parameters['nrep']):
        activity[:, :, n] = spikes(images, weights, parameters)

    return activity


def load_mnist_images(data_dir, kind="train"):
    """Loads MNIST images."""
    if kind == "train":
        labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
        images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
    elif kind == "test":
        labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
        images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    else:
        raise ValueError("Invalid kind argument. Must be 'train' or 'test'.")

    with open(labels_path, 'rb') as lbpath:
        lbpath.read(8)
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        imgpath.read(16)
        images = np.frombuffer(imgpath.read(),
                               dtype=np.uint8).reshape(len(labels),
                                                       784).T

    return images, labels


def load_cifar10_images(data_dir):
    """Loads CIFAR-10 images."""

    def unpickle(file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    base_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    filepath = os.path.join(data_dir, filename)

    if not os.path.exists(filepath):
        print(f"Downloading {filename}...")
        response = requests.get(base_url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    import tarfile
    print("Extracting CIFAR-10 data...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(data_dir)

    images = []
    labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, "cifar-10-batches-py",
                                  f"data_batch_{i}")
        batch_data = unpickle(batch_file)
        images.append(batch_data[b'data'])
        labels.extend(batch_data[b'labels'])

    images = np.concatenate(images).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    # Convert to grayscale
    images = np.array([
        np.array(Image.fromarray(img).convert('L')).flatten()
        for img in images
    ]).T
    labels = np.array(labels)
    return images, labels


def download_extract_mnist(data_dir):
    """Downloads and extracts the MNIST dataset."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    base_url = "http://yann.lecun.com/exdb/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for file in files:
        filepath = os.path.join(data_dir, file)
        if not os.path.exists(filepath):
            print(f"Downloading {file}...")
            try:
                response = requests.get(base_url + file, stream=True)
                response.raise_for_status()
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {file}: {e}")
                # Try a mirror if the main download fails
                mirror_url = "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/"
                print(f"Trying to download from mirror: {mirror_url + file}")
                try:
                    response = requests.get(mirror_url + file, stream=True)
                    response.raise_for_status()
                    with open(filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                except requests.exceptions.RequestException as e:
                    print(f"Error downloading from mirror: {e}")
                    return

        print(f"Extracting {file}...")
        try:
            with gzip.open(filepath, 'rb') as f_in:
                with open(filepath[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
        except (gzip.BadGzipFile, OSError) as e:
            print(f"Error extracting {file}: {e}")
            return