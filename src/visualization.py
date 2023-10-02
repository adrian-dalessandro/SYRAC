import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data.images import patchify

def error_rate_boxplot(y_pred, y_true, save_fig=False, fig_name="error_rate_boxplot.png"):
    # Compute the absolute error
    error = np.abs(y_pred - y_true)
    
    # Define the ranges for x-axis labels
    ranges = [ i for i in range(y_true.min(),min(1500,y_true.max()), 20)] +[np.inf]
    
    # Create an array to store the boxplot data for each range
    data = [error[(y_true >= ranges[i]) & (y_true < ranges[i+1])] for i in range(len(ranges)-1)]
    
    # Create the x-axis labels based on the specified ranges
    x_labels = ['{}-{}'.format(ranges[i], ranges[i+1]-1) for i in range(len(ranges)-1)]
    x_labels[-1] = '> 1500'
    
    # Plot the boxplots
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[2, 1])
    ax = fig.add_subplot(gs[0])
    ax.boxplot(data, showfliers=False)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.hlines(xmin=0, xmax=40, y=146)
    ax.set_title("Distribution of Errors Per Count Range")
    ax.set_ylabel("Mean Absolute Error")
    ax1 = fig.add_subplot(gs[1])
    ax1.bar(x_labels, [len(list(d)) for d in data])
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_ylabel("Number of Examples")
    ax1.set_title("Distribution of Number of Examples Per Count Rannge")
    plt.title("Errors Compared for Different Count Ranges")
    plt.tight_layout()
    if save_fig:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()

def visualize_patch_count(image, N, predictions, y_true, save_fig, fig_name = "patch_count.png"):
    """
    Displays a visualization of the image divided into patches, with each patch labeled with its predicted count.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        N (int): Number of patches to divide the image into along each dimension (N x N patches).
        predictions (np.ndarray): Predicted counts for each patch.
        y_true (int): True count of objects in the image.
        save_fig (bool, optional): Whether to save the visualization as an image file. Default is False.
        fig_name (str, optional): Name of the saved figure if 'save_fig' is True. Default is "patch_count.png".

    Returns:
        None
    """
    # Create subplots for displaying patches
    _, axes = plt.subplots(ncols=N, nrows=N, figsize=(6,5), facecolor="w")

    # Iterate through patches and predictions
    for ax, patch, p in zip(axes.flatten(), patchify(image[0], N), predictions):
        ax.imshow(patch)
        ax.axis("off")
        # Display predicted count as text on the patch
        ax.text(2, 2, np.round(p).astype(np.int32),
            horizontalalignment='left',
            verticalalignment='top', fontsize="large", fontweight="extra bold", 
                color="white", backgroundcolor="black")
    # Set the title with true and rounded sum of predicted counts
    plt.suptitle("True Count: {}, Predicted Count: {}".format(y_true, np.round(np.sum(predictions)).astype(np.int32)))
    plt.tight_layout(pad=0.5)
    if save_fig:
        plt.savefig(fig_name)
        plt.close()
    else:
        plt.show()