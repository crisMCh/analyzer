import pydicom
import argparse
import pandas as pd
import numpy as np
from pydicom.dataset import Dataset
import os
import csv
import ast
import glob
import matplotlib.pyplot  as plt
import cv2
import uuid
import matplotlib.patches as patches
from datetime import datetime

import matplotlib.patches as patches

##use this if you know the top left corner and the width and height of the roi
#def plot_with_rois(image, ax, args, roi1, roi2=None):
#    """
#    Plots an image with two ROIs highlighted.
#
#    Parameters:
#    - image: The image to display.
#    - roi1: The first ROI as (x, y, w, h).
#    - roi2: The second ROI as (x, y, w, h).
#    - ax: The matplotlib axis to plot on.
#    """
#    # Display the image
#    ax.imshow(image, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
#    
#    # Draw the first ROI
#    if roi1:
#        x1, y1, w1, h1 = roi1
#        rect1 = patches.Rectangle((x1, y1), w1, h1, linewidth=1, edgecolor='red', facecolor='none')
#        ax.add_patch(rect1)
#    
#    # Draw the second ROI
#    if roi2:
#        x2, y2, w2, h2 = roi2
#        rect2 = patches.Rectangle((x2, y2), w2, h2, linewidth=1, edgecolor='blue', facecolor='none')
#        ax.add_patch(rect2)
#


#use this if you know the center point and the width and height of the roi
def plot_with_rois(image, ax, args, roi1, roi2=None, diff = False):
    """
    Plots an image with two ROIs highlighted.

    Parameters:
    - image: The image to display.
    - roi1: The first ROI as (cx, cy, w, h).
    - roi2: The second ROI as (cx, cy, w, h).
    - ax: The matplotlib axis to plot on.
    """
    # Display the image
    if diff:
        ax.imshow(image, vmin=args.trunc_min, vmax=args.trunc_max)
    else:
        ax.imshow(image, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
    
    # Draw the first ROI
    if roi1:
        cx1, cy1, h1, w1 = roi1
        x1 = cx1 - w1 // 2
        y1 = cy1 - h1 // 2
        rect1 = patches.Rectangle((x1, y1), w1, h1, linewidth=1, edgecolor='red', facecolor='none')
        ax.add_patch(rect1)
    
    # Draw the second ROI
    if roi2:
        cx2, cy2, w2, h2 = roi2
        x2 = cx2 - w2 // 2
        y2 = cy2 - h2 // 2
        rect2 = patches.Rectangle((x2, y2), w2, h2, linewidth=1, edgecolor='blue', facecolor='none')
        ax.add_patch(rect2)
        

def trunc(mat, args):
        mat[mat <= args.trunc_min] = args.trunc_min
        mat[mat >= args.trunc_max] = args.trunc_max
        return mat

def denormalize_(image, args):
        image = image * (args.norm_range_max - args.norm_range_min) + args.norm_range_min
        return image

def extract_roi(slice_array, roi):
    """
    Extract a region of interest (ROI) from a given slice.

    Args:
        slice_array (numpy.ndarray): 2D array representing the image slice.
        roi (tuple): Region of interest as (x, y, width, height).

    Returns:
        numpy.ndarray: Extracted ROI patch from the slice.
    """
    x, y, width, height = roi
    return slice_array[y:y + height, x:x + width]


def extract_roi(slice_array, roi):
    """
    Extract a region of interest (ROI) from a given slice.

    Args:
        slice_array (numpy.ndarray): 2D array representing the image slice.
        roi (tuple): Region of interest as (cx, cy, width, height), where (cx, cy) is the center of the ROI.

    Returns:
        numpy.ndarray: Extracted ROI patch from the slice.
    """
    cx, cy, height, width = roi
    x = cx - width // 2
    y = cy - height // 2
    return slice_array[y:y + height, x:x + width]

def show_roi(image, roi):
    x, y, w, h = roi  # Assuming ROI is (x, y, width, height)
    return cv2.rectangle(image.copy(), (x, y), (x + w, y + h), (255, 0, 0), 3)


def dicom_path(path,model,noise):
      """Finds path following the pattern"""
      pattern = os.path.join(path,f'{model}_*_n{noise}','dcm')
      matches = glob.glob(pattern)
      return matches[0] if matches else None

def show_interactive_plot(image, original_img, diff, args):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes = axes.tolist()  # Convert axes array to a Python list
    titles = [f"Reconstructed {args.model} {args.noise_level}", "Original", "Difference"]
    images = [image, original_img, diff]
    print(np.max(np.abs(diff)))


   # Display images
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap=plt.cm.gray, vmin=(args.trunc_min if i < 2 else np.min(np.abs(diff))),
                  vmax=(args.trunc_max if i < 2 else np.max(np.abs(diff))))
        ax.set_title(titles[i], fontsize=30)
        ax.axis("off")

    # Function to display pixel values
    def on_click(event):
        if event.inaxes in axes:  # Check if the click was inside one of the subplots
            ax_idx = axes.index(event.inaxes)  # Get the index of the clicked subplot
            x, y = int(event.xdata), int(event.ydata)
            pixel_values = [img[y, x] for img in images]
            print(f"Coordinates: ({x}, {y})")
            print(f"Low dose value: {pixel_values[0]:.2f}")
            print(f"Full dose value: {pixel_values[1]:.2f}")
            print(f"Difference value: {pixel_values[2]:.2f}")

            # Annotate the pixel value on the plots
            for i, ax in enumerate(axes):
                ax.clear()
                ax.imshow(images[i], cmap=plt.cm.gray, vmin=(args.trunc_min if i < 2 else 0),
                          vmax=(args.trunc_max if i < 2 else np.max(np.abs(diff))))
                ax.set_title(titles[i], fontsize=30)
                ax.axis("off")
                if i == ax_idx:
                    ax.scatter(x, y, color="red", s=100)
                    ax.annotate(f"{pixel_values[i]:.2f}", (x, y), color="red",
                                fontsize=12, ha="center", va="bottom")

            fig.canvas.draw_idle()

    # Connect the event
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.tight_layout()
    plt.show()

def show_interactive_plot(image, original_img, diff, args):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    axes = axes.tolist()  # Convert axes array to a Python list
    titles = [f"Reconstructed {args.model} {args.noise_level}", "Original", "Difference"]
    images = [image, original_img, diff]
    print(np.max(np.abs(diff)))


   # Display images
    for i, ax in enumerate(axes):
        ax.imshow(images[i], cmap=plt.cm.gray, vmin=(args.trunc_min if i < 2 else np.min(np.abs(diff))),
                  vmax=(args.trunc_max if i < 2 else np.max(np.abs(diff))))
        ax.set_title(titles[i], fontsize=30)
        ax.axis("off")

    # Function to display pixel values
    def on_click(event):
        if event.inaxes in axes:  # Check if the click was inside one of the subplots
            ax_idx = axes.index(event.inaxes)  # Get the index of the clicked subplot
            x, y = int(event.xdata), int(event.ydata)
            pixel_values = [img[y, x] for img in images]
            print(f"Coordinates: ({x}, {y})")
            print(f"Low dose value: {pixel_values[0]:.2f}")
            print(f"Full dose value: {pixel_values[1]:.2f}")
            print(f"Difference value: {pixel_values[2]:.2f}")

            # Annotate the pixel value on the plots
            for i, ax in enumerate(axes):
                ax.clear()
                ax.imshow(images[i], cmap=plt.cm.gray, vmin=(args.trunc_min if i < 2 else 0),
                          vmax=(args.trunc_max if i < 2 else np.max(np.abs(diff))))
                ax.set_title(titles[i], fontsize=30)
                ax.axis("off")
                if i == ax_idx:
                    ax.scatter(x, y, color="red", s=100)
                    ax.annotate(f"{pixel_values[i]:.2f}", (x, y), color="red",
                                fontsize=12, ha="center", va="bottom")

            fig.canvas.draw_idle()

    # Connect the event
    fig.canvas.mpl_connect('button_press_event', on_click)

    plt.tight_layout()
    plt.show()


def plot_comparison(slices, originals, models, images_dir, args):
    """
    Plot multiple slices and model predictions in a grid.

    Args:
        slices (list): List of prediction file names (e.g., DICOM files).
        originals (list): List of corresponding ground truth file names.
        models (list): List of model names.
        images_dir (str): Path to the folder containing ground truth files.
        args (Namespace): Contains parameters such as trunc_min, trunc_max, noise_level, etc.
    """
    n_models = len(models)
    n_slices = len(slices)
    

    fig, axes = plt.subplots(n_slices, n_models + 2, figsize=(3 * (n_models + 2), 3 * n_slices), constrained_layout=False)
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    if n_models == 1 or n_slices == 1:
        axes = np.array(axes).reshape(n_slices, n_models + 2)

    for i, (slice_file, original_file) in enumerate(zip(slices, originals)):
        # Load noisy input and ground truth
        noisy_path = os.path.join(images_dir, "input", str(args.noise_level), original_file)
        groundtruth_path = os.path.join(images_dir, 'groundtruth', original_file)

        noisy_img = denormalize_(np.load(noisy_path), args)
        original_img = denormalize_(np.load(groundtruth_path), args)

        # Plot ground truth and noisy input
        axes[i, 0].imshow(original_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        if i == 0:
            axes[i, 0].set_title(f'NDCT', fontsize=15)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        if i == 0:
            axes[i, 1].set_title(f'LDCT', fontsize=15)
        axes[i, 1].axis('off')

        for j, model in enumerate(models):
            dicom_folder = dicom_path(args.predictions_path, model, args.noise_level)
            prediction_path = os.path.join(dicom_folder, slice_file)

            # Load the model prediction
            reconstructed_img = pydicom.dcmread(prediction_path).pixel_array

            # Plot model prediction
            axes[i, j + 2].imshow(reconstructed_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
            if i == 0:
                axes[i, j + 2].set_title(f'{model}', fontsize=15)
            axes[i, j + 2].axis('off')

    plt.show()


def plot_noise_simulation(slices, originals, noise_levels, images_dir, args, roi=None):
    """
    Plot all the noise levels for multiple slices, including the original images, in a single figure with optional ROI selection.

    Args:
        slices (list): List of file names for the slices.
        originals (list): List of corresponding ground truth file names.
        noise_levels (list): List of noise levels to simulate.
        images_dir (str): Path to the folder containing ground truth and noisy files.
        args (Namespace): Contains parameters such as trunc_min, trunc_max, etc.
        roi (tuple, optional): Region of interest as (x, y, width, height). Default is None.
    """
    n_slices = len(slices)
    n_levels = len(noise_levels)

    fig, axes = plt.subplots(n_slices, n_levels + 1, figsize=(3 * (n_levels + 1), 3 * n_slices))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for row, (slice_file, original_file) in enumerate(zip(slices, originals)):
        # Load ground truth
        groundtruth_path = os.path.join(images_dir, 'groundtruth', original_file)
        original_img = denormalize_(np.load(groundtruth_path), args)

        # Apply ROI if provided
        if roi:
            original_img = extract_roi(original_img, roi)

        # Plot ground truth
        axes[row, 0].imshow(original_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        axes[row, 0].set_title('NDCT' if row == 0 else '', fontsize=12)
        axes[row, 0].axis('off')

        for col, noise_level in enumerate(noise_levels):
            # Load noisy input
            noisy_path = os.path.join(images_dir, "input", str(noise_level), original_file)
            noisy_img = denormalize_(np.load(noisy_path), args)

            # Apply ROI if provided
            if roi:
                noisy_img = extract_roi(noisy_img, roi)

            # Plot noisy input
            axes[row, col + 1].imshow(noisy_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
            if row == 0:
                axes[row, col + 1].set_title(f'Noise {noise_level}', fontsize=12)
            axes[row, col + 1].axis('off')

    plt.show()
    plt.close(fig)


def plot_comparison_noises(slice_file, original_file, models, noise_levels, images_dir, args, uid, roi1=None, roi2=None, roi_print=False, diff=False):
    """
    Plot predictions from multiple models across different noise levels for a single slice.
    IMPORTANT: the plots are saved at args.output_figs_path! 

    Args:
        slice_file (str): Prediction file name (e.g., DICOM file) for the slice.
        original_file (str): Corresponding ground truth file name.
        models (list): List of model names.
        noise_levels (list): List of noise levels to compare.
        images_dir (str): Path to the folder containing ground truth files.
        args (Namespace): Contains parameters such as trunc_min, trunc_max, etc.
        roi1 (tuple or None): Coordinates for the 1st region of interest (ROI).
        roi2 (tuple or None): Coordinates for the 2nd region of interest (ROI).
        roi_print (bool): Whether to print the ROI as a red square instead of extracting it.
        diff (bool): Whether to show the difference image instead of the reconstruction.
    """
    n_models = len(models)
    n_levels = len(noise_levels)

    fig, axes = plt.subplots(n_levels, n_models + 2, figsize=(3 * (n_models + 2), 3 * n_levels))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    # Load ground truth
    groundtruth_path = os.path.join(images_dir, 'groundtruth', original_file)
    original_img = denormalize_(np.load(groundtruth_path), args)

    # Find the global max difference for color normalization
    global_max_diff = 0
    for noise_level in noise_levels:
        for model in models:
            if model == 'LIT':
                base, ext = os.path.splitext(slice_file)
                parts = base.split('_')
                number = int(parts[-1]) + 1
                slice_file_fixed = f"{'_'.join(parts[:-1])}_{number:04d}{ext}"
            else:
                slice_file_fixed = slice_file

            dicom_folder = dicom_path(args.predictions_path, model, noise_level)
            prediction_path = os.path.join(dicom_folder, slice_file_fixed)
            reconstructed_img = pydicom.dcmread(prediction_path).pixel_array
            diff_img = np.abs(original_img - reconstructed_img)
            diff_img[diff_img > args.diff_max_threshold] = args.diff_max_threshold
            global_max_diff = max(global_max_diff, np.max(diff_img))

    for i, noise_level in enumerate(noise_levels):
        # Load noisy input
        noisy_path = os.path.join(images_dir, "input", str(noise_level), original_file)
        save_path = args.output_figs_path
        noisy_img = denormalize_(np.load(noisy_path), args)
        
        if roi1:
            if roi_print:
                # Highlight ROI on the original image 
                plot_with_rois(original_img, axes[i, 0], args, roi1)
                plot_with_rois(noisy_img, axes[i, 1], args, roi1)
                if roi2: 
                    plot_with_rois(original_img, axes[i, 0], args, roi1, roi2)
                    plot_with_rois(noisy_img, axes[i, 1], args, roi1, roi2)
            else: 
                noisy_img = extract_roi(noisy_img, roi1)
                original_img_r = extract_roi(original_img, roi1)
                # Plot ground truth and noisy input
                axes[i, 0].imshow(original_img_r, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
                axes[i, 1].imshow(noisy_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        else:
            # Plot ground truth and noisy input without ROI
            axes[i, 0].imshow(original_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
            axes[i, 1].imshow(noisy_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)

        if i == 0:
            axes[i, 0].set_title(f'NDCT', fontsize=12)
        axes[i, 0].axis('off')

        if i == 0:
            axes[i, 1].set_title(f'LDCT', fontsize=12)
        axes[i, 1].axis('off')

        # Add noise level as a label to the left of each row (before first column)
        if noise_level == 40000:
            axes[i, 0].annotate(f'Low noise', xy=(-0.15, 0.5), xycoords='axes fraction', size=12, ha='right', va='center', rotation=90)
        elif noise_level == 20000:
            axes[i, 0].annotate(f'Medium noise', xy=(-0.15, 0.5), xycoords='axes fraction', size=12, ha='right', va='center', rotation=90)
        elif noise_level == 10000:
            axes[i, 0].annotate(f'High noise', xy=(-0.15, 0.5), xycoords='axes fraction', size=12, ha='right', va='center', rotation=90)
        elif noise_level == 5000:
            axes[i, 0].annotate(f'Extreme noise', xy=(-0.15, 0.5), xycoords='axes fraction', size=12, ha='right', va='center', rotation=90)
        else:
            raise TypeError("No valid noise level!")

        for j, model in enumerate(models):
            if model == 'LIT':
                base, ext = os.path.splitext(slice_file)
                parts = base.split('_')
                number = int(parts[-1]) + 1
                slice_file_fixed = f"{'_'.join(parts[:-1])}_{number:04d}{ext}"
            else:
                slice_file_fixed = slice_file

            dicom_folder = dicom_path(args.predictions_path, model, noise_level)
            prediction_path = os.path.join(dicom_folder, slice_file_fixed)

            # Load the model prediction
            reconstructed_img = pydicom.dcmread(prediction_path).pixel_array
            
            diff_img = np.abs(original_img - reconstructed_img)
            # Apply thresholds to diff_img
            diff_img[diff_img < args.diff_min_threshold] = 0
            diff_img[diff_img > args.diff_max_threshold] = args.diff_max_threshold

            if roi1:
                if roi_print:
                    if diff:
                        im = axes[i, j + 2].imshow(diff_img, vmin=0, vmax=global_max_diff)
                    else:
                        # Highlight ROI on the original image 
                        plot_with_rois(reconstructed_img, axes[i, j + 2], args, roi1)
                        if roi2: 
                            plot_with_rois(reconstructed_img, axes[i, j + 2], args, roi1, roi2)
                else: 
                    reconstructed_img = extract_roi(reconstructed_img, roi1)
                    diff_img = extract_roi(diff_img, roi1)
                    # Plot model prediction or difference image
                    if diff:
                        im = axes[i, j + 2].imshow(diff_img, vmin=0, vmax=global_max_diff)
                    else:
                        axes[i, j + 2].imshow(reconstructed_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
            else:
                # Plot model prediction or difference image without ROI
                if diff:
                    im = axes[i, j + 2].imshow(diff_img, vmin=0, vmax=global_max_diff)
                else:
                    axes[i, j + 2].imshow(reconstructed_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)

            if i == 0:
                axes[i, j + 2].set_title(f'{model}', fontsize=12)
            axes[i, j + 2].axis('off')

    # Add colorbar once for the entire figure
    if diff:
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax)

    # Save each image
    save_name = f"{uid}_{os.path.splitext(slice_file)[0]}.png"
    save_full_path = os.path.join(save_path, save_name)
    fig.savefig(save_full_path, bbox_inches='tight')
    
    print(f"Plot saved at: {save_full_path}")
    plt.close()

def visualize_differences_with_threshold(original_file, prediction_file, models, noise_levels, args, threshold_value=None, show_orig=False):
    """
    Visualize the differences between the original and prediction images for multiple models and noise levels with an optional threshold.
    
    Args:
        original_file (str): Path to the original image file.
        prediction_file (str): Path to the prediction image file.
        models (list): List of model names.
        noise_levels (list): List of noise levels to compare.
        args (Namespace): Contains parameters such as trunc_min, trunc_max, etc.
        threshold_value (float, optional): Threshold value to filter the differences. Default is None.
        show_orig (bool, optional): Whether to show the original images. Default is True.
    """
    n_models = len(models)
    n_levels = len(noise_levels)
    n_cols = n_models * 3 + 2 if show_orig else n_models * 2 + 2

    fig, axes = plt.subplots(n_levels, n_cols, figsize=(3 * n_cols, 3 * n_levels))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    # Load ground truth
    groundtruth_path = os.path.join(args.images_dir, 'groundtruth', original_file)
    original_img = denormalize_(np.load(groundtruth_path), args)

    for i, noise_level in enumerate(noise_levels):
        # Load noisy input
        noisy_path = os.path.join(args.images_dir, "input", str(noise_level), original_file)
        noisy_img = denormalize_(np.load(noisy_path), args)

        # Plot ground truth
        axes[i, 0].imshow(original_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        axes[i, 0].set_title('NDCT' if i == 0 else '', fontsize=12)
        axes[i, 0].axis('off')

        # Plot noisy input
        axes[i, 1].imshow(noisy_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        axes[i, 1].set_title('LDCT' if i == 0 else '', fontsize=12)
        axes[i, 1].axis('off')

        for j, model in enumerate(models):
            dicom_folder = dicom_path(args.predictions_path, model, noise_level)
            prediction_path = os.path.join(dicom_folder, prediction_file)

            # Load the model prediction
            reconstructed_img = pydicom.dcmread(prediction_path).pixel_array

            # Calculate the difference
            diff_img = np.abs(original_img - reconstructed_img)
            # Apply threshold if provided
            if threshold_value is not None:
                diff_img[diff_img < threshold_value] = 0

            # Calculate and print average difference
            avg_diff = np.mean(diff_img)
            #print(f"Average difference for model {model} at noise level {noise_level}: {avg_diff:.2f}")

            if show_orig:
                # Plot model prediction
                axes[i, j + 2].imshow(reconstructed_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
                axes[i, j + 2].set_title(f'{model}' if i == 0 else '', fontsize=12)
                axes[i, j + 2].axis('off')

                # Plot the difference
                axes[i, j + 2 + n_models].imshow(diff_img, cmap=plt.cm.gray, vmin=0, vmax=np.max(diff_img))
                axes[i, j + 2 + n_models].set_title(f'{model} Diff' if i == 0 else '', fontsize=12)
                axes[i, j + 2 + n_models].axis('off')

                # Plot the heatmap of the difference between differences
                if j > 0:
                    prev_diff_img = (original_img - pydicom.dcmread(os.path.join(dicom_path(args.predictions_path, models[j-1], noise_level), prediction_file)).pixel_array)
                    diff_of_diff = (diff_img - prev_diff_img)
                    axes[i, j + 2 + 2 * n_models].imshow(diff_of_diff, cmap='viridis', vmin=np.min(diff_of_diff), vmax=0)
                    axes[i, j + 2 + 2 * n_models].set_title(f'{model} Diff Heatmap' if i == 0 else '', fontsize=12)
                    axes[i, j + 2 + 2 * n_models].axis('off')
                    print("min: {}, max: {}".format(np.min(diff_of_diff), np.max(diff_of_diff)))
            else:
                # Plot only the difference
                axes[i, j + 2].imshow(diff_img, cmap=plt.cm.gray, vmin=0, vmax=np.max(diff_img))
                axes[i, j + 2].set_title(f'{model} Diff' if i == 0 else '', fontsize=12)
                axes[i, j + 2].axis('off')

    plt.tight_layout()
    
    # Save the figure
    slice_number = os.path.splitext(prediction_file)[0].split('_')[-1]
    save_name = f"comparison_{slice_number}.png"
    save_full_path = os.path.join(args.output_figs_path, save_name)
    fig.savefig(save_full_path, bbox_inches='tight')
    print(f"Plot saved at: {save_full_path}")
    
    #plt.show()

def main():
    parser = argparse.ArgumentParser(description="Export and print")

    #window
    parser.add_argument('--trunc_min', type=float, default=-1000)#-160, -1000lung
    parser.add_argument('--trunc_max', type=float, default=1000)#240 #400lung, -24
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=1000.0)
    parser.add_argument('--diff_min_threshold', type=float, default=40)
    parser.add_argument('--diff_max_threshold', type=float, default=400)
    


    #paths
    #parser.add_argument("--output_figs_path", type=str,  default='./figs/0_pred/iviolin', help="Path to save the plots.")
    parser.add_argument("--output_figs_path", type=str,  default='./figs', help="Path to save the plots.")
    
    #parser.add_argument("--predictions_path", type=str,  default='./predictions/iviolin', help="Path to ")
    parser.add_argument("--predictions_path", type=str,  default='./predictions/', help="Path to ")
    #parser.add_argument("--output_figs_path", type=str,  default='./figs/iviolin/rois', help="Path to save the plots.")
   
    parser.add_argument('--images_dir', type=str, default='G:/Cristina/Thesis/Models/Uformer/dataset/lung/test', help='Directory of test data, used to retireve full dose and low doses')#testi for iviolin, test for own datset
    
   #---------------------------- need to rename files here ^^^^^^^^ done

    #Model
    parser.add_argument("--model", type=str,  default='UFormer', help="Model Name") #REDCNN,EDCNN, UFormer
    parser.add_argument("--noise_level", type=int,  default=10000, help="")

    # Parse arguments
    args = parser.parse_args()
    print('test')
    ##### x,   y,   w,  h
    roi_s56=[180, 140, 100, 80]  #slice56 liver  
    roi_s24=[300, 270, 80, 65] # slice24 lung
    roi_s91l=[315, 180, 85, 110]  #slice91 lung mip left
    roi_s91r=[125, 210, 70, 90]  #slice91 lung mip right
    roi_s123=[175, 170 , 115, 130] ##123 shoulder
    rois = [roi_s56, roi_s24, roi_s91r, roi_s123]

    #slices = ["prediction_0056.dcm","prediction_0091.dcm", "prediction_0123.dcm"]
    slices = ["prediction_0056.dcm","prediction_0024.dcm","prediction_0091.dcm", "prediction_0123.dcm"]
    slices2 = ["prediction_0018.dcm","prediction_0023.dcm","prediction_0040.dcm", "prediction_0064.dcm" , "prediction_0079.dcm", "prediction_0118.dcm"]

    #originals = ["13094367_I00056_target.npy","13094367_I00091_target.npy", "13094367_I00123_target.npy"]
    originals = ["13094367_I00056_target.npy","13094367_I00024_target.npy","13094367_I00091_target.npy", "13094367_I00123_target.npy"]
    originals2 = ["13094367_I00018_target.npy","13094367_I00023_target.npy","13094367_I00040_target.npy","13094367_I00064_target.npy", "13094367_I00079_target.npy", "13094367_I00118_target.npy"]
    rois2= [[424, 231, 74, 75], [307, 364, 54, 96], [153, 274, 55, 67], [248, 203, 71, 111], [189, 187, 37, 57], [246, 176, 106, 130] ]
    

    #iviolin
    slicesi = ["prediction_0000.dcm","prediction_0001.dcm","prediction_0002.dcm", "prediction_0003.dcm", "prediction_0004.dcm", "prediction_0005.dcm", "prediction_0006.dcm", "prediction_0007.dcm", "prediction_0008.dcm"]
    originalsi = ["iviolin_abdo_pat1_target.npy","iviolin_abdo_pat2_target.npy","iviolin_thor_acc_pat1_target.npy", "iviolin_thor_acc_pat2_target.npy", "iviolin_thor_acc_pat3_target.npy", "iviolin_thor_acc_pat4_target.npy", "iviolin_thor_nacc_pat2_target.npy", "iviolin_thor_nacc_pat3_target.npy", "iviolin_thor_nacc_pat4_target.npy"]
    roisi1f = [
        [158, 382, 18, 18], [315, 289, 18, 18], [327, 336, 18, 18], [351, 322, 18, 18],
        [367, 297, 18, 18], [317, 348, 18, 18], [356, 321, 18, 18], [409, 264, 18, 18],
        [329, 343, 18, 18]
    ]
    roisi2f = [
        [142, 369, 18, 18], [330, 296, 18, 18], [313, 338, 18, 18], [392, 332, 18, 18],
        [389, 286, 18, 18], [313, 338, 18, 18], [325, 363, 18, 18], [407, 272, 18, 18],
        [379, 326, 18, 18]
    ]
    
    #iviolin short
    slicesi2 = ["prediction_0000.dcm","prediction_0002.dcm", "prediction_0004.dcm", "prediction_0006.dcm", ]
    originalsi2 = ["iviolin_abdo_pat1_target.npy","iviolin_thor_acc_pat1_target.npy", "iviolin_thor_acc_pat3_target.npy", "iviolin_thor_nacc_pat2_target.npy"]
    roisi1 = [[158, 382, 18, 18],[327, 336, 18, 18], [367, 297, 18, 18], [356, 321 , 18, 18]] 
    roisi2 = [[142, 369, 18, 18],[313, 338, 18, 18], [389, 286, 18, 18], [325, 363, 18, 18]]

    #models = ['REDCNN', 'EDCNN', 'Uformer','LIT','WIT']
    models = ['REDCNN', 'EDCNN', 'Uformer','LIT']
    noise_levels=[20000,10000]
    #noise_levels = [40000]


    #show_interactive_plot(reconstructed_img, original_img, diff_img, args)
    #plot_comparison(slices, originals, models, args.images_dir, args)
    
    allfiles = len(glob.glob(os.path.join(args.images_dir, 'groundtruth', '*')))
    uid = datetime.now().strftime("%M%S")
    
    for i in range(len(originalsi)):
        slice = f"prediction_{i:04d}.dcm"
        original = f"13094367_I{i:05d}_target.npy"
        #plot_comparison_noises(slicesi[i], originalsi[i], models, noise_levels, args.images_dir, args=args, uid = uid, roi_print=False, diff=True)
        #plot_comparison_noises(slicesi2[i], originalsi2[i], models, noise_levels, args.images_dir, roi1=roisi1[i], roi2=roisi2[i], args=args, uid = uid, roi_print=True, diff = False)
        #plot_comparison_noises(slicesi[i], originalsi[i], models, noise_levels, args.images_dir, args=args, uid = uid, roi_print=False, diff = False)
        #visualize_differences_with_threshold(original, slice, models, noise_levels, args, threshold_value=40)
        
    
    
    #for i in range(len(slicesi2)):
    #    plot_comparison_noises(slicesi2[i], originalsi2[i], models, noise_levels, args.images_dir, args, roisi1[i], roisi2[i], roi_print=True)
        
        #plot_comparison_noises(slicesi[0], originalsi[0], models, noise_levels, args.images_dir, args, roisi1[0], roisi2[0], roi_print=True)
        #visualize_differences_with_threshold(originals2[i], slices2[i],  models, noise_levels, args, threshold_value=40)
        
    plot_comparison(slices, originals, models, args.images_dir, args)

    
    #plot_noise_simulation(slices, originals, noise_levels, args.images_dir, args)
 
    #visualize_differences_with_threshold(original_img, reconstructed_img, )

    #plt.show()



if __name__ == '__main__':
      main()