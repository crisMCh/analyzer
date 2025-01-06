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

def trunc(mat, args):
        mat[mat <= args.trunc_min] = args.trunc_min
        mat[mat >= args.trunc_max] = args.trunc_max
        return mat

def denormalize_(image, args):
        image = image * (args.norm_range_max - args.norm_range_min) + args.norm_range_min
        return image

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

def plot_noise_simulation(slices, originals, noise_levels, images_dir, args):
    """
    Plot all the noise levels for multiple slices, including the original images, in a single figure.

    Args:
        slices (list): List of file names for the slices.
        originals (list): List of corresponding ground truth file names.
        noise_levels (list): List of noise levels to simulate.
        images_dir (str): Path to the folder containing ground truth and noisy files.
        args (Namespace): Contains parameters such as trunc_min, trunc_max, etc.
    """
    n_slices = len(slices)
    n_levels = len(noise_levels)

    fig, axes = plt.subplots(n_slices, n_levels + 1, figsize=(3 * (n_levels + 1), 3 * n_slices))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    for row, (slice_file, original_file) in enumerate(zip(slices, originals)):
        # Load ground truth
        groundtruth_path = os.path.join(images_dir, 'groundtruth', original_file)
        original_img = denormalize_(np.load(groundtruth_path), args)

        # Plot ground truth
        axes[row, 0].imshow(original_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        axes[row, 0].set_title('NDCT' if row == 0 else '', fontsize=12)
        axes[row, 0].axis('off')

        for col, noise_level in enumerate(noise_levels):
            # Load noisy input
            noisy_path = os.path.join(images_dir, "input", str(noise_level), original_file)
            noisy_img = denormalize_(np.load(noisy_path), args)

            # Plot noisy input
            axes[row, col + 1].imshow(noisy_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
            if row == 0:
                if noise_level == 20000:
                    axes[row, col + 1].set_title(f'Low noise', fontsize=12)
                elif noise_level == 10000:
                    axes[row, col + 1].set_title(f'Medium noise', fontsize=12)
                elif noise_level == 5000:
                    axes[row, col + 1].set_title(f'High noise', fontsize=12)
                else:
                    TypeError("No valid noise level!")
            axes[row, col + 1].axis('off')

    plt.show()
    plt.close(fig)


def plot_comparison_noises(slice_file, original_file, models, noise_levels, images_dir, args):
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
    """
    n_models = len(models)
    n_levels = len(noise_levels)

    fig, axes = plt.subplots(n_levels, n_models + 2, figsize=(3 * (n_models + 2), 3 * n_levels))
    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    # Load ground truth
    groundtruth_path = os.path.join(images_dir, 'groundtruth', original_file)
    original_img = denormalize_(np.load(groundtruth_path), args)

    for i, noise_level in enumerate(noise_levels):
        # Load noisy input
        noisy_path = os.path.join(images_dir, "input", str(noise_level), original_file)
        noisy_img = denormalize_(np.load(noisy_path), args)
        save_path = args.output_figs_path

        # Plot ground truth and noisy input
        axes[i, 0].imshow(original_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        if i == 0:
            axes[i, 0].set_title(f'NDCT', fontsize=12)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(noisy_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
        if i == 0:
            axes[i, 1].set_title(f'LDCT', fontsize=12)
        axes[i, 1].axis('off')

        # Add noise level as a label to the left of each row (before first column)
        #axes[i, 0].annotate(f'{noise_level}', xy=(-0.15, 0.5), xycoords='axes fraction', 
        #                   size=12, ha='right', va='center', rotation=90)

        if noise_level == 20000:
            axes[i, 0].annotate(f'Low noise', xy=(-0.15, 0.5), xycoords='axes fraction', size=12, ha='right', va='center', rotation=90)
                            
        elif noise_level == 10000:
            axes[i, 0].annotate(f'Medium noise', xy=(-0.15, 0.5), xycoords='axes fraction', size=12, ha='right', va='center', rotation=90)

        elif noise_level == 5000:
            axes[i, 0].annotate(f'High noise', xy=(-0.15, 0.5), xycoords='axes fraction', size=12, ha='right', va='center', rotation=90)
        else:
            TypeError("No valid noise level!")

        for j, model in enumerate(models):
            dicom_folder = dicom_path(args.predictions_path, model, noise_level)
            prediction_path = os.path.join(dicom_folder, slice_file)

            # Load the model prediction
            reconstructed_img = pydicom.dcmread(prediction_path).pixel_array

            # Plot model prediction
            axes[i, j + 2].imshow(reconstructed_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
            if i == 0:
                axes[i, j + 2].set_title(f'{model}', fontsize=12)
            axes[i, j + 2].axis('off')

            # Save each image
            #save_name = f"{os.path.splitext(slice_file)[0]}_{model}_{noise_level}.png"
            save_name = f"{os.path.splitext(slice_file)[0]}.png"
            save_full_path = os.path.join(save_path, save_name)
            fig.savefig(save_full_path, bbox_inches='tight')
    print(f"Plot saved at: {save_full_path}")
        

    #plt.show()
    plt.close()

                

def main():
    parser = argparse.ArgumentParser(description="Export and print")

    #window
    parser.add_argument('--trunc_min', type=float, default=-1000)#-160, -1000lung
    parser.add_argument('--trunc_max', type=float, default=400)#240 #400lung, -24
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=1000.0)
    parser.add_argument('--diff_threshold', type=float, default=50)


    #paths
    parser.add_argument("--predictions_path", type=str,  default='./predictions/iviolin', help="Path to ")
    parser.add_argument("--output_figs_path", type=str,  default='./figs/iviolin', help="Path to save the plots.")
    parser.add_argument('--images_dir', type=str, default='G:/Cristina/Thesis/Models/Uformer/dataset/lung/testi', help='Directory of test data, used to retireve full dose and low doses')#testi for iviolin, test for own datset
    
   #---------------------------- need to rename files here ^^^^^^^^ done

    #Model
    parser.add_argument("--model", type=str,  default='UFormer', help="Model Name") #REDCNN,EDCNN, UFormer
    parser.add_argument("--noise_level", type=int,  default=20000, help="")

    # Parse arguments
    args = parser.parse_args()
    print('test')


    

    #slices = ["prediction_0056.dcm","prediction_0091.dcm", "prediction_0123.dcm"]
    slices = ["prediction_0056.dcm","prediction_0024.dcm","prediction_0091.dcm", "prediction_0123.dcm"]
    #originals = ["13094367_I00056_target.npy","13094367_I00091_target.npy", "13094367_I00123_target.npy"]
    originals = ["13094367_I00056_target.npy","13094367_I00024_target.npy","13094367_I00091_target.npy", "13094367_I00123_target.npy"]

    #iviolin
    slicesi = ["prediction_0000.dcm","prediction_0001.dcm","prediction_0002.dcm", "prediction_0003.dcm", "prediction_0004.dcm", "prediction_0005.dcm", "prediction_0006.dcm", "prediction_0007.dcm", "prediction_0008.dcm"]
    originalsi = ["iviolin_abdo_pat1_target.npy","iviolin_abdo_pat2_target.npy","iviolin_thor_acc_pat1_target.npy", "iviolin_thor_acc_pat2_target.npy", "iviolin_thor_acc_pat3_target.npy", "iviolin_thor_acc_pat4_target.npy", "iviolin_thor_nacc_pat2_target.npy", "iviolin_thor_nacc_pat3_target.npy", "iviolin_thor_nacc_pat4_target.npy"]
    
    slicesi2 = ["prediction_0000.dcm","prediction_0002.dcm", "prediction_0004.dcm", "prediction_0006.dcm", ]
    originalsi2 = ["iviolin_abdo_pat1_target.npy","iviolin_thor_acc_pat1_target.npy", "iviolin_thor_acc_pat3_target.npy", "iviolin_thor_nacc_pat2_target.npy"]


    models = ['REDCNN', 'EDCNN', 'Uformer','DUGAN']
    noise_levels=[20000,10000,5000]


    #show_interactive_plot(reconstructed_img, original_img, diff_img, args)
    #plot_comparison(slices, originals, models, args.images_dir, args)
    #for i in range(len(slicesi)):
        #plot_comparison_noises(slicesi[i], originalsi[i], models, noise_levels, args.images_dir, args)
    
    plot_comparison(slicesi2, originalsi2, models, args.images_dir, args)

    #plot_noise_simulation(slices, originals, noise_levels, args.images_dir, args)


    #visualize_differences_with_threshold(original_img, reconstructed_img, )

    plt.show()
    





if __name__ == '__main__':
      main()