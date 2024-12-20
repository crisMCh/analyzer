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

def main():
    parser = argparse.ArgumentParser(description="Export and print")

    #window
    parser.add_argument('--trunc_min', type=float, default=-1000)#-160, -1000lung
    parser.add_argument('--trunc_max', type=float, default=400)#240 #400lung
    parser.add_argument('--norm_range_min', type=float, default=-1024.0)
    parser.add_argument('--norm_range_max', type=float, default=1000.0)
    parser.add_argument('--diff_threshold', type=float, default=50)


    #paths
    parser.add_argument("--predictions_path", type=str,  default='./predictions', help="Path to ")
    parser.add_argument("--output_figs_path", type=str,  default='./figs', help="Path to save the plots.")
    parser.add_argument('--images_dir', type=str, default='G:/Cristina/Thesis/Models/Uformer/dataset/lung/test', help='Directory of test data, used to retireve full dose and low doses')
   #---------------------------- need to rename files here ^^^^^^^^

    #Model
    parser.add_argument("--model", type=str,  default='UFormer', help="Model Name") #REDCNN,EDCNN, UFormer
    parser.add_argument("--noise_level", type=int,  default=20000, help="")

    # Parse arguments
    args = parser.parse_args()
    print('test')
    dicom_folder = dicom_path(args.predictions_path,args.model,args.noise_level)

    image_path = os.path.join(dicom_folder, "prediction_0091.dcm")
    original_path = os.path.join(args.images_dir, 'groundtruth', "13094367_I00091_target.npy")
    print(image_path)
    print(original_path)

    reconstructed_img =  pydicom.dcmread(image_path).pixel_array
    original_img= denormalize_(np.load(original_path), args)
    diff_img = trunc(original_img,args) - trunc(reconstructed_img,args)
    #diff = cv2.subtract(trunc(original_img,args), trunc(image,args))
    
    # # Apply a threshold to the difference map to avoid small diferences in pixel variation 
    #diff[np.abs(diff) < args.diff_threshold] = 0 
    '''
    plt.subplot(1,3,1)
    plt.imshow(reconstructed_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
    plt.title(f'Low dose {args.noise_level}', fontsize=30)

    plt.subplot(1,3,2)
    plt.imshow(original_img, cmap=plt.cm.gray, vmin=args.trunc_min, vmax=args.trunc_max)
    plt.title('Full dose', fontsize=30)

    plt.subplot(1,3,3)
    plt.imshow(diff_img, cmap=plt.cm.gray, vmin=0, vmax=np.max(np.abs(diff_img)))
    plt.title('Difference', fontsize=30)
    #'''
    show_interactive_plot(reconstructed_img, original_img, diff_img, args)


    #visualize_differences_with_threshold(original_img, reconstructed_img, )


    plt.show()
    





if __name__ == '__main__':
      main()