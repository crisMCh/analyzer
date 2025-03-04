import pydicom
import numpy as np
from matplotlib.widgets import RectangleSelector
import os
import matplotlib.pyplot as plt


def load_dicom_and_calculate_mean(file_path):
    # Load the DICOM file
    dicom_data = pydicom.dcmread(file_path)
    
    # Extract the pixel array
    pixel_array = dicom_data.pixel_array
    
    # Calculate the mean of the pixel intensity values
    mean_intensity = np.mean(pixel_array)
    
    return mean_intensity

def plot_dicom_image(file_path):
    # Load the DICOM file
    dicom_data = pydicom.dcmread(file_path)
    
    # Extract the pixel array
    pixel_array = dicom_data.pixel_array
    
    # Plot the image
    plt.imshow(pixel_array, cmap='gray')
    plt.title('DICOM Image')
    plt.axis('off')
    plt.show()

def plot_numpy_image(image_array):
    # Plot the numpy image array
    plt.imshow(image_array, cmap='gray')
    plt.title('Numpy Image')
    plt.axis('off')
    plt.show()
   
def select_roi_and_plot(file_path_npy):
    # Load the numpy image array
    image_array = np.load(file_path_npy)
    
    # Plot the numpy image array
    fig, ax = plt.subplots()
    ax.imshow(image_array, cmap='gray')
    plt.title('Select ROI')
    
    # Function to be called when ROI is selected
    def onselect(eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        center_x = x1 + width // 2
        center_y = y1 + height // 2
        roi_info = [center_x, center_y, height, width]
        print(f"ROI center and dimensions: {roi_info}")
    
    # Create a RectangleSelector
    rect_selector = RectangleSelector(ax, onselect, interactive=True,
                                      button=[1], minspanx=5, minspany=5, spancoords='pixels')
    
    plt.show()


def rename_files_in_directory(directory_path):
    # List all files in the directory
    files = os.listdir(directory_path)
    
    # Sort the files to ensure consistent renaming
    files.sort()
    
    # Rename each file
    for i, filename in enumerate(files):
        # Construct the new filename
        new_filename = f"prediction_{i:04d}.dcm"
        
        # Get the full path of the old and new filenames
        old_file_path = os.path.join(directory_path, filename)
        new_file_path = os.path.join(directory_path, new_filename)
        
        # Rename the file
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_file_path} to {new_file_path}")

# Example usage
# rename_files_in_directory('path_to_directory')
# Example usage
if __name__ == "__main__":
    #file_path = 'predictions\WITi_e105_n40000\dcm\prediction_0001.dcm'
    file_path = 'predictions\WIT_e105_n40000\dcm\prediction_0000.dcm'
    
    #file_path_npy = r'G:\Cristina\Thesis\Models\LIT-Former\results\test_results\predictions\LIT_e105_n40000\npy\prediction_0001.npy'
    file_path_npy = r'G:\Cristina\Thesis\Models\Uformer\dataset\lung\test\groundtruth\13094367_I00118_target.npy'
    
    
    #mean_intensity_dicom = load_dicom_and_calculate_mean(file_path)
    
    #plot_dicom_image(file_path)
    #plot_numpy_image(np.load(file_path_npy))
    #print(f"Mean pixel intensity: {mean_intensity_dicom}")
    
    #select_roi_and_plot(file_path_npy)  # Select a region of interest in the numpy image
    rename_files_in_directory('predictions\LIT_e150_n10000\dcm2')  # Rename all files in the directory
    
    