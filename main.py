import pydicom
import argparse
import pandas as pd
import numpy as np
from pydicom.dataset import Dataset
import os
import csv
import matplotlib.pyplot as plt
import ast

def extract_dicom_metadata(original_dicom_path, output_metadata_path):
    """
    Extracts metadata from a DICOM file and saves it to a CSV file.

    Args:
        original_dicom_path (str): Path to the original DICOM file.
        output_metadata_path (str): Path to save the extracted metadata as a CSV file.
    """
    # Read the DICOM file
    dicom_data = pydicom.dcmread(original_dicom_path)
    
    # Extract metadata
    metadata = []
    for tag in dicom_data.dir():
        value = dicom_data.get(tag, None)
        if isinstance(value, pydicom.multival.MultiValue):  # Handle MultiValue fields
            value = str(value)  # Convert it to a string representation
        metadata.append((tag, value))
    
    # Convert metadata to DataFrame
    metadata_df = pd.DataFrame(metadata, columns=["Tag", "Value"])
    
    # Save to CSV
    metadata_df.to_csv(output_metadata_path, index=False)
    
    print(f"Metadata saved to {output_metadata_path}")

def load_dicom_metadata_from_csv(csv_path):
    """
    Loads DICOM metadata from a CSV file and returns it as a dictionary.
    
    Args:
        csv_path (str): Path to the CSV file containing the DICOM metadata.
        
    Returns:
        dict: Metadata dictionary where keys are DICOM tags and values are their respective values.
    """
    csv.field_size_limit(2000000)  # Set a higher limit (2MB, adjust as needed)
    
    dicom_metadata = {}
    with open(csv_path, mode='r') as file:
        reader = csv.reader(file)
        for row in reader:
            if len(row) == 2:
                tag, value = row
                dicom_metadata[tag] = value
    return dicom_metadata

def dicom_to_hu(dicom_file, npy):
    """
    Converts a DICOM file to a Hounsfield Units (HU) numpy array.
    """
    # Load the DICOM file
    dicom = pydicom.dcmread(dicom_file)

    # Extract the raw pixel data
    image = npy

    # Get the Rescale Slope and Intercept for HU conversion
    rescale_slope = getattr(dicom, "RescaleSlope", 1)
    rescale_intercept = getattr(dicom, "RescaleIntercept", 0)

    # Convert pixel values to Hounsfield Units (HU)
    hu_image = image * rescale_slope + rescale_intercept


    return hu_image

def denormalize_hu(normalized_image, min_hu=-1000, max_hu=1000):
    # Ensure the input is a NumPy array
    if not isinstance(normalized_image, np.ndarray):
        raise ValueError("Input must be a NumPy array")
    
    # Check that the normalized values are within [0, 1]
    normalized_image = np.clip(normalized_image, 0, 1)
        
    # Denormalize the image to HU
    hu_image = normalized_image * (max_hu - min_hu) + min_hu
    
    return hu_image

def save_npy_as_dicom(npy_file, original_dicom_path, output_folder):
    """
    Converts a .npy file to a DICOM file and adds the metadata.
    
    Args:
        npy_file (str): Path to the .npy file.
        original_dicom_path (dcm): original DICOM.
        output_folder (str): Folder where the DICOM files will be saved.
    """
    # Load the numpy array
    np_array = np.load(npy_file)
    if np_array.min() >= 0 and np_array.max() <= 1:
        np_array = denormalize_hu(np_array)
    
    np_array = denormalize_hu(np_array)
    # Ensure the numpy array is of type that DICOM expects (usually unsigned int16 or float32)
    
    np_array = np_array.astype(np.uint16)  # Adjust depending on your pixel data type
    
    # Plot the numpy array in grayscale
    #plt.imshow(np_array, cmap='gray')
    #plt.title(f"Preview of {npy_file}")
    #plt.axis('off')  # Hide the axis
    #plt.show()
    # Create a new DICOM dataset
    #dicom_file = Dataset()

    # Set pixel data (do this separately, do not overwrite later)
    original_dicom = pydicom.dcmread(original_dicom_path)
    #original_dicom.PixelData = dicom_to_hu(original_dicom_path, np_array).tobytes()

    original_dicom.PixelData = np_array.tobytes()



    # Save the DICOM file
    output_file = os.path.join(output_folder, os.path.basename(npy_file).replace('.npy', '.dcm'))
    #dicom_file.save_as(output_file)
    original_dicom.save_as(output_file)
    
    print(f"Saved DICOM file: {output_file}")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Extract metadata from a DICOM file and save it as JSON.")

    #parameters for extracting metadata
    #parser.add_argument("--original_dicom_path", type=str,  default='./original_dicom/abdo_pat1', help="Path to the original DICOM file.")
    parser.add_argument("--original_dicom_path", type=str,  default='./original_dicom/I0000002', help="Path to the original DICOM file.")
    parser.add_argument("--output_metadata_path", type=str,  default='./original_dicom/metadata.csv', help="Path to save the extracted metadata as a JSON file.")
    
    #
    parser.add_argument("--predictions_folder", type=str,  default='./predictions/iviolin/original', help="Path to the prediction folder.")
    #parser.add_argument("--predictions_folder", type=str,  default='./predictions/40000', help="Path to the prediction folder.")
 

    
    # Parse arguments
    args = parser.parse_args()
    
    # Extract metadata
    #dicom_metadata = load_dicom_metadata_from_csv(args.output_metadata_path)


    
    # Convert each .npy file in the predictions folder to a DICOM file
    
    predictions_npy = os.path.join(args.predictions_folder, 'npy')
    output_folder = os.path.join(args.predictions_folder, 'dcm')
         # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    

    for npy_file in os.listdir(predictions_npy):
        if npy_file.endswith(".npy"):
            npy_file_path = os.path.join(predictions_npy, npy_file)
            

            # Load the numpy array
            npy_array = np.load(npy_file_path)

            # Plot the numpy array in grayscale
            #plt.imshow(npy_array, cmap='gray')
            #plt.title(f"Preview of {npy_file}")
            #plt.axis('off')  # Hide the axis
            #plt.show()
            
            save_npy_as_dicom(npy_file_path, args.original_dicom_path, output_folder)

if __name__ == "__main__":
    main()
