import os, sys
import pickle
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.utils import transform_point

def bboxes_preprocessing_pipeline(metadata_filepath: str = "../data/metadata.pkl", transformations_meta_filepath: str = "data/coordinates_transformation_meta.json", output_filepath: str = "data/preprocessed_bboxes.json", verbose: bool = False):
    """
    Preprocess bounding boxes according to the transformations applied to the images during preprocessing.
    This function reads the original bounding boxes and the transformation metadata, applies the transformations
    to each bounding box, returns and optionally saves the updated bounding boxes to a new json file.
    
    Parameters
    ----------
        metadata_filepath (str): 
            Path to the pickle file containing the original images metadata, including bounding boxes' coordinates.
        transformations_meta_filepath (str): 
            Path to the json file containing the transformations metadata for each image.
        output_filepath (str): 
            Path to save the preprocessed bounding boxes as a json file.
        verbose (bool): 
            If True, prints progress and debug information.
            
    Returns
    -------
        preprocessed_bboxes (dict):
            Dictionary containing the preprocessed bounding boxes for each image.
    """
    bboxes_dict = {}
    
    # Load original metadata with bounding boxes
    if verbose:
        print(f"Loading original metadata from {metadata_filepath}...")
    with open(metadata_filepath, 'rb') as f:
        metadata = pickle.load(f)
        
    file_names = metadata['file_name']
    
    # Restrict filenames and bboxes to only those where the filename starts with 'dev_' 
    file_names = file_names[file_names.str.startswith('dev_')]
    
    bboxes_list = metadata['bboxes']
    bboxes_list = bboxes_list[file_names.index]
    
    if verbose:
        print(f"Loaded metadata for {len(file_names)} images.")
    
    # Load transformations metadata
    if verbose:
        print(f"Loading transformations metadata from {transformations_meta_filepath}...")
        
    with open(transformations_meta_filepath, 'r') as f:
        transformations_meta = json.load(f)
        
    if verbose:
        print(f"Loaded transformations metadata for {len(transformations_meta)} images.")
        
    # Process each image and its bounding boxes
    for idx, file_name in enumerate(file_names):
        if verbose:
            print(f"Processing bounding boxes for image: {file_name} ({idx + 1}/{len(file_names)})")
        
        index = file_names.values.tolist().index(file_name)
        original_boxes = bboxes_list[index]
        
        if file_name not in bboxes_dict:
                bboxes_dict[file_name] = []
        
        for bbox in original_boxes:
                
            pts = {}
            for i in range(1, 5):
                nx, ny = transform_point(bbox[f'x{i}'], bbox[f'y{i}'], transformations_meta[file_name])
                
                # Store the transformed points in the same way as the original bbox format to keep consistency
                pts[f'x{i}'] = nx
                pts[f'y{i}'] = ny

            bboxes_dict[file_name].append(pts)  # Append the transformed bbox to the list for this image. The format for multiple bboxes is: {file_name: [ {bbox1}, {bbox2}, ... ]}
            
    # Save the preprocessed bounding boxes to a json file
    if output_filepath is not None:
        if verbose:
            print(f"Saving preprocessed bounding boxes to {output_filepath}...")
        with open(output_filepath, 'w') as f:
            json.dump(bboxes_dict, f, indent=4)
        if verbose:
            print("Preprocessed bounding boxes saved successfully.")
            
    return bboxes_dict

if __name__ == "__main__":
    # Preprocessing the bounding bounding boxes for the dev set
    preprocessed_bboxes = bboxes_preprocessing_pipeline(
        metadata_filepath="data/metadata.pkl",
        transformations_meta_filepath="data/coordinates_transformation_meta.json",
        output_filepath="data/preprocessed_bboxes.json",
        verbose=True
    )