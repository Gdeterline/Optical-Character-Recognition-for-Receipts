import os, sys
import numpy as np
import cv2
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.utils import lighten_binarize_grayscale_image, compute_skew_angle_robust, rotate_image, resize_image, compute_average_image_size, compute_nearest_32_multiple, segment_image_gmm, segment_image_kmeans, extract_receipt_cluster_central, crop_to_receipt, check_if_second_cropping_needed, perform_second_cropping

def images_preprocessing_pipeline(raw_images_dir: str = "data/images", data_subset: str = "dev_", segmentation_method: str = 'gmm', n_clusters: int = 2, second_cropping_threshold: float = 0.85, output_images_path: str = None, output_meta_file: str = None, verbose: bool = False):
    """
    Complete preprocessing pipeline for receipt images.
    The steps include:
    - Resizing the images to a standard width and height, based on the median dimensions of the dataset. Aspect ratio is preserved.
    - Segmenting the receipt from the background using GMM-based segmentation (or optionally K-Means).
    - Cropping the segmented receipt with a mask, based on keeping the most central cluster.
    - Assessing if a second cropping is necessary (based on area ratio) and performing it if needed.
    - Converting the cropped image to grayscale.
    - Lightening and binarizing the grayscale image using morphological operations and Otsu's thresholding, to remove noise and wrinkles,
        resulting in a clean binary image suitable for OCR.
    - Computing the skew angle of the binarized image and deskewing it.
    
    Parameters
    ----------
        raw_images_dir (str): 
            Directory containing the raw receipt images.
        data_subset (str): 
            Subset of data to process, either 'dev_' or 'test_'.
        segmentation_method (str): 
            Method for segmentation: 'gmm' for Gaussian Mixture Model, 'kmeans' for K-Means clustering.
        n_clusters (int): 
            Number of clusters to use for segmentation.
        second_cropping_threshold (float): 
            Threshold for deciding if a second cropping is needed based on area ratio.
        output_images_path (str, optional): 
            Directory to save the preprocessed images. If None, images are not saved, only returned.
        output_meta_file (str, optional): 
            File path to save the transformation metadata of each file as a JSON file. If None, metadata is not saved.
        verbose (bool): 
            If True, prints progress and debug information.
            
    Returns
    -------
        preprocessed_images (list of np.ndarray):
            List of preprocessed images, theoretically ready for OCR.
        transformation_meta_files (dict):
            Dictionary containing metadata about the transformations applied to each image.
    """
    preprocessed_images = []
    transformation_meta_files = {}
    
    # Ensure data_subset is either 'dev_' or 'test_'
    if data_subset not in ['dev_', 'test_']:
        raise ValueError("data_subset must be either 'dev_' or 'test_'")
    
    if verbose:
        print(f"Starting preprocessing pipeline for {data_subset} images in {raw_images_dir}...")
    # List all image files in the directory (theoretically there are only png files, but just in case)
    image_files = [f for f in os.listdir(raw_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')) and f.lower().startswith(data_subset)]
    
    if verbose:
        print(f"Found {len(image_files)} images to process. Starting processing...")
    
    for index, image_file in enumerate(image_files):
        if verbose:
            print(f"Processing image: {image_file} ({index + 1}/{len(image_files)})")
        
        
            image_path = os.path.join(raw_images_dir, image_file)
    
            # Creating the metadata dictionary that keeps track of all transformations (to later apply the same to the bboxes' coordiantes)
            meta = {
                'orig_dim': (0, 0),
                'target_dim': (0, 0),
                'scale_1': (1.0, 1.0),      # First resize (New/Old)
                'crop_1': (0, 0),           # First crop offset (x, y)
                'scale_2': (1.0, 1.0),      # Resize after first crop
                'crop_2': (0, 0),           # Second crop offset (x, y)
                'scale_3': (1.0, 1.0),      # Resize after second crop
                'deskew_angle': 0.0         # Final angle used for rotation
            }
            
            # Step 1: Resize the image
            avg_width, avg_height = compute_average_image_size(raw_images_dir)
            target_w = compute_nearest_32_multiple(avg_width)
            target_h = compute_nearest_32_multiple(avg_height)
            
            resized_image, (orig_w, orig_h) = resize_image(image_path, output_path=None, new_width=target_w, new_height=target_h)
            
            # Update meta dict with original dimensions and scale factors
            meta['orig_dim'] = (orig_w, orig_h)
            meta['target_dim'] = (target_w, target_h)
            meta['scale_1'] = (target_w / orig_w, target_h / orig_h)
            
            # Step 2: Segment the image
            if segmentation_method == 'kmeans':
                segmented_image = segment_image_kmeans(resized_image, n_clusters=n_clusters)
            else:
                segmented_image = segment_image_gmm(resized_image, n_components=n_clusters)
            
            # Step 3: Extract the receipt cluster
            receipt_mask = extract_receipt_cluster_central(segmented_image)
            
            # Step 4: Crop the image to the receipt area
            cropped_image, (c1_x, c1_y, c1_w, c1_h) = crop_to_receipt(resized_image, receipt_mask)
            # Update the dict 
            meta['crop_1'] = (c1_x, c1_y)
            
            # Step 5: Resize cropped image back to original average size
            cropped_image_resized = cv2.resize(cropped_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            # Update the dict 
            meta['scale_2'] = (target_w / c1_w, target_h / c1_h)
            
            # First "smart" cropping is done. Image channels are BGR.
            
            if verbose:
                print(f"    First cropping done for image {image_file}.")
            
            # Step 6: Check if second cropping is needed
            needs_second = check_if_second_cropping_needed(cropped_image_resized, threshold=second_cropping_threshold)
            
            # Initialize final_image with the result of the first crop/resize
            final_image = cropped_image_resized 
            
            if needs_second:
                if verbose:
                    print(f"    Second cropping needed for {image_file}.")
                    
                final_image, (c2_x, c2_y, c2_w, c2_h) = perform_second_cropping(
                    cropped_image_resized, 
                    raw_images_dir, 
                    segmentation_method='gmm', 
                    n_clusters=3
                )
                
                meta['crop_2'] = (c2_x, c2_y)
                meta['scale_3'] = (target_w / c2_w, target_h / c2_h)
                
            # Step 7: Grayscale, Lighten, Binarize
            gray_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2GRAY)
                
            _, _, _, clean_binary_image = lighten_binarize_grayscale_image(gray_image, output_path=None)
                
            # Step 8: Compute skew angle and deskew
            skew_angle = compute_skew_angle_robust(clean_binary_image)
            
            rotation_angle = skew_angle
            
            # Given the text detection step later is more robust to slight over-correction than under-correction
            # we add a small offset (0.5 degrees seemed to be sufficient) when the skew angle is very small. Otherwise, we deskew normally.
            if -0.1 < skew_angle < 0.1:
                rotation_angle += 0.5
                
            deskewed_image = rotate_image(clean_binary_image, rotation_angle)
                
            # Update meta dict with deskew angle
            meta['deskew_angle'] = rotation_angle
            
            preprocessed_images.append(deskewed_image)
            transformation_meta_files[image_file] = meta            
            
    if verbose:
        print("Preprocessing pipeline completed.")
    
    if output_images_path:
        if not os.path.exists(output_images_path):
            os.makedirs(output_images_path)
        for filename, preprocessed_image in zip(image_files, preprocessed_images):
            save_path = os.path.join(output_images_path, filename)
            cv2.imwrite(save_path, preprocessed_image)
        if verbose:
            print(f"Preprocessed images saved to {output_images_path}")
        
    if output_meta_file:
        with open(output_meta_file, 'w') as f:
            json.dump(transformation_meta_files, f, indent=4)
        if verbose:
            print(f"Transformation metadata saved to {output_meta_file}")
        
    return preprocessed_images, transformation_meta_files
        
        
if __name__ == "__main__":
    # Preprocessing pipeline execution for the dev set
    preprocessed_images, transformation_meta_files = images_preprocessing_pipeline(
        raw_images_dir="data/images",
        data_subset="dev_",
        segmentation_method='gmm',
        n_clusters=2,
        second_cropping_threshold=0.85,
        output_images_path="data/preprocessed_images",
        output_meta_file="data/coordinates_transformation_meta.json",
        verbose=True
    )