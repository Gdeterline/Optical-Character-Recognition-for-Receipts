import os, sys
import numpy as np
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.utils import cropping_pipeline, lighten_binarize_grayscale_image, compute_skew_angle_robust, rotate_image

def preprocessing_pipeline(raw_images_dir: str = "data/images", data_subset: str = "dev_", segmentation_method: str = 'gmm', n_clusters: int = 2, second_cropping_threshold: float = 0.85, output_path: str = None, verbose: bool = False) -> list:
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
        output_path (str, optional): 
            Directory to save the preprocessed images. If None, images are not saved, only returned.
        verbose (bool): 
            If True, prints progress and debug information.
            
    Returns
    -------
        preprocessed_images (list of np.ndarray):
            List of preprocessed images, theoretically ready for OCR.
    """
    preprocessed_images = []
    
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
        # Step 1: Cropping pipeline
        cropped_image, needs_second_cropping = cropping_pipeline(raw_images_dir, image_file, segmentation_method=segmentation_method, n_clusters=n_clusters, second_cropping_threshold=second_cropping_threshold, verbose=verbose)
        
        # Step 2: Convert the cropped image to grayscale
        gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        
        # Step 3: Lighten and binarize the grayscale image
        _, _, _, clean_binary_image = lighten_binarize_grayscale_image(gray_image, output_path=None)
        
        # Step 4: Compute skew angle and deskew the image
        skew_angle = compute_skew_angle_robust(clean_binary_image)
        deskewed_image = rotate_image(clean_binary_image, skew_angle)
        
        preprocessed_images.append(deskewed_image)
    
    if verbose:
        print("Preprocessing pipeline completed.")
    
    if output_path:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        for filename, preprocessed_image in zip(image_files, preprocessed_images):
            save_path = os.path.join(output_path, filename)
            cv2.imwrite(save_path, preprocessed_image)
        if verbose:
            print(f"Preprocessed images saved to {output_path}")
            
    return preprocessed_images

if __name__ == "__main__":
    # Example usage
    preprocessed_images = preprocessing_pipeline(
        raw_images_dir="data/images_subset",
        data_subset="dev_",
        segmentation_method='gmm',
        n_clusters=2,
        second_cropping_threshold=0.85,
        output_path="data/preprocessed_images",
        verbose=True
    )