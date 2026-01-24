import os, sys
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def compute_average_image_size(raw_images_dir):
    """
    Compute the average width and height of images in a directory.

    Parameters
    ----------
        raw_images_dir (str): 
            Path to the directory containing raw images.
    Returns
    -------
        avg_width (int), avg_height (int):
            A tuple containing the average width and average height of the images.
    """
    image_sizes = []
    for image_file in os.listdir(raw_images_dir):
        img = Image.open(os.path.join(raw_images_dir, image_file))
        image_sizes.append(img.size)  # (width, height)
    med_width = int(np.median([size[0] for size in image_sizes]))
    med_height = int(np.median([size[1] for size in image_sizes]))
    
    return med_width, med_height


def compute_nearest_32_multiple(x):
    """
    Compute the nearest multiple of 32 for a given value.

    Parameters
    ----------
        x (int): 
            The input value.
    Returns
    -------
        int: 
            The nearest multiple of 32.
    """
    return int(np.round(x / 32) * 32)


def resize_image(image_path: str, output_path: str, new_width: int, new_height: int, save: bool = False):
    """
    Resize an image to the specified dimensions.

    Parameters
    ----------
        image_path (str): 
            Path to the input image.
        output_path (str): 
            Path to save the resized image (if save is True).
        new_width (int): 
            New width for the resized image.
        new_height (int): 
            New height for the resized image.
        save (bool, optional): 
            Whether to save the resized image. Defaults to False.
    Returns
    -------
        PIL.Image.Image: 
            The resized image.
    """
    img = Image.open(image_path)
    img_resized = img.resize((new_width, new_height), Image.BICUBIC)
    
    if save:
        if output_path and not os.path.exists(output_path):
            os.makedirs(output_path)
        image_filename = os.path.basename(image_path)
        print(f"Image {image_filename} resized and saved to {output_path}")
        output_path = os.path.join(output_path, image_filename)
        img_resized.save(output_path)
        
    # Convert back to OpenCV format
    img_resized = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        
    return img_resized


def segment_image_kmeans(image: np.ndarray, n_clusters: int = 2) -> np.ndarray:
    """
    Segment an image using K-Means clustering.

    Parameters
    ----------
        image (np.ndarray): 
            The input image in BGR format.
        n_clusters (int, optional):
            The number of clusters for K-Means. Defaults to 2.
    Returns
    -------
        np.ndarray: 
            The segmented image.
    """
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float64(pixel_values)

    # Define criteria and apply K-Means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixel_values)

    # Convert back to the original image shape
    segmented_image = labels.reshape(image.shape[:2])

    return segmented_image


def segment_image_gmm(image: np.ndarray, n_components: int = 2) -> np.ndarray:
    """
    Segment an image using Gaussian Mixture Models (GMM).

    Parameters
    ----------
        image (np.ndarray): 
            The input image in BGR format.
        n_components (int, optional):
            The number of components for GMM. Defaults to 2.
    Returns
    -------
        np.ndarray: 
            The segmented image.
    """
    # Reshape the image to a 2D array of pixels
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float64(pixel_values)

    # Fit GMM to the pixel values
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(pixel_values)
    except Exception:
        # If default fitting fails (often due to singular covariance matrices on uniform images),
        # retry with a higher regularization value.
        print("GMM fitting failed with default parameters. Retrying with higher regularization.")
        gmm = GaussianMixture(n_components=n_components, random_state=42, reg_covar=1e-4)
        gmm.fit(pixel_values)
        
    labels = gmm.predict(pixel_values)

    # Convert back to the original image shape
    segmented_image = labels.reshape(image.shape[:2])

    return segmented_image

            
# We now have a segmented image where each pixel is labeled with its cluster index.
# We want to define a function that extract the cluster corresponding to the receipt.
def extract_receipt_cluster_largest(segmented_image: np.ndarray) -> np.ndarray:
    """
    Extract the cluster corresponding to the receipt from the segmented image based on largest area.
    This function assumes that the receipt is the largest cluster in the segmented image, which may not always be true.
    The image sample 'dev_receipt_00016.png' shows a case where this assumption fails.

    Parameters
    ----------
        segmented_image (np.ndarray): 
            The segmented image with cluster labels.
    Returns
    -------
        np.ndarray: 
            A binary mask of the receipt cluster.
    """
    # Assuming the receipt is the largest cluster
    unique, counts = np.unique(segmented_image, return_counts=True)
    largest_cluster = unique[np.argmax(counts)]

    # Create a binary mask for the receipt cluster
    receipt_mask = np.where(segmented_image == largest_cluster, 255, 0).astype(np.uint8)

    return receipt_mask


# Try extracting the receipt cluster based on the central position
def extract_receipt_cluster_central(segmented_image: np.ndarray) -> np.ndarray:
    """
    Extract the cluster corresponding to the receipt based on central position.
    This function makes the hypothesis that the receipt is located near the center of the image, which seems reasonable.

    Parameters
    ----------
        segmented_image (np.ndarray): 
            The segmented image with cluster labels.
    Returns
    -------
        np.ndarray: 
            A binary mask of the receipt cluster.
    """
    height, width = segmented_image.shape
    center_x, center_y = width // 2, height // 2
    central_cluster = segmented_image[center_y, center_x]

    # Create a binary mask for the receipt cluster
    receipt_mask = np.where(segmented_image == central_cluster, 255, 0).astype(np.uint8)

    return receipt_mask


# Crop the image to remove what is non-receipt area based on the segmented image and the mask
def crop_to_receipt(image: np.ndarray, receipt_mask: np.ndarray) -> np.ndarray:
    """
    Crop the image to the bounding box of the receipt area.
    Parameters
    ----------
        image (np.ndarray): 
            The original image.
        receipt_mask (np.ndarray):
            The binary mask of the receipt cluster.
    Returns
        np.ndarray: 
            The cropped image containing only the receipt area.
    """
    # Find contours of the receipt area
    contours, _ = cv2.findContours(receipt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return image  # No contours found, return original image

    # Get the bounding box of the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box
    cropped_image = image[y:y+h, x:x+w]

    return cropped_image


def check_if_second_cropping_needed(cropped_image: np.ndarray, threshold: float = 0.85) -> bool:
    """
    Check if a second cropping is needed.
    The function performs a segmentation on the input image, with 2 clusters.
    In case the largest cluster covers less than threshold (default 85%) of the image area, we consider that a second cropping is needed.
    The hypothesis is that if the receipt is not covering enough area, there are still non-receipt areas present.
    
    Parameters
    ----------
        cropped_image (np.ndarray): 
            The cropped image to check.
        threshold (float, optional): 
            The area coverage threshold to decide if second cropping is needed. Defaults to 0.85.
    Returns
    -------
        bool: 
            True if a second cropping is needed, False otherwise.
    """
    segmented_image = segment_image_gmm(cropped_image, n_components=2)
    unique, counts = np.unique(segmented_image, return_counts=True)
    largest_cluster_area = np.max(counts)
    total_area = cropped_image.shape[0] * cropped_image.shape[1]
    coverage_ratio = largest_cluster_area / total_area
    
    return coverage_ratio < threshold


def perform_second_cropping(cropped_image, raw_images_dir: str, segmentation_method: str = 'gmm', n_clusters: int = 3) -> np.ndarray:
    """
    Perform a second cropping on the cropped image.
    
    Parameters
    ----------
        cropped_image (Matlike): 
            The cropped image to process.
        segmentation_method (str, optional): 
            The segmentation method to use ('kmeans' or 'gmm'). Defaults to 'gmm'.
        n_clusters (int, optional):
            The number of clusters/components for segmentation. Defaults to 3, to better separate receipt from non-receipt areas.
            Assumes that the cropping with n_clusters=2 was insufficient.
            
    Returns
    -------
        np.ndarray: 
            The further cropped image containing only the receipt area.
    """
    # Step 1: Segment the cropped image
    if segmentation_method == 'kmeans':
        segmented_image = segment_image_kmeans(cropped_image, n_clusters=n_clusters)
    else:
        segmented_image = segment_image_gmm(cropped_image, n_components=n_clusters)
    
    # Step 2: Extract the receipt cluster
    receipt_mask = extract_receipt_cluster_central(segmented_image)
    
    # Step 3: Crop the image to the receipt area
    further_cropped_image = crop_to_receipt(cropped_image, receipt_mask)
    avg_width, avg_height = compute_average_image_size(raw_images_dir)
    new_width = compute_nearest_32_multiple(avg_width)
    new_height = compute_nearest_32_multiple(avg_height)
    further_cropped_image = cv2.resize(further_cropped_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # The image is now further cropped, and the channels are kept as in the original image (BGR)
    
    return further_cropped_image


def cropping_pipeline(raw_images_dir: str, image_filename: str, segmentation_method: str = 'gmm', n_clusters: int = 2, second_cropping_threshold: float = 0.85, verbose: bool = False) -> np.ndarray:
    """
    Complete cropping pipeline to resize, segment, extract receipt cluster, and crop the image, with optional second cropping if needed.

    Parameters
    ----------
        raw_images_dir (str): 
            Path to the directory containing raw images.
        image_filename (str): 
            The filename of the image to process.
        segmentation_method (str, optional): 
            The segmentation method to use ('kmeans' or 'gmm'). Defaults to 'gmm'.
        n_clusters (int, optional):
            The number of clusters/components for segmentation. Defaults to 2.
        verbose (bool, optional):
            Whether to print verbose output. Defaults to False.
    Returns
    -------

    """
    image_path = os.path.join(raw_images_dir, image_filename)
    
    # Step 1: Resize the image
    if verbose:
        print(f"Processing image: {image_filename}")
    avg_width, avg_height = compute_average_image_size(raw_images_dir)
    new_width = compute_nearest_32_multiple(avg_width)
    new_height = compute_nearest_32_multiple(avg_height)
    resized_image = resize_image(image_path, None, new_width, new_height, save=False)
    
    # Step 2: Segment the image
    if segmentation_method == 'kmeans':
        segmented_image = segment_image_kmeans(resized_image, n_clusters=n_clusters)
    else:
        segmented_image = segment_image_gmm(resized_image, n_components=n_clusters)
    
    # Step 3: Extract the receipt cluster
    receipt_mask = extract_receipt_cluster_central(segmented_image)
    
    # Step 4: Crop the image to the receipt area
    cropped_image = crop_to_receipt(resized_image, receipt_mask)
    
    # Step 5: Resize cropped image back to original average size
    cropped_image = cv2.resize(cropped_image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    
    # First "smart" cropping is done. Image channels are BGR.
    
    if verbose:
        print(f"First cropping done for image {image_filename}.")
    
    # Step 6: Check if second cropping is needed
    needs_second_cropping = check_if_second_cropping_needed(cropped_image, threshold=second_cropping_threshold)
    if needs_second_cropping:
        if verbose:
            print(f"Second cropping needed for image {image_filename}. Consider performing a second cropping step.")
        cropped_image2 = perform_second_cropping(cropped_image, raw_images_dir, segmentation_method='gmm', n_clusters=3)
        if verbose:
            print(f"Second cropping done for image {image_filename}.")
        return cropped_image2, True
        
    return cropped_image, False


############# Denoising, Binarization and Deskewing Utilities #############


def lighten_binarize_grayscale_image(mat_image, output_path=None):
    """
    Denoise and binarize a grayscale image using morphological operations (closing) and division normalization.
    
    Parameters
    ----------
        mat_image (np.ndarray):
            The input image (2D numpy array). The image is expected to be in grayscale format.
            If not, it will be converted to grayscale.
        output_path (str, optional):
            Path to save the denoised image. If None, the image is not saved. Defaults to None.
    Returns
    -------
        mat_image (np.ndarray):
            The original image. If the input was not grayscale, it is converted, then returned as grayscale.
        background_model (np.ndarray):
            The estimated background model of the image, given a kernel size of 5x5.
        result (np.ndarray):
            The denoised image after division normalization.
        binary (np.ndarray):
            The binarized version of the denoised image using Otsu's thresholding (after de-wrinkling).
    """
    # Step 0. Ensure the image is in grayscale
    if mat_image.ndim == 3:
        mat_image = cv2.cvtColor(mat_image, cv2.COLOR_BGR2GRAY) # Assuming input is BGR if not grayscale
    elif mat_image.ndim != 2:
        raise ValueError("Input image must be either grayscale or BGR format.")
    
    # Step 1. we define the kernel (structuring element)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    # Step 2. Morphological Closing
    background_model = cv2.morphologyEx(mat_image, cv2.MORPH_CLOSE, kernel)
    
    # Slight blur on background to reduce artifacts
    background_model = cv2.GaussianBlur(background_model, (5, 5), 0)

    # Step 3. Division Normalization ("De-wrinkling" step)
    result = mat_image.astype(float) / background_model.astype(float)
    
    # Scale back to 0-255 range
    result = result * 255
    result = np.clip(result, 0, 255).astype(np.uint8)

    # Step 4. Binarization
    _, binary = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 5. Save output if path is provided
    if output_path:
        cv2.imwrite(output_path, result)
    
    return mat_image, background_model, result, binary


def compute_skew_angle(binary_image):
    """
    Compute the skew angle of a binary image using contour detection and minimum area rectangles.
    
    Parameters
    ----------
        binary_image (np.ndarray): 
            The input binary image (2D numpy array).
    Returns
    -------
        float: 
            The computed skew angle in degrees. Trigonometric convention is used.
    """    
    # Step 1. Invert image (Background must be black, text white for contours)
    # 127 is mid-gray threshold - usual value to start with
    if binary_image[0, 0] > 127:
        inverted = cv2.bitwise_not(binary_image)
    else:
        inverted = binary_image

    # Step 2. Dilate to connect text into lines
    # Kernel: Wide and short. (20, 1) connects words horizontally.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    dilated = cv2.dilate(inverted, kernel, iterations=1)

    # Step 3. Find Contours of the text lines
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    angles = []
    for contour in contours:
        # Filter small noise (dots, specks)
        if cv2.contourArea(contour) < 100: 
            continue
            
        # Fit a rotated rectangle
        rect = cv2.minAreaRect(contour) 
        # rect = ((center_x, center_y), (width, height), angle)
        (center, (w, h), angle) = rect
        
        # Step 4. Correct the angle based on rectangle orientation
        # We normalize to ensure the angle represents the horizontal axis deviation.
        
        if w < h:
            # If width is less than height, the angle corresponds to the vertical side.
            # We shift it by 90 degrees to get the horizontal alignment.
            angle = angle - 90
            
        angles.append(angle)

    # Step 5. Determine the median angle (Robust to outliers)
    if len(angles) > 0:
        median_angle = np.median(angles)
    else:
        median_angle = 0.0

    return median_angle

def rotate_image(image, angle):
    """
    Rotate an image by a specified angle.
    Parameters
    ----------
        image (np.ndarray): 
            The input image to rotate.
        angle (float): 
            The angle in degrees to rotate the image. Positive values mean trigonometric rotation.
    Returns
        np.ndarray: 
            The rotated image.
    """
    # Get image center
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform the rotation
    # borderValue=(255,255,255) fills the new empty corners with white
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated


# Debug code
if __name__ == "__main__":
    pass
    
    # Test the dewrinkle, denoising, binarization and deskewing functions on a subset of images
    
    raw_images_dir = "data/_debug/"
    debug_output_dir = "data/_debug2/"
    
    if not os.path.exists(debug_output_dir):
        os.makedirs(debug_output_dir)
        
    image_filenames = os.listdir(raw_images_dir)
    
    # Consider only images a subset of 10 images that start with 'dev_'
    image_filenames = [f for f in image_filenames if f.startswith('dev_')][:10]
    
    for index, image_filename in enumerate(image_filenames):
        print(f"Processing image: {index+1}/{len(image_filenames)} - {image_filename}")
        mat_image = cv2.imread(os.path.join(raw_images_dir, image_filename), cv2.IMREAD_GRAYSCALE)
        original_image, background_model, denoised_image, binary_denoised_image = lighten_binarize_grayscale_image(mat_image)
        skew_angle = compute_skew_angle(binary_denoised_image)
        print(f"Detected Skew Angle: {skew_angle:.2f} degrees")
        deskewed_image = rotate_image(binary_denoised_image, skew_angle)
        
        # Save outputs
        cv2.imwrite(os.path.join(debug_output_dir, f"{image_filename}"), deskewed_image)
    
    
    ##################################################################################################################################
    
    # # Perform croppings on all images and save them in data/_debug, to conduct experiments on them (PCA, etc.)    
    # raw_images_dir = "data/images/"
    # debug_output_dir = "data/_debug/"
    # if not os.path.exists(debug_output_dir):
    #     os.makedirs(debug_output_dir)
    
    # image_filenames = os.listdir(raw_images_dir)
    
    # # Consider only images that start with 'dev_'
    # image_filenames = [f for f in image_filenames if f.startswith('dev_')]
    
    # for index, image_filename in enumerate(image_filenames):
    #     print(f"Processing image: {index+1}/{len(image_filenames)} - {image_filename}")
    #     if image_filename not in os.listdir(debug_output_dir):
    #         cropped_image, needs_second_cropping = cropping_pipeline(raw_images_dir, image_filename, segmentation_method='gmm', n_clusters=2, second_cropping_threshold=0.85, verbose=False)
    #         if needs_second_cropping:
    #             print(f"Second cropping was needed for image {image_filename}.")
    #         debug_image_path = os.path.join(debug_output_dir, image_filename)
    #         cv2.imwrite(debug_image_path, cropped_image)
            
    
    ##################################################################################################################################
    
    # # test the check_if_second_cropping_needed function on two images: one that needs second cropping and one that does not
    # raw_images_dir = "data/images/"
    # image_filenames = ["dev_receipt_00091.png", "dev_receipt_00016.png"]
    
    # for image_filename in image_filenames:
    #     cropped_image, needs_second_cropping = cropping_pipeline(raw_images_dir, image_filename, segmentation_method='gmm', n_clusters=2, second_cropping_threshold=0.85, verbose=False)
    #     print(f"Does image {image_filename} need second cropping? Answer: {needs_second_cropping}")    
        
    #     plt.figure(figsize=(12, 6))   
    #     plt.subplot(1, 2, 1)
    #     original_image = cv2.imread(os.path.join(raw_images_dir, image_filename))
    #     plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    #     plt.title("Original Image")
    #     plt.axis('off')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    #     plt.title("Cropped Image")
    #     plt.axis('off')
    #     plt.suptitle(f"Cropping Pipeline Results for {image_filename} - Second Cropping: {needs_second_cropping}", fontsize=16)
    #     plt.tight_layout()
    #     #plt.savefig(f"reports/figures/cropping_pipeline_{image_filename.replace('.png','')}.png")
    #     plt.show()
        
    ##################################################################################################################################
        
    # Display two examples of the full cropping pipeline for report/slides
    # raw_images_dir = "data/images/"
    # image_filenames = ["dev_receipt_00080.png", "dev_receipt_00016.png"]
    
    # # Plot the original images, segmented images and cropped images, side by side for each image
    # for image_filename in image_filenames:
    #     cropped_image = cropping_pipeline(raw_images_dir, image_filename, segmentation_method='gmm', n_clusters=2)
        
    #     print(f"Cropped image shape for {image_filename}: {cropped_image.shape}")
    #     # Visualize the original, segmented and cropped images in three side-by-side plots
    #     plt.figure(figsize=(12, 8))
    #     original_image = cv2.imread(os.path.join(raw_images_dir, image_filename))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    #     plt.title(f"Original Image - Size: {original_image.shape[1]}x{original_image.shape[0]}")
    #     plt.axis('off')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    #     plt.title(f"Cropped Image - Size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    #     plt.axis('off')
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    #     plt.title(f"Cropped Image - Size: {cropped_image.shape[1]}x{cropped_image.shape[0]}")
    #     plt.axis('off')
    #     plt.suptitle(f"Cropping Pipeline Results for {image_filename}", fontsize=16)
    #     plt.savefig(f"reports/figures/cropping_pipeline_{image_filename.replace('.png','')}.png")
    #     plt.show()
    
    ##################################################################################################################################    
    
    ##################################################################################################################################
    
    # First tryout of each function individually
    
    # raw_images_dir = "data/images/"
    # avg_width, avg_height = compute_average_image_size(raw_images_dir)
    # new_width = compute_nearest_32_multiple(avg_width)
    # new_height = compute_nearest_32_multiple(avg_height)
    # print(f"Average image size: {avg_width}x{avg_height}")
    # print(f"Resized image size (nearest multiple of 32): {new_width}x{new_height}")
    
    # # Example of resizing an image
    # example_image_path = os.path.join(raw_images_dir, "dev_receipt_00091.png")
    # resized_image = resize_image(example_image_path, None, new_width, new_height, save=False)
    # print(f"Resized image shape: {resized_image.shape}")
    
    # # # Segment the resized image using K-Means
    # # segmented_image_kmeans = segment_image_kmeans(resized_image, n_clusters=2)
    # # print(f"Segmented image shape: {segmented_image_kmeans.shape}")
    
    # # Segment the resized image using GMM
    # segmented_image_gmm = segment_image_gmm(resized_image, n_components=2)
    # print(f"Segmented image shape (GMM): {segmented_image_gmm.shape}")
    
    # # Extract the receipt cluster
    # receipt_mask = extract_receipt_cluster_central(segmented_image_gmm)
    # print(f"Receipt mask shape: {receipt_mask.shape}")
    
    # # Crop the resized image to the receipt area
    # cropped_image = crop_to_receipt(resized_image, receipt_mask)
    # print(f"Cropped image shape: {cropped_image.shape}")
    
    # # Check if second crop is needed
    # needs_second_cropping = check_if_second_cropping_needed(cropped_image, threshold=0.85)
    # print(f"Does the cropped image need second cropping? {needs_second_cropping}")
    
    # if needs_second_cropping:
        
    #     # segment the cropped image again using GMM with 3 components
    #     segment_image_gmm2 = segment_image_gmm(cropped_image, n_components=3)
        
    #     cropped_image = perform_second_cropping(cropped_image, raw_images_dir, segmentation_method='gmm', n_clusters=3)

    # # Display the resized image
    # plt.figure(figsize=(8, 8))
    # plt.imshow(resized_image)
    # plt.title("Resized Image")
    # plt.axis('off')
    # plt.show()
    
    # # Display the segmented image
    # plt.figure(figsize=(8, 8))
    # plt.imshow(segmented_image_gmm, cmap='gray')
    # plt.title("Segmented Image (GMM)")
    # plt.axis('off')
    # plt.show()

    # # # Display the receipt mask
    # # plt.figure(figsize=(8, 8))
    # # plt.imshow(receipt_mask, cmap='gray')
    # # plt.title("Receipt Mask")
    # # plt.axis('off')
    # # plt.show()

    # # Display the cropped image
    # plt.figure(figsize=(8, 8))
    # plt.imshow(cropped_image)
    # plt.title("Cropped Image (Receipt Area)")
    # plt.axis('off')
    # plt.show()
    
    # # Display the second segmented image
    # if needs_second_cropping:
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(segment_image_gmm2, cmap='gray')
    #     plt.title("Segmented Image after First Cropping (GMM with 3 components)")
    #     plt.axis('off')
    #     plt.show()
    
    # # Display the further cropped image if second cropping was performed
    # if needs_second_cropping:
    #     plt.figure(figsize=(8, 8))
    #     plt.imshow(cropped_image)
    #     plt.title("Further Cropped Image (Receipt Area after Second Cropping)")
    #     plt.axis('off')
    #     plt.show()    
