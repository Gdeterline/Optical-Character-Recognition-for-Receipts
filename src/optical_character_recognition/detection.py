import cv2
import numpy as np

def detect_text_boxes(binary_img, kernel_size=(20, 1)):
    """
    Detects text boxes in a binary image using contour detection.
    The principle is the same as we used for skew detection, but here we dilate more aggressively to capture whole lines of text.
    
    Parameters
    ----------
    binary_img (np.ndarray): 
        A binary image where text is white on a black background.
    Returns
    -------
    results (list of tuples):
        List of bounding boxes around detected text areas. Each box is represented as (x, y, w, h).
    """
    # Invert if text is black (contours need white text on black background)
    # The function does not render anything good otherwise
    if binary_img[0, 0] > 127:
        binary_img = cv2.bitwise_not(binary_img)
    
    # Define a kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    
    # Proceed with dilation to connect text regions
    dilated = cv2.dilate(binary_img, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Extract bounding boxes from contours
    results = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter out small boxes that are msot likely noise
        if w * h < 400: 
            continue 
        
        # Filter out boxes that are too tall (likely not text lines)
        if h > (binary_img.shape[0] * 0.15): 
            continue
        
        # Filter out boxes that are on the edges of the image (likely borders)
        if x <= 1 or y <= 1 or (x + w) >= (binary_img.shape[1] - 2) or (y + h) >= (binary_img.shape[0] - 2):
            continue
        
        results.append((x, y, w, h))
        
    return results