import os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.utils import crop_words_from_receipts

def words_annotations_preprocessing_pipeline(
    preprocessed_images_dir: str,
    bboxes_json_file: str,
    metadata_filename: str,
    output_cropped_words_dir: str,
    map_filename_to_word_files_json: str,
    size: tuple = (128, 128),
    verbose: bool = True
):
    """
    Preprocess receipt images to crop words based on provided bounding boxes and annotations.
    
    Parameters
    ----------
    preprocessed_images_dir : str
        Directory containing preprocessed receipt images.
    bboxes_json_file : str
        JSON file containing bounding boxes for words in the images.
    metadata_filename : str
        Pickle file containing metadata with annotations/labels for each word.
    output_cropped_words_dir : str
        Directory to save cropped word images.
    map_filename_to_word_files_json : str
        JSON file to save mapping of image filenames to their corresponding cropped word image file paths.
    size : tuple, optional
        Size to which each cropped word image will be resized, by default (128, 128).
    verbose : bool, optional
        Whether to print progress messages, by default True.
        
    Returns
    -------
    None
    """
    
    # List all image files in the preprocessed images directory
    image_files = [f for f in os.listdir(preprocessed_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    
    if verbose:
        print(f"Found {len(image_files)} preprocessed images to process for word cropping.")
        
    for index, image_file in enumerate(image_files):
        if verbose:
            print(f"Processing image: {image_file} ({index + 1}/{len(image_files)})")
        
        # Crop words from the receipt image
        cropped_images, filename_to_word_files, words = crop_words_from_receipts(
            image_filename=image_file,
            preprocessed_images_dir=preprocessed_images_dir,
            metadata_filename=metadata_filename,
            preprocessed_bboxes_jsonfile=bboxes_json_file,
            size=size,
            output_cropped_words_dir=output_cropped_words_dir,
            map_filename_to_word_files_json=map_filename_to_word_files_json,
        )
        
    if verbose:
        print("Completed cropping words from all images.")
        
    return

if __name__ == "__main__":
    preprocessed_images_dir = "data/preprocessed_images/"
    bboxes_json_file = "data/preprocessed_bboxes.json"
    metadata_filename = "data/metadata.pkl"
    output_cropped_words_dir = "data/cropped_words/"
    map_filename_to_word_files_json = "data/filename_to_word_files.json"
    size = (128, 128)
    
    words_annotations_preprocessing_pipeline(
        preprocessed_images_dir,
        bboxes_json_file,
        metadata_filename,
        output_cropped_words_dir,
        map_filename_to_word_files_json,
        size=size,
        verbose=True
    )