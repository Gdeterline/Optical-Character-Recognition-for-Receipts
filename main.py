import json
import random
import os, sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset.dataset import extract_charset, build_vocab, split_receipts, build_samples, OCRDataset, build_dataloaders
from optical_character_recognition.recognition import CRNN
from preprocessing.bboxes_preprocessing_pipeline import bboxes_preprocessing_pipeline
from preprocessing.images_preprocessing_pipeline import images_preprocessing_pipeline

# Make preidctions using the trained model
def predict(model, image_tensor, idx2char, blank_idx):
    """
    Makes predictions on a single image tensor using the trained CRNN model.
    
    Parameters
    ----------
    model : CRNN
        Trained CRNN model.
    image_tensor : torch.Tensor
        Input image tensor of shape (1, 1, H, W).
    idx2char : dict
        Mapping from character indices to characters.
    blank_idx : int
        Index of the CTC blank character.
    
    Returns
    -------
    str
        Predicted text string.
    """
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)  # (T, 1, num_classes)
        probs = nn.functional.softmax(logits, dim=2)  # (T, 1, num_classes)
        _, pred_indices = probs.max(2)  # (T, 1)
        pred_indices = pred_indices.squeeze(1).cpu().numpy()  # (T,)

        # Decode using CTC decoding (remove duplicates and blanks)
        prev_idx = None
        pred_text = ""
        for idx in pred_indices:
            if idx != blank_idx and idx != prev_idx:
                pred_text += idx2char[idx]
            prev_idx = idx

    return pred_text


def load_model(model_path, num_classes, device):
    """
    Loads the trained CRNN model from the specified path.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model weights.
    num_classes : int
        Number of output classes for the model.
    device : torch.device
        Device to load the model onto (CPU or GPU).
        
    Returns
    -------
    CRNN
        Loaded CRNN model.
    """
    model = CRNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


images_preprocessing_pipeline(raw_images_dir="../data/images", data_subset="test_", segmentation_method="gmm", )
bboxes_preprocessing_pipeline(metadata_filepath="../data/metadata.pkl", transformations_meta_filepath="data/coordinates_transformation_meta.json", output_filepath="data/preprocessed_bboxes.json", subset="test_", verbose=True)

def preprocess_test_set(
    raw_images_dir: str = "../data/images", 
    data_subset: str = "test_", 
    segmentation_method: str = "gmm", 
    n_clusters: int = 2, 
    second_cropping_threshold: float = 0.85, 
    output_images_path: str = None, 
    output_meta_file: str = None, 
    output_filename_word_labels: str = "data/preprocessed_bboxes_test.json",
    verbose: bool = True
    ):
    """
    Preprocesses entirely the test set of receipt images for OCR, without doing detection though.
    It includes image preprocessing and bounding box preprocessing.
    
    Parameters
    ----------
    For details, see the docstrings of images_preprocessing_pipeline and bboxes_preprocessing_pipeline functions.
    """
    
    if verbose:
        print("Starting preprocessing of the images test set...")
    
    preprocessed_images, transformation_meta_files = images_preprocessing_pipeline(
        raw_images_dir=raw_images_dir,
        data_subset=data_subset,
        segmentation_method=segmentation_method,
        n_clusters=n_clusters,
        second_cropping_threshold=second_cropping_threshold,
        output_images_path=output_images_path,
        output_meta_file=output_meta_file,
        verbose=False
    )
    
    if verbose:
        print("Image preprocessing completed. Starting bounding boxes preprocessing...")
    
    preprocessed_bboxes = bboxes_preprocessing_pipeline(
        metadata_filepath=transformation_meta_files,
        transformations_meta_filepath=transformation_meta_files,
        output_filepath=output_filename_word_labels,
        subset=data_subset,
        verbose=False
    )
    
    if verbose:
        print("Bounding boxes preprocessing completed.")
        print("Starting to build dataloader for the test set...")
        
    chars = extract_charset(output_filename_word_labels)
    char2idx, idx2char, blank_idx, num_classes = build_vocab(chars)
    
    # Select 50 random receipts from the test set for prediction
    with open(output_filename_word_labels, "r") as f:
        data = json.load(f)
        
    test_receipts = list(data.keys())
    random.shuffle(test_receipts)
    selected_receipts = test_receipts[:50]
    
    test_samples = build_samples(output_filename_word_labels, selected_receipts)
    test_dataset = OCRDataset(test_samples, char2idx)
    
    _, test_loader = build_dataloaders(
        train_dataset=None,
        val_dataset=test_dataset,
        batch_size=8,
    )
    
    if verbose:
        print("Dataloader for the test set built successfully.")
    
    return test_loader, char2idx, idx2char, blank_idx, num_classes

if __name__ == "__main__":
    test_loader, char2idx, idx2char, blank_idx, num_classes = preprocess_test_set(
    )