import json
import random
import os, sys
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset.dataset import extract_charset, build_vocab, split_receipts, build_samples, OCRDataset, build_dataloaders
from optical_character_recognition.recognition import CRNN
from preprocessing.bboxes_preprocessing_pipeline import bboxes_preprocessing_pipeline
from preprocessing.images_preprocessing_pipeline import images_preprocessing_pipeline
from preprocessing.words_annotations_preprocessing_pipeline import words_annotations_preprocessing_pipeline
from evaluation.utils import greedy_decode, cer

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

def preprocess_test_set(
    raw_images_dir: str = "data/images", 
    data_subset: str = "test_", 
    segmentation_method: str = "gmm", 
    n_clusters: int = 2, 
    second_cropping_threshold: float = 0.85, 
    pickle_metadata_filepath: str = "data/metadata.pkl",
    output_images_path: str = "data/preprocessed_images_test", 
    output_meta_file: str = "data/coordinates_transformation_meta_test.json", 
    output_preprocessed_bboxes_filepath: str = "data/preprocessed_bboxes_test.json",
    dir_output_cropped_words: str = "data/cropped_words_test/",
    output_filename_word_labels: str = "data/filename_to_word_files_test.json",
    verbose: bool = True
    ):
    """
    Preprocesses entirely the test set of receipt images for OCR, without doing detection though.
    It includes image preprocessing and bounding box preprocessing.
    
    Parameters
    ----------
    For details, see the docstrings of images_preprocessing_pipeline, bboxes_preprocessing_pipeline, and words_annotations_preprocessing_pipeline functions.
    """
    if verbose:
        print("Starting preprocessing of the images test set...")
        
    # preprocessed_images, transformation_meta_files = images_preprocessing_pipeline(
    #     raw_images_dir=raw_images_dir,
    #     data_subset=data_subset,
    #     segmentation_method=segmentation_method,
    #     n_clusters=n_clusters,
    #     second_cropping_threshold=second_cropping_threshold,
    #     output_images_path=output_images_path,
    #     output_meta_file=output_meta_file,
    #     verbose=verbose
    # )
    
    # if verbose:
    #     print("Image preprocessing completed. Starting bounding boxes preprocessing...")
    
    # # Done directly with the bboxes_preprocessing_pipeline because was causing issues when called from main.py: no sure why ????
    # # Problem was with .iloc 
    # preprocessed_bboxes = bboxes_preprocessing_pipeline(
    #     metadata_filepath=pickle_metadata_filepath,
    #     transformations_meta_filepath=output_meta_file,
    #     output_filepath=output_preprocessed_bboxes_filepath,
    #     subset=data_subset,
    #     verbose=True
    # )
    
    if verbose:
        print("Bounding boxes preprocessing completed.")
        print("Starting cropping words from the preprocessed images...")
        
    words_annotations_preprocessing_pipeline(
        preprocessed_images_dir=output_images_path,
        bboxes_json_file=output_preprocessed_bboxes_filepath,
        metadata_filename=pickle_metadata_filepath,
        output_cropped_words_dir=dir_output_cropped_words,
        map_filename_to_word_files_json=output_filename_word_labels,
        size=(128, 128),
        verbose=True
    )
    
    if verbose:
        print("Cropping words completed.")
    
    return

def predict_receipt(
    receipt_filename: str,
    output_filename_word_labels: str,
    model: CRNN,
    idx2char: dict,
    blank_idx: int,
    device: torch.device
    ):
    """
    Predicts all words for a given receipt and compares with ground truth.
    
    Parameters
    ----------
    receipt_filename : str
        Filename of the receipt (key in the JSON file), e.g., "test_receipt_00000.png".
    output_filename_word_labels : str
        Path to the JSON mapping receipt filenames to word files.
    model : CRNN
        Trained CRNN model.
    idx2char : dict
        Vocabulary mapping.
    blank_idx : int
        Blank index for CTC.
    device : torch.device
        Device.
        
    Returns
    -------
    list of dict
        List of dictionaries containing word_file, predicted_word, and true_word.
    """
    with open(output_filename_word_labels, "r") as f:
        data = json.load(f)
        
    if receipt_filename not in data:
        raise ValueError(f"Receipt {receipt_filename} not found in {output_filename_word_labels}")
        
    word_entries = data[receipt_filename]
    results = []
    
    transform = T.ToTensor()
    
    model.eval()
    
    for entry in word_entries:
        word_file_path = entry["word_file"]
        true_word = entry["word"]
        
        # Open and preprocess image
        if not os.path.exists(word_file_path):
            print(f"Warning: File {word_file_path} does not exist. Skipping.")
            continue
            
        try:
            image = Image.open(word_file_path).convert("L")
            image_tensor = transform(image).unsqueeze(0).to(device) # (1, 1, H, W)
            
            predicted_word = predict(model, image_tensor, idx2char, blank_idx)
            
            results.append({
                "word_file": word_file_path,
                "predicted_word": predicted_word,
                "true_word": true_word
            })
        except Exception as e:
            print(f"Error processing {word_file_path}: {e}")
            
    return results


def build_test_dataloader(
    output_filename_word_labels: str = "data/filename_to_word_files_test.json",
    number_of_samples: int = 50,
    verbose: bool = True
    ):        
    
    chars = extract_charset(output_filename_word_labels)
    char2idx, idx2char, blank_idx, num_classes = build_vocab(chars)
    
    if verbose:
        print("Character set and vocabulary built successfully.")
        print(f"Selecting {number_of_samples} random samples from the test set for prediction...")
    
    # Select 50 random receipts from the test set for prediction
    with open(output_filename_word_labels, "r") as f:
        data = json.load(f)
        
    test_receipts = list(data.keys())
    random.shuffle(test_receipts)
    selected_receipts = test_receipts[:number_of_samples]
    
    if verbose:
        print(f"Selected {number_of_samples} receipts for the test set.")
        print("Building samples and dataloader...")
    
    test_samples = build_samples(output_filename_word_labels, selected_receipts)
    test_dataset = OCRDataset(test_samples, char2idx)
    
    _, test_loader = build_dataloaders(
        train_dataset=test_dataset,
        val_dataset=test_dataset,
        batch_size=8,
    )
    
    if verbose:
        print("Dataloader for the test set built successfully.")
    
    return test_loader, char2idx, idx2char, blank_idx, num_classes


# Function that runs predictions on n_receipts for a given model
def run_predictions_on_test_set(
    output_filename_word_labels: str = "data/filename_to_word_files_test.json",
    model_path: str = "crnn_weights.pth",
    device: torch.device = torch.device("cpu"),
    number_of_samples: int = 50,
):
    test_loader, char2idx, idx2char, blank_idx, num_classes = build_test_dataloader(
        output_filename_word_labels=output_filename_word_labels,
        number_of_samples=number_of_samples,
        verbose=True
    )
    
    # Load the trained model
    model = load_model(model_path, num_classes, device)
    model.eval()
    
    # Compute, over the number_of_samples samples from the test set, the average cer and the word accuracy
    total_cer = 0.0
    total_word_acc = 0.0
    
    total_words = 0
    total_correct_words = 0
    total_chars = 0
    total_errors = 0
    
    for batch_idx, (images, _, _, texts) in enumerate(test_loader):
        images = images.to(device)
        
        logits = model(images)
        preds = greedy_decode(logits, idx2char, blank_idx)
        
        for i in range(images.size(0)):
            gt = texts[i]
            pred = preds[i]
            
            # Compute character error rate (CER)
            cer_value = cer(gt=gt, pred=pred)
            total_cer += cer_value
            
            # Compute word accuracy
            if gt == pred:
                total_correct_words += 1
            total_words += 1
            
            total_chars += len(gt)
            total_errors += cer_value * len(gt)
            
    average_cer = total_errors / total_chars if total_chars > 0 else 0.0
    word_accuracy = total_correct_words / total_words if total_words > 0 else 0.0
    
    print(f"Average Character Error Rate (CER) over {number_of_samples} samples: {average_cer:.4f}")
    print(f"Word Accuracy over {number_of_samples} samples: {word_accuracy:.4f}")
    
    return average_cer, word_accuracy    



if __name__ == "__main__":
    
    device = torch.device("cpu")
    chars = extract_charset("data/filename_to_word_files.json")
    char2idx, idx2char, blank_idx, num_classes = build_vocab(chars)
    
    receipts_all_predictions = []
    model = load_model("crnn_weights_opt.pth", num_classes=num_classes, device=device)
    
    
    for index, filename in enumerate(os.listdir("data/preprocessed_images_test/")):
        if filename.startswith("test_") and filename.endswith(".png"):
            print(f"Predicting for receipt: {filename}... {index + 1}/{len(os.listdir('data/preprocessed_images_test/'))}")
            results = predict_receipt(
                receipt_filename=filename,
                output_filename_word_labels="data/filename_to_word_files_test.json",
                model=model,
                idx2char=idx2char,
                blank_idx=blank_idx,
                device=device
            )
            
            receipts_all_predictions.append({
                "receipt_filename": filename,
                "predictions": results
            })
            
            # COmpute cer for this receipt
            total_chars = 0
            total_errors = 0
            for res in results:
                gt = res["true_word"]
                pred = res["predicted_word"]
                total_chars += len(gt)
                total_errors += cer(gt, pred) * len(gt)
            
    # Save all predictions to a JSON file
    with open("data/test_set_predictions.json", "w") as f:
        json.dump(receipts_all_predictions, f, indent=4)
        
    # Display three few receipts, their predictions and the cer score
    plt.figure(figsize=(21, 7))
        
    for i, receipt_prediction in enumerate(receipts_all_predictions[-3:]):
        receipt_filename = receipt_prediction["receipt_filename"]
        predictions = receipt_prediction["predictions"]
        
        total_chars = 0
        total_errors = 0
        for res in predictions:
            gt = res["true_word"]
            pred = res["predicted_word"]
            total_chars += len(gt)
            total_errors += cer(gt, pred) * len(gt)
        
        receipt_cer = total_errors / total_chars if total_chars > 0 else 0.0
        
        # print the ground truths and the predictions for each receipt
        print(f"\nReceipt: {receipt_filename}")
        for res in predictions:
            print(f"True: {res['true_word']}, Predicted: {res['predicted_word']}")
        
        plt.subplot(1, 3, i + 1)
        image_path = os.path.join("data/preprocessed_images_test", receipt_filename)
        image = Image.open(image_path).convert("L")
        plt.imshow(image, cmap='gray')
        plt.title(f"Receipt: {receipt_filename}\nCER: {receipt_cer:.4f}", fontsize=14)
        plt.axis('off')
        
    plt.suptitle("Sample Receipt Predictions from the Test Set", fontsize=16)
    plt.tight_layout()
    plt.savefig("reports/figures/sample_receipt_predictions.png")
    plt.show()