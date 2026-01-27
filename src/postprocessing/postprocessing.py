import os, sys
import json
import re
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as T

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import extract_charset, build_vocab
from optical_character_recognition.recognition import CRNN


def predict_word(model, image_tensor, idx2char, blank_idx):
    """
    Makes predictions on a single word image tensor using the trained CRNN model.
    """
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor)
        probs = nn.functional.softmax(logits, dim=2)
        _, pred_indices = probs.max(2)
        pred_indices = pred_indices.squeeze(1).cpu().numpy()

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
    """
    model = CRNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def is_amount(text: str) -> bool:
    """
    Check if a text string looks like an amount (contains digits, possibly with currency symbols, commas, periods).
    """
    # Remove common currency symbols and whitespace
    cleaned = re.sub(r'[Rp\.\s€\$£¥]', '', text)
    # Check if what remains has digits and possibly commas/periods
    # If so, we consider it an amount
    return bool(re.search(r'\d', cleaned))


def get_y_center(bbox: dict) -> float:
    """
    Get the vertical center of a bounding box.
    """
    return (bbox['y1'] + bbox['y2'] + bbox['y3'] + bbox['y4']) / 4


def get_x_center(bbox: dict) -> float:
    """
    Get the horizontal center (used for sorting left to right).
    """
    return (bbox['x1'] + bbox['x2'] + bbox['x3'] + bbox['x4']) / 4


def group_words_by_line(bboxes: list, word_predictions: list, y_margin: int = 20) -> list:
    """
    Group words by their y-coordinate (same line) with a small margin.
    Returns a list of lines, each line being a list of (x_center, predicted_word) tuples sorted by x.
    """
    # Combine bboxes with predictions
    word_data = []
    for i, (bbox, pred) in enumerate(zip(bboxes, word_predictions)):
        y_center = get_y_center(bbox)
        x_center = get_x_center(bbox)
        word_data.append({
            'y_center': y_center,
            'x_center': x_center,
            'prediction': pred
        })
    
    # Sort by y_center
    word_data.sort(key=lambda w: w['y_center'])
    
    # Group into lines
    lines = []
    current_line = []
    current_y = None
    
    for word in word_data:
        if current_y is None:
            current_y = word['y_center']
            current_line.append(word)
        elif abs(word['y_center'] - current_y) <= y_margin:
            current_line.append(word)
        else:
            # New line
            if current_line:
                lines.append(current_line)
            current_line = [word]
            current_y = word['y_center']
    
    if current_line:
        lines.append(current_line)
    
    # Sort each line by x_center (left to right)
    for line in lines:
        line.sort(key=lambda w: w['x_center'])
    
    return lines


def extract_object_amount_from_line(line: list) -> tuple:
    """
    From a line of words, extract the object description and the amount.
    The amount is the rightmost element that looks like a number.
    Returns (object_text, amount_text) or (None, None) if no amount found.
    """
    if not line:
        return None, None
    
    # Find the rightmost amount
    amount_idx = None
    for i in range(len(line) - 1, -1, -1):
        if is_amount(line[i]['prediction']):
            amount_idx = i
            break
    
    if amount_idx is None:
        return None, None
    
    # Object is everything to the left of the amount
    object_words = [w['prediction'] for w in line[:amount_idx]]
    amount_text = line[amount_idx]['prediction']
    
    if not object_words:
        return None, amount_text
    
    object_text = ' '.join(object_words)
    return object_text, amount_text


def receipts_to_dataframe(
    model_path: str = "crnn_weights_opt.pth",
    vocab_file: str = "data/filename_to_word_files.json",
    bboxes_file: str = "data/preprocessed_bboxes.json",
    word_labels_file: str = "data/filename_to_word_files.json",
    cropped_words_dir: str = "data/cropped_words",
    device: torch.device = None,
    y_margin: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    For each receipt in the dataset (no matter whether dev_ or test_), extract all tuples of (object, amount associated)
    in the receipt. The assumption is that a tuple is defined as an object and the amount on the same line,
    where the amount is the rightmost number on that line.
    To do so, read the coordinates of each word bounding box from the annotations file, group the words by their y-coordinate
    with a small margin (to account for small misalignments), and for each group, identify the rightmost number as the amount.
    The function returns a pandas DataFrame with columns: 'receipt_filename', 'object', 'amount'.
    
    Parameters
    ----------
    model_path : str
        Path to the trained CRNN model weights.
    vocab_file : str
        Path to the JSON file used to build vocabulary.
    bboxes_file : str
        Path to the preprocessed bboxes JSON file.
    word_labels_file : str
        Path to the JSON mapping receipt filenames to word files.
    cropped_words_dir : str
        Directory containing cropped word images.
    device : torch.device
        Device to run inference on.
    y_margin : int
        Margin for grouping words on the same line.
    verbose : bool
        Whether to print progress information.
        
    Returns
    -------
    df_receipts : pd.DataFrame
        DataFrame containing all (object, amount) tuples for each receipt in the dataset.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print("Building vocabulary from training data...")
    
    # Build vocabulary from training data
    chars = extract_charset(vocab_file)
    char2idx, idx2char, blank_idx, num_classes = build_vocab(chars)
    
    if verbose:
        print(f"Vocabulary built with {num_classes} classes.")
        print(f"Loading model from {model_path}...")
    
    # Load model
    model = load_model(model_path, num_classes, device)
    model.eval()
    
    if verbose:
        print("Model loaded successfully.")
        print("Loading bounding boxes and word labels...")
    
    # Load bboxes
    with open(bboxes_file, 'r') as f:
        all_bboxes = json.load(f)
    
    # Load word labels (to get word file paths)
    with open(word_labels_file, 'r') as f:
        word_labels = json.load(f)
    
    if verbose:
        print(f"Loaded {len(all_bboxes)} receipts. Starting predictions...")
    
    transform = T.ToTensor()
    
    results = []
    total_receipts = len(all_bboxes)
    total_receipts = len(all_bboxes)
    
    for idx, receipt_filename in enumerate(all_bboxes.keys()):
        if verbose:
            print(f"Processing receipt {idx + 1}/{total_receipts}...")
        
        bboxes = all_bboxes[receipt_filename]
        
        # Get the word files for this receipt
        if receipt_filename not in word_labels:
            continue
        
        word_entries = word_labels[receipt_filename]
        
        # Predict each word
        word_predictions = []
        for entry in word_entries:
            word_file_path = entry["word_file"]
            
            if not os.path.exists(word_file_path):
                word_predictions.append("")
                continue
            
            try:
                image = Image.open(word_file_path).convert("L")
                image_tensor = transform(image).unsqueeze(0).to(device)
                pred = predict_word(model, image_tensor, idx2char, blank_idx)
                word_predictions.append(pred)
            except Exception as e:
                word_predictions.append("")
        
        # Group words by line
        if len(bboxes) != len(word_predictions):
            # Mismatch, skip this receipt
            continue
        
        lines = group_words_by_line(bboxes, word_predictions, y_margin=y_margin)
        
        # Extract object-amount pairs from each line
        for line in lines:
            obj, amount = extract_object_amount_from_line(line)
            if obj and amount:
                results.append({
                    'receipt_filename': receipt_filename,
                    'object': obj,
                    'amount': amount
                })
    
    df_receipts = pd.DataFrame(results)
    
    if verbose:
        print(f"Done! Extracted {len(df_receipts)} (object, amount) pairs from {total_receipts} receipts.")
    
    return df_receipts


def cer(pred: str, gt: str) -> float:
    """
    Character Error Rate using Levenshtein distance.
    """
    import numpy as np

    dp = np.zeros((len(pred)+1, len(gt)+1), dtype=int)

    for i in range(len(pred)+1):
        dp[i][0] = i
    for j in range(len(gt)+1):
        dp[0][j] = j

    for i in range(1, len(pred)+1):
        for j in range(1, len(gt)+1):
            cost = 0 if pred[i-1] == gt[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,      # deletion
                dp[i][j-1] + 1,      # insertion
                dp[i-1][j-1] + cost  # substitution
            )

    return dp[len(pred)][len(gt)] / max(1, len(gt))


def evaluate_receipts_to_dataframe(
    model_path: str = "crnn_weights_opt.pth",
    vocab_file: str = "data/filename_to_word_files.json",
    word_labels_file: str = "data/filename_to_word_files.json",
    device: torch.device = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    For each receipt and each word, computes prediction quality metrics.
    Returns a DataFrame with columns: 'receipt_filename', 'ground_truth', 'prediction', 'cer', 'word_accuracy'.
    The last row contains average CER and word accuracy across all words.
    
    Parameters
    ----------
    model_path : str
        Path to the trained CRNN model weights.
    vocab_file : str
        Path to the JSON file used to build vocabulary (should be training vocab).
    word_labels_file : str
        Path to the JSON mapping receipt filenames to word files (with ground truth).
    device : torch.device
        Device to run inference on.
    verbose : bool
        Whether to print progress information.
        
    Returns
    -------
    df_evaluation : pd.DataFrame
        DataFrame with columns: 'receipt_filename', 'ground_truth', 'prediction', 'cer', 'word_accuracy'.
        Last row contains averages.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if verbose:
        print("Building vocabulary from training data...")
    
    # Build vocabulary from training data
    chars = extract_charset(vocab_file)
    char2idx, idx2char, blank_idx, num_classes = build_vocab(chars)
    
    if verbose:
        print(f"Vocabulary built with {num_classes} classes.")
        print(f"Loading model from {model_path}...")
    
    # Load model
    model = load_model(model_path, num_classes, device)
    model.eval()
    
    if verbose:
        print("Model loaded successfully.")
        print("Loading word labels...")
    
    # Load word labels (contains ground truth and word file paths)
    with open(word_labels_file, 'r') as f:
        word_labels = json.load(f)
    
    if verbose:
        print(f"Loaded {len(word_labels)} receipts. Starting predictions...")
    
    transform = T.ToTensor()
    
    results = []
    total_receipts = len(word_labels)
    total_cer = 0.0
    total_correct = 0
    total_words = 0
    
    for idx, receipt_filename in enumerate(word_labels.keys()):
        if verbose:
            print(f"Processing receipt {idx + 1}/{total_receipts}...")
        
        word_entries = word_labels[receipt_filename]
        
        # Predict each word and compare with ground truth
        for entry in word_entries:
            word_file_path = entry["word_file"]
            ground_truth = entry["word"]
            
            if not os.path.exists(word_file_path):
                continue
            
            try:
                image = Image.open(word_file_path).convert("L")
                image_tensor = transform(image).unsqueeze(0).to(device)
                prediction = predict_word(model, image_tensor, idx2char, blank_idx)
                
                # Compute CER
                word_cer = cer(prediction, ground_truth)
                
                # Compute word accuracy (1 if exact match, 0 otherwise)
                word_acc = 1.0 if prediction == ground_truth else 0.0
                
                results.append({
                    'receipt_filename': receipt_filename,
                    'ground_truth': ground_truth,
                    'prediction': prediction,
                    'cer': word_cer,
                    'word_accuracy': word_acc
                })
                
                total_cer += word_cer
                total_correct += word_acc
                total_words += 1
                
            except Exception as e:
                if verbose:
                    print(f"Error processing {word_file_path}: {e}")
    
    # Compute averages
    avg_cer = total_cer / total_words if total_words > 0 else 0.0
    avg_word_acc = total_correct / total_words if total_words > 0 else 0.0
    
    # Add summary row
    results.append({
        'receipt_filename': '---',
        'ground_truth': '---',
        'prediction': '---',
        'cer': avg_cer,
        'word_accuracy': avg_word_acc
    })
    
    df_evaluation = pd.DataFrame(results)
    
    if verbose:
        print(f"Done! Evaluated {total_words} words from {total_receipts} receipts.")
        print(f"Average CER: {avg_cer:.4f}")
        print(f"Average Word Accuracy: {avg_word_acc:.4f}")
    
    return df_evaluation


if __name__ == "__main__":
    pass
    # df_dev = receipts_to_dataframe(
    #     model_path="crnn_weights_opt.pth",
    #     vocab_file="data/filename_to_word_files.json",
    #     bboxes_file="data/preprocessed_bboxes.json",
    #     word_labels_file="data/filename_to_word_files.json",
    #     cropped_words_dir="data/cropped_words",
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     y_margin=20,
    #     verbose=True
    # )
    # print(df_dev.head(20))
    # print(f"\nTotal rows: {len(df_dev)}")
    
    # # save to CSV
    # df_dev.to_csv("results/dev_receipts_object_amounts.csv", index=False)
    
    # df_test = receipts_to_dataframe(
    #     model_path="crnn_weights_opt.pth",
    #     vocab_file="data/filename_to_word_files.json",  # Use TRAINING vocab to match model
    #     bboxes_file="data/preprocessed_bboxes_test.json",
    #     word_labels_file="data/filename_to_word_files_test.json",
    #     cropped_words_dir="data/cropped_words_test",
    #     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    #     y_margin=20,
    #     verbose=True
    # )
    # print(df_test.head(20))
    # print(f"\nTotal rows: {len(df_test)}")
    
    # # save to CSV
    # df_test.to_csv("results/test_receipts_object_amounts.csv", index=False)
    
    
    df_eval = evaluate_receipts_to_dataframe(
        model_path="crnn_weights_opt.pth",
        vocab_file="data/filename_to_word_files.json",
        word_labels_file="data/filename_to_word_files.json",
        verbose=True
    )
    print(df_eval.tail(10))
    
    # save to CSV
    df_eval.to_csv("results/receipts_evaluation_dev.csv", index=False)
    
    df_eval_test = evaluate_receipts_to_dataframe(
        model_path="crnn_weights_opt.pth",
        vocab_file="data/filename_to_word_files.json",
        word_labels_file="data/filename_to_word_files_test.json",
        verbose=True
    )
    print(df_eval_test.tail(10))
    
    # save to CSV
    df_eval_test.to_csv("results/receipts_evaluation_test.csv", index=False)
    
    