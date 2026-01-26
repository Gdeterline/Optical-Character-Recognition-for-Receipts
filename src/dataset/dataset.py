import os, sys
import json
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from sklearn.model_selection import train_test_split
from collections import Counter

sys.path.append(os.path.abspath(os.path.join('../src')))

# add parent directory to sys.path to access data without having to put "../data"
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

#################################################### Functions to extract charset and build vocab ####################################################

# Extract character set from JSON annotations
def extract_charset(json_path):
    """
    Extracts the set of unique characters from the JSON annotations file.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON annotations file.
        
    Returns
    -------
    chars : list of str
        Sorted list of unique characters.
    """
    chars = set()

    with open(json_path, "r") as f:
        data = json.load(f)

    for _, words in data.items():
        for entry in words:
            chars.update(entry["word"])

    chars = sorted(chars)
    return chars

# Build vocabulary mappings from character set
def build_vocab(chars):
    """
    Builds character to index and index to character mappings, along with blank index and number of classes from the given character set.

    Parameters
    ----------
    chars : list of str
        List of unique characters.
        
    Returns
    -------
    char2idx : dict
        Mapping from character to index.
    idx2char : dict
        Mapping from index to character.
    blank_idx : int
        Index reserved for the blank token.
    num_classes : int
        Total number of classes (including blank).
    """
    char2idx = {c: i + 1 for i, c in enumerate(chars)}  # 0 = blank
    idx2char = {i + 1: c for i, c in enumerate(chars)}
    blank_idx = 0
    num_classes = len(chars) + 1

    return char2idx, idx2char, blank_idx, num_classes


#################################################### Functions to split data and build samples ####################################################

def word_category(word: str):
    """
    Takes a word string and returns its category: 'alpha', 'digit', 'mixed', or 'other'.
    """
    has_alpha = any(c.isalpha() for c in word)
    has_digit = any(c.isdigit() for c in word)

    if has_alpha and has_digit:
        return "mixed"
    elif has_alpha:
        return "alpha"
    elif has_digit:
        return "digit"
    else:
        return "other"

def receipt_category(words):
    """
    Assign one category to a receipt based on majority vote. 
    The goal is to categorize receipts into 'alpha', 'digit', 'mixed', or 'other' based on the words they contain, to create a somewhat balanced split.
    
    Parameters
    ----------
    words : list of str
        List of words in the receipt.
    Returns
    -------
    str
        Category of the receipt.
    """
    cats = [word_category(w) for w in words]
    counts = Counter(cats)
    return counts.most_common(1)[0][0]

def split_receipts(json_path, train_ratio=0.8, seed=42):
    """
    Splits receipt filenames into training and validation sets, at the receipt level (to avoid data leakage).
    
    Parameters
    ----------
    json_path : str
        Path to the JSON annotations file.
    train_ratio : float
        Proportion of receipts to include in the training set.
    seed : int
        seed for reproducibility.
    """
    random.seed(seed)

    with open(json_path, "r") as f:
        data = json.load(f)

    receipts = list(data.keys())
    random.shuffle(receipts)

    n_train = int(len(receipts) * train_ratio)

    train_receipts = receipts[:n_train]
    val_receipts = receipts[n_train:]

    return train_receipts, val_receipts



def stratified_split_receipts(
    json_path,
    train_ratio=0.8,
    seed=42
):
    """
    Same as split_receipts but ensures stratification based on receipt main categories.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    receipts = list(data.keys())

    # Compute label for each receipt
    receipt_labels = []
    for receipt in receipts:
        words = [entry["word"] for entry in data[receipt]]
        label = receipt_category(words)
        receipt_labels.append(label)

    train_receipts, val_receipts = train_test_split(
        receipts,
        test_size=1 - train_ratio,
        random_state=seed,
        stratify=receipt_labels
    )

    return train_receipts, val_receipts

def count_receipt_labels(receipts, json_path="data/filename_to_word_files.json"):
    
    with open(json_path) as f:
        data = json.load(f)
    labels = []
    for r in receipts:
        words = [e["word"] for e in data[r]]
        labels.append(receipt_category(words))
    return Counter(labels)

# Build samples list from receipt filenames
def build_samples(json_path, receipt_filenames):
    """
    Builds a list of samples (image_path, text) from the given receipt filenames.
    
    Parameters
    ----------
    json_path : str
        Path to the JSON annotations file.
    receipt_filenames : list of str
        List of receipt filenames to build samples from.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    samples = []

    for filename in receipt_filenames:
        for entry in data[filename]:
            samples.append(
                (entry["word_file"], entry["word"])
            )

    return samples


##################################################### OCR Dataset Class ###################################################################

class OCRDataset(Dataset):
    """
    This class implements a PyTorch Dataset for our OCR task.
    It handles loading images and their corresponding text labels,
    encoding the text into indices, and applying any necessary transformations.
    """
    def __init__(self, samples, char2idx, transform=None):
        """
        samples: list of tuples (image_path, text)
        char2idx: dictionary mapping characters to indices
        """
        self.samples = samples
        self.char2idx = char2idx
        self.to_tensor = T.ToTensor()

    def encode(self, text):
        """
        The encode method converts a text string into a tensor of character indices.
        This is essential for preparing the target labels for training the OCR model.
        """
        return torch.tensor(
            [self.char2idx[c] for c in text],
            dtype=torch.long
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, text = self.samples[idx]

        # the file_names in the json are stored as data/... so we need to go one directory up
        image = Image.open(img_path).convert("L")

        # convert to tensor [1, 128, 128]
        image = self.to_tensor(image)

        target = self.encode(text)
        target_length = len(target)

        return image, target, target_length, text
    
    
############################################# DataLoader and Collate Function ########################################################################
    
    
def ctc_collate_fn(batch):
    """
    Custom collate function for CTC loss that handles variable-length targets.
    In classification tasks, this is not needed, because all targets are of the same length. 
    Given we are not doing classification, but recognition with CTC loss, we need this function.
    
    Parameters
    ----------
    batch : list of tuples
        Each tuple contains (image, target, target_length, text).
        
    Returns
    -------
    images : torch.Tensor
        Batched images tensor.
    targets : torch.Tensor
        Concatenated targets tensor.
    target_lengths : torch.Tensor
        Tensor of target lengths.
    texts : list of str
        List of original text strings.
    """
    images, targets, target_lengths, texts = zip(*batch)

    images = torch.stack(images)
    targets = torch.cat(targets)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    return images, targets, target_lengths, texts


def build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size=16,
):
    """
    Builds DataLoader objects for training and validation datasets.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=ctc_collate_fn,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=ctc_collate_fn,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    
    json_path = "data/filename_to_word_files.json"
    
    train_receipts, val_receipts = stratified_split_receipts(json_path)
    train_samples = build_samples(json_path, train_receipts)
    val_samples   = build_samples(json_path, val_receipts)

    print("Train receipt distribution:", count_receipt_labels(train_receipts, json_path))
    print("Val receipt distribution:", count_receipt_labels(val_receipts, json_path))