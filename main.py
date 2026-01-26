import os, sys
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset.dataset import extract_charset, build_vocab, split_receipts, build_samples, OCRDataset, build_dataloaders
from optical_character_recognition.recognition import CRNN

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

