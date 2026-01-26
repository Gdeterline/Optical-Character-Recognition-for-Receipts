import os, sys
import torch
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import extract_charset, build_vocab, build_dataloaders, split_receipts, build_samples, OCRDataset
from optical_character_recognition.recognition import CRNN

def greedy_decode(logits, idx2char, blank_idx=0):
    """
    Greedy CTC decoding.

    Parameters
    ----------
    logits : torch.Tensor
        Logits output from the model of shape (T, B, num_classes).
    idx2char : dict
        Mapping from index to character.
    blank_idx : int, optional
        Index of the CTC blank character. Default is 0.
        
    Returns
    -------
    list of str
        Decoded text strings for each batch element.
    """
    probs = logits.softmax(2)
    max_probs = torch.argmax(probs, dim=2)  # (T, B)

    decoded_texts = []

    for b in range(max_probs.size(1)):
        prev_char = None
        decoded = []

        for t in range(max_probs.size(0)):
            char_idx = max_probs[t, b].item()

            # CTC rules
            if char_idx != blank_idx and char_idx != prev_char:
                decoded.append(idx2char[char_idx])

            prev_char = char_idx

        decoded_texts.append("".join(decoded))

    return decoded_texts


def cer(pred, gt):
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


def evaluate_ocr(model, dataloader, idx2char, blank_idx=0, device="cpu", max_batches=4):
    """
    Runs greedy decoding on a few validation batches and prints predictions.
    Also computes average CER.
    """
    model.eval()
    total_cer = 0
    count = 0

    with torch.no_grad():
        for batch_idx, (images, _, _, texts) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            images = images.to(device)
            logits = model(images)

            preds = greedy_decode(logits, idx2char, blank_idx)

            for gt, pred in zip(texts, preds):
                sample_cer = cer(pred, gt)
                total_cer += sample_cer
                count += 1

                print(f"Ground Truth: {gt}")
                print(f"Prediction  : {pred}")
                print(f"CER: {sample_cer:.2f}")
                print("-" * 40)

    avg_cer = total_cer / max(1, count)
    print(f"\nAverage CER (preview): {avg_cer:.3f}")

    return avg_cer

# function to display the predictions on a few validation samples with the images and the ground truth
def display_predictions(model, dataloader, idx2char, blank_idx=0, device="cpu", num_images=8, max_batches=4):
    """
    Displays images with their ground truth and predicted texts, in subplots.
    """
    model.eval()
    images_displayed = 0
    plt.figure(figsize=(12, 6))
    with torch.no_grad():
        for batch_idx, (images, _, _, texts) in enumerate(dataloader):
            if batch_idx >= max_batches:
                plt.suptitle("OCR Predictions vs Ground Truths on the validation set", fontsize=16)
                plt.tight_layout()
                plt.savefig("reports/figures/ocr_predictions.png")
                plt.show()
                break

            images = images.to(device)
            logits = model(images)

            preds = greedy_decode(logits, idx2char, blank_idx)

            for i in range(images.size(0)):
                if images_displayed >= num_images:
                    break

                img = images[i].cpu().squeeze(0).numpy()
                gt = texts[i]
                pred = preds[i]

                plt.subplot(1, num_images, images_displayed + 1)
                plt.imshow(img, cmap='gray')
                plt.title(f"GT: {gt}\nPred: {pred}")
                plt.axis('off')

                images_displayed += 1

            if images_displayed >= num_images:
                plt.suptitle("OCR Predictions vs Ground Truths on the validation set", fontsize=16)
                plt.tight_layout()
                plt.savefig("reports/figures/ocr_predictions.png")
                plt.show()
                break


if __name__ == "__main__":

    json_annotations_path = "data/filename_to_word_files.json"
    chars = extract_charset(json_annotations_path)
    char2idx, idx2char, blank_idx, num_classes = build_vocab(chars)

    # seed is the same as for the training split, so no data leakage occurs

    train_receipts, val_receipts = split_receipts(json_annotations_path, train_ratio=0.8)
    train_samples = build_samples(json_annotations_path, train_receipts)
    val_samples   = build_samples(json_annotations_path, val_receipts)
    
    _, val_loader = build_dataloaders(
        train_dataset=OCRDataset(train_samples, char2idx),
        val_dataset=OCRDataset(val_samples, char2idx),
        batch_size=8,
    )

    model = CRNN(num_classes=num_classes)
    model.load_state_dict(torch.load("crnn_weights.pth", map_location="cpu"))

    # evaluate_ocr(
    #     model,
    #     val_loader,
    #     idx2char,
    #     blank_idx=blank_idx,
    #     device="cpu",
    #     max_batches=4
    # )
    
    display_predictions(
        model,
        val_loader,
        idx2char,
        blank_idx=blank_idx,
        device="cpu",
        num_images=8,
        max_batches=4
    )
    
    