import sys, os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
import matplotlib.pyplot as plt
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from dataset.dataset import extract_charset, build_vocab, split_receipts, build_samples, OCRDataset, build_dataloaders
from recognition import CRNN

def train(
    json_annotations_path="data/filename_to_word_files.json",
    batch_size=8,
    epochs=15,
    lr=1e-3,
    model_save_path="./crnn_weights.pth",
    log_file="./training_log.txt",
    device=torch.device("cpu"),
    train_ratio=0.8,
    plot_errors=True,
    verbose=True
):
    # Prepare data
    if verbose:
        print("Preparing data...")
    chars = extract_charset(json_annotations_path)
    char2idx, idx2char, blank_idx, num_classes = build_vocab(chars)

    # Split data into training and validation sets
    train_receipts, val_receipts = split_receipts(json_annotations_path, train_ratio=train_ratio)
    train_samples = build_samples(json_annotations_path, train_receipts)
    val_samples   = build_samples(json_annotations_path, val_receipts)

    # Build datasets and dataloaders
    train_dataset = OCRDataset(train_samples, char2idx)
    val_dataset   = OCRDataset(val_samples, char2idx)

    train_loader, val_loader = build_dataloaders(train_dataset, val_dataset, batch_size=batch_size)
    if verbose:
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}")

    # Initialize model
    if verbose:
        print("Initializing model...")
    model = CRNN(num_classes=num_classes)
    model.to(device)
    if verbose:
        print(f"Model parameters: {model.count_parameters()}")

    # Defining Loss and optimizer
    criterion = nn.CTCLoss(blank=blank_idx, zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Logging setup
    if os.path.exists(log_file):
        os.remove(log_file)

    def log(msg):
        with open(log_file, "a") as f:
            f.write(msg + "\n")

    # Training loop
    train_losses = []
    val_losses = []
    epoch_times = []

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        if verbose:
            print(f"Epoch {epoch}/{epochs}")
        model.train()
        epoch_loss = 0.0

        for images, targets, target_lengths, _ in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # (T, B, C)

            input_lengths = torch.full(
                size=(images.size(0),),
                fill_value=outputs.size(0),
                dtype=torch.long
            ).to(device)

            loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * images.size(0)

        epoch_loss /= len(train_dataset)
        train_losses.append(epoch_loss)
        log(f"[{datetime.now()}] Epoch {epoch}/{epochs} - Train loss: {epoch_loss:.4f}")

        # Validation
        model.eval()
        val_loss_total = 0.0

        with torch.no_grad():
            for images, targets, target_lengths, _ in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)

                outputs = model(images)

                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=outputs.size(0),
                    dtype=torch.long
                ).to(device)

                loss = criterion(outputs.log_softmax(2), targets, input_lengths, target_lengths)
                val_loss_total += loss.item() * images.size(0)

        val_loss_total /= len(val_dataset)
        val_losses.append(val_loss_total)
        log(f"[{datetime.now()}] Epoch {epoch}/{epochs} - Validation loss: {val_loss_total:.4f}")

        end_time = time.time()
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)
        avg_time = sum(epoch_times) / len(epoch_times)
        remaining_time = avg_time * (epochs - epoch)
        remaining_time_format = time.strftime('%H:%M:%S', time.gmtime(remaining_time))

        if verbose:
            print(f"Epoch {epoch}/{epochs} - Training Loss: {epoch_loss:.4f} - Validation loss: {val_loss_total:.4f}")
            print(f"Epoch duration: {epoch_duration:.2f}s - Average: {avg_time:.2f}s - ETA: {remaining_time_format}")
        # Save model after each epoch
        torch.save(model.state_dict(), model_save_path)

    # Plot losses
    if plot_errors:
        plt.figure()
        plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
        plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("CTC Loss")
        plt.title("CRNN Training and Validation Loss")
        plt.legend()
        plt.savefig("reports/figures/loss_plot.png")
        plt.show()

    log("Training finished and model saved.")
    if verbose:
        print("Training finished and model saved.")
    return model, train_losses, val_losses


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(
        json_annotations_path="data/filename_to_word_files.json",
        batch_size=8,
        epochs=50,
        lr=1e-3,
        model_save_path="./crnn_weights.pth",
        log_file="./training_log.txt",
        device=device,
        train_ratio=0.8,
        plot_errors=True,
        verbose=True
    )
