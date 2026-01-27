import os, sys
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import extract_charset, build_vocab

class CRNN(nn.Module):
    """
    Lightweight CRNN for OCR with CTC loss, with VGG-like CNN backbone.
    - 3 Conv + Max Pooling layers
    - 1 BiLSTM layer (Bidirectional LSTM)
    - Fully connected output layer with num_classes outputs
    """
    def __init__(self, num_classes, lstm_hidden_size=256, lstm_layers=1, dropout=0.3):
        super(CRNN, self).__init__()
        
        self.dropout_rate = dropout

        # Step 1. CNN Backbone
        # input: (B, 1, 128, 128)
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # 64x64
            nn.Dropout2d(dropout),  # Dropout après premier bloc

            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # 32x32
            nn.Dropout2d(dropout),  # Dropout après deuxième bloc

            nn.Conv2d(64, 128, 3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),  # 16x16
            nn.Dropout2d(dropout),  # Dropout après troisième bloc
        )

        # Step 2. Compute dimensions after CNN for RNN input
        # height after CNN = 16
        # width after CNN = 16 → sequence length T = 16
        self.cnn_output_height = 128 // (2*2*2)  # 16
        self.cnn_output_channels = 128
        self.seq_length = 16  # width after pooling

        # Step 3. RNN/LSTM layer
        # Input size = channels * height (flatten height)
        self.rnn_input_size = self.cnn_output_channels * self.cnn_output_height
        self.lstm_hidden_size = lstm_hidden_size

        self.lstm = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0  # Dropout entre couches LSTM (si > 1 couche)
        )
        
        # Dropout après LSTM
        self.lstm_dropout = nn.Dropout(dropout)

        # Step 4. Fully connected output layer
        self.fc = nn.Linear(lstm_hidden_size*2, num_classes) # *2 for bidirectional

    def forward(self, x):
        """
        x: (B, 1, H, W)
        returns: logits (T, B, num_classes) as expected by nn.CTCLoss
        """
        B = x.size(0)

        # CNN Backbone
        x = self.cnn(x)  # (B, C, H, W) -> (B, 128, 16, 16)

        # Prepare for RNN
        x = x.permute(0, 3, 1, 2)  # (B, W, C, H)
        x = x.contiguous().view(B, x.size(1), -1)  # (B, T=16, C*H=128*16)

        # RNN
        x, _ = self.lstm(x)  # (B, T, hidden*2)
        
        # Dropout après LSTM
        x = self.lstm_dropout(x)

        # Output layer
        x = self.fc(x)  # (B, T, num_classes)

        # For CTCLoss, need (T, B, C)
        x = x.permute(1, 0, 2)  # (T, B, num_classes)

        return x


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    
    
    
    
if __name__ == "__main__":
    
    json_annotations_path = "data/filename_to_word_files.json"
    chars = extract_charset(json_annotations_path)
    
    char2idx, idx2char, blank_idx, num_classes = build_vocab(chars)
    print(f"Number of classes (including blank): {num_classes}")
    
    model = CRNN(num_classes=num_classes)
    print(f"Model has {model.count_parameters()} trainable parameters.")