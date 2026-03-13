import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    # Requires 32 x W image input (Width 'W' can be any dynamic size)
    def __init__(self, img_channel, num_class, rnn_hidden=256):
        super(CRNN, self).__init__()

        # --- 1. CNN Feature Extractor ---
        self.cnn = nn.Sequential(
            # Block 1 (batch, img_channel, 32, W)
            nn.Conv2d(img_channel, 64, kernel_size=3, stride=1, padding=1), # -> (Batch, 64, 32, W)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                          # -> (Batch, 64, 16, W/2)

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),         # -> (Batch, 128, 16, W/2)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                          # -> (Batch, 128, 8, W/4)

            # Block 3 (no maxpool)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),        # -> (Batch, 256, 8, W/4)
            nn.ReLU(inplace=True),

            # Block 4
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),        # -> (Batch, 256, 8, W/4)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),# -> (Batch, 256, 4, W/4 + 1)

            # Block 5
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),        # -> (Batch, 512, 4, W/4 + 1)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),

            # Block 6
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),        # -> (Batch, 512, 4, W/4 + 1)
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=512),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 1), padding=(0, 1)),# -> (Batch, 512, 2, W/4 + 2)

            # Block 7
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),        # -> (Batch, 512, 1, W/4 + 1)
            nn.ReLU(inplace=True),                                          
        )

        # --- 2. RNN Sequence Reader ---
        self.rnn1 = nn.LSTM(input_size=512, hidden_size=rnn_hidden, bidirectional=True, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=rnn_hidden * 2, hidden_size=rnn_hidden, bidirectional=True, batch_first=True)

        # --- 3. Final Classifier ---
        self.fc = nn.Linear(rnn_hidden * 2, num_class)
    

    @staticmethod
    def compute_seq_len(width):
        """Compute CNN output sequence length from input image width.
        Must be updated if the CNN pooling/stride architecture changes.
        """
        return width // 2 // 2 + 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Conv layers (elevate raw input into feature space)
        # Input x shape: (Batch, 1, 32, W)
        feature_maps: torch.Tensor = self.cnn(x) 
        # feature_maps shape: (Batch, 512, 1, W_pooled)
        
        # 2. Map to sequential: (batch, channels, height, width) -> (batch, width, channels*height)
        batch_size, channels, height, width = feature_maps.size()
        
        sequential_feature_maps: torch.Tensor = feature_maps.permute(0, 3, 1, 2).reshape(batch_size, width, channels*height)

        # 3. RNN layers (contextualize the feature space)
        contextual_features, _ = self.rnn1(sequential_feature_maps)
        contextual_features, _ = self.rnn2(contextual_features)

        # 4. Linear layer (map to class predictions)
        out = self.fc(contextual_features)
        
        # 5. Log Softmax (Required for CTCLoss)
        # out shape remains: (Batch, W_pooled, num_class)
        out = F.log_softmax(out, dim=2)
        
        return out