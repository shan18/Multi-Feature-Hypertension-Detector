import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()

        self.conv1 = self._create_conv_sequence(input_dim, output_dim, 128, dropout)
        self.conv2 = self._create_conv_sequence(input_dim, output_dim, 256, dropout)
        self.conv3 = self._create_conv_sequence(input_dim, output_dim, 512, dropout)

        self.pointwise = nn.Conv1d(output_dim * 3, output_dim, 1)
    
    def _create_conv_sequence(self, input_dim, output_dim, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, seq):
        features1 = self.conv1(seq)
        features2 = self.conv2(seq)
        features3 = self.conv3(seq)

        features = torch.cat((features1, features2, features3), dim=1)

        features = self.pointwise(features)

        return features


class HypertensionDetectorConvGRU(nn.Module):
    def __init__(self, feature_dim, hidden_dim, seq_meta_len, n_layers, dropout):
        super().__init__()

        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.seq_meta_fc = nn.Linear(seq_meta_len, hidden_dim)

        self.conv1 = ConvBlock(1, feature_dim, dropout=dropout)
        self.conv2 = ConvBlock(feature_dim, feature_dim, dropout=dropout)
        self.conv3 = ConvBlock(feature_dim, feature_dim, dropout=dropout)
        self.pool = nn.MaxPool1d(2)

        self.rnn = nn.GRU(feature_dim, hidden_dim, num_layers=n_layers, bidirectional=True, dropout=dropout)

        self.fc1 = nn.Linear(2 * n_layers * hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, seq, seq_meta):
        """Input shapes

        seq: [batch_size, seq_len]
        seq_meta: [batch_size, seq_meta_len]
        """

        batch_size, seq_len = seq.shape

        seq = seq.unsqueeze(1)  # [batch_size, 1, seq_len]

        features = self.conv1(seq)  # [batch_size, feature_dim, seq_len]
        features = self.pool(features)
        features = self.conv2(features)  # [batch_size, feature_dim, seq_len / 2]
        features = self.pool(features)
        features = self.conv3(features)  # [batch_size, feature_dim, seq_len / 4]

        features = features.permute(2, 0, 1)  # [seq_len / 4, batch_size, feature_dim]

        seq_meta = self.seq_meta_fc(seq_meta)  # [batch_size, hidden_dim]
        seq_meta = seq_meta.unsqueeze(0).repeat(self.n_layers * 2, 1, 1)  # [n_layers * 2, batch_size, hidden_dim]

        _, hidden = self.rnn(
            features, seq_meta
        )  # [2 * num_layers, batch_size, hidden_dim]

        hidden = hidden.permute(1, 0, 2).reshape(batch_size, -1)  # [batch_size, 2 * num_layers * hidden_dim]

        output = self.fc1(hidden)  # [batch_size, hidden_dim]
        output = self.fc2(output)  # [batch_size, 1]
    
        return output
