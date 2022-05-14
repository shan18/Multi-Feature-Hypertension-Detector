import torch
from torch import nn


class HypertensionDetectorConvGRU(nn.Module):
    def __init__(self, rnn_hidden_dim, seq_meta_len, n_layers, fc_dim, dropout):
        super().__init__()

        self.arch = 'cnn_gru'
        self.rnn_hidden_dim = rnn_hidden_dim
        self.fc_dim = fc_dim
        self.seq_meta_len = seq_meta_len
        self.n_layers = n_layers
        self.dropout = dropout

        self.conv1 = self._create_conv_sequence(1, 32, 3, dropout)
        self.conv2 = self._create_conv_sequence(32, 64, 3, dropout)
        self.conv3 = self._create_conv_sequence(64, 128, 3, dropout)
        self.pool = nn.MaxPool1d(2)

        self.seq_meta_fc = nn.Linear(seq_meta_len, rnn_hidden_dim)

        self.rnn = nn.GRU(
            128, rnn_hidden_dim, num_layers=n_layers, bidirectional=True,
            dropout=dropout if n_layers > 1 else 0
        )

        self.fc1 = nn.Linear(2 * n_layers * rnn_hidden_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, 1)

    def _create_conv_sequence(self, input_dim, output_dim, kernel_size, dropout):
        return nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding='same'),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, seq, seq_meta):
        """Input shapes

        seq: [batch_size, seq_len]
        seq_meta: [batch_size, seq_meta_len]
        """

        batch_size, seq_len = seq.shape

        seq = seq.unsqueeze(1)  # [batch_size, 1, seq_len]

        features = self.conv1(seq)  # [batch_size, 32, seq_len]
        features = self.pool(features)
        features = self.conv2(features)  # [batch_size, 64, seq_len / 2]
        features = self.pool(features)
        features = self.conv3(features)  # [batch_size, 128, seq_len / 4]
        features = features.permute(2, 0, 1)  # [seq_len / 4, batch_size, 128]

        seq_meta = self.seq_meta_fc(seq_meta)  # [batch_size, rnn_hidden_dim]
        seq_meta = seq_meta.unsqueeze(0).repeat(self.n_layers * 2, 1, 1)  # [n_layers * 2, batch_size, rnn_hidden_dim]

        _, hidden = self.rnn(
            features, seq_meta
        )  # [2 * num_layers, batch_size, rnn_hidden_dim]

        hidden = hidden.permute(1, 0, 2).reshape(batch_size, -1)  # [batch_size, 2 * num_layers * rnn_hidden_dim]

        output = self.fc1(hidden)  # [batch_size, fc_dim]
        output = self.fc2(output)  # [batch_size, 1]

        return output
